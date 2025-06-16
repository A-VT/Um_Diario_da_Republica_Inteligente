from utils.retriever.model_type import ModelType
from utils.retriever.retriever_word2vec import Word2VecRetriever
from utils.retriever.retriever_tfidf import TfidfRetriever
from utils.retriever.retriever_bm25 import BM25Retriever
from utils.retriever.retriever_wiki_word2vec import WikiWord2VecRetriever
from utils.retriever.retriever_sbert import SbertRetriever

from utils.mongo_conn import connect_to_mongo
from utils.retriever.process_queries import preprocess_query
from utils.IR_direct_querying.IR_mongo_query import mongo_text_search
from utils.IR_direct_querying.IR_elastic_query import elastic_query_search
from collections import defaultdict
from dotenv import load_dotenv
import os

from utils.progress_messenger import ProgressMessenger
from utils.llm_communicator import LLMCommunicator
from utils.json_file_handler import JSONFileHandler


class IRSystem:
    def __init__(self, directory_answers, prog_messenger:ProgressMessenger, llm_client:LLMCommunicator):
        self.prog_messenger = prog_messenger
        self.llm = llm_client

        self.client, self.db, self.collection_dados, self.collection_metadados = self._connect_to_db()
        self.documents = self._prep_model()
        self.directory_answers = directory_answers
        

    def _connect_to_db(self):
        load_dotenv()
        cred_mongo_user, cred_mongo_password = os.getenv("MONGO_USER"), os.getenv("MONGO_PASSWORD")
        return connect_to_mongo(cred_mongo_user, cred_mongo_password)

    def _fetch_documents(self):
        try:
            return list(self.collection_metadados.find({}, {"_id": 1, "db_ID": 1, "Titulo": 1, "Sumario": 1}))
        except Exception as e:
            print(f"Error fetching documents: {e}")
            self.prog_messenger.send(f"Error fetching documents: {e}")
            return []

    def _preprocess_documents(self, documents):
        processed_docs = []
        for doc in documents:
            searchable_content = doc.get("Sumario", "").strip() or doc.get("Titulo", "").strip()
            processed_docs.append({
                "id": str(doc["_id"]),
                "db_ID": str(doc["db_ID"]),
                "search_content": searchable_content
            })
        return processed_docs

    def _prep_model(self):
        raw_documents = self._fetch_documents()
        return self._preprocess_documents(raw_documents)

    def _init_retrievers(self, user_models:list):
        self.n_models = len(user_models)
        retrievers = {}

        model_to_retriever = {
            ModelType.TF_IDF: TfidfRetriever,
            ModelType.WORD2VEC: Word2VecRetriever,
            ModelType.BM25: BM25Retriever,
            ModelType.WIKI_WORD2VEC: WikiWord2VecRetriever,
            ModelType.SBERT: SbertRetriever,
        }

        for model_type in user_models:
            retriever_class = model_to_retriever.get(model_type)
            if retriever_class:
                retrievers[model_type] = retriever_class(directory_answers=self.directory_answers)
            else:
                print(f"Invalid model: {model_type}. Skipping...")

        return retrievers

    def _retrieve_or_create_models(self, retrievers):
        for model_type, retriever in retrievers.items():

            self.prog_messenger.send(f"Processing with {model_type} model...")
            retriever.load_model()
            
            if retriever.model is None:
                self.prog_messenger.send(f"Building model for {model_type} model...")

                retriever.build_model(self.documents)
                retriever.save_model()
                self.prog_messenger.send(f"Processing with {model_type} model...")

        return retrievers
    
    def _create_analysis_file(self, name_file, data):
        file_handler = JSONFileHandler(f"{self.directory_answers}/{name_file}")
        file_handler.delete_results()
        file_handler.save_results(results=data)

    def search(self, user_query, user_models, user_autokeywords, user_nres):
        print("[IR] Selecting documents...")
        self.prog_messenger.send(f"Selecting documents...")

        self.n_results = user_nres

        retrievers = self._init_retrievers(user_models)
        retrievers = self._retrieve_or_create_models(retrievers)

        self.search_terms = None
        if user_autokeywords == True:
            search_terms = set()
            search_terms.update(preprocess_query(user_query, user_autokeywords))
            self.search_terms = list(set(search_terms))
        else:
            self.search_terms = user_query

        self._create_analysis_file("search_terms.json", self.search_terms)

        #results = []
        all_model_results = {}
        for model_type, retriever in retrievers.items():
            
            #self.prog_messenger.send(f"Gathering results for {model_type} model.")

            if model_type in {ModelType.TF_IDF, ModelType.BM25, ModelType.SBERT}:
                temp_results = retriever.find_most_similar(self.search_terms, self.n_results)
            elif model_type in {ModelType.WORD2VEC, ModelType.WIKI_WORD2VEC}:
                temp_results = retriever.find_most_similar(self.search_terms, self.documents, self.n_results)            
            else:
                print(f"Cannot handle model {model_type}")
                self.prog_messenger.send(f"Cannot handle model {model_type}.")
                continue

            
            self.prog_messenger.send(f"Received temporary results from {model_type}.")
            
            # Aggregate results
            all_model_results[model_type.name] = temp_results
            #results = self._add_results(results, temp_results, model_type)

        # Balance results by average score and return the top results
        results = self._aggregate_results(all_model_results, self.n_results, user_models)
        #self._global_balance_results(results)
        #self.prog_messenger.send(f"Balanced final results.")
        

        #comparative searches
        self._mongo_direct_querying()
        self._elastic_direct_querying()

        return results
    

    def _compute_weights(self, user_models):
        model_names = sorted([m.name for m in user_models])

        if len(model_names) == 1:
            return {model_names[0]: 1.0}

        else:
            sparse_models = {"TF_IDF", "BM25"}
            mid_dense_models = {"WIKI_WORD2VEC", "WORD2VEC"}
            high_dense_models = {"SBERT"}

            weights = {}
            total = 0

            for model in model_names:
                if model in sparse_models:
                    weights[model] = 1.0
                elif model in mid_dense_models:
                    weights[model] = 0.8
                elif model in high_dense_models:
                    weights[model] = 1.2
                else:
                    weights[model] = 1.0

                total += weights[model]

            for model in weights: # Normalization to sum to 1
                weights[model] /= total

            return weights


    def _aggregate_results(self, all_model_results, top_k, user_models, weights=None):
        weights = self._compute_weights(user_models)
        
        scores_by_id = defaultdict(lambda: {
            "normalized_scores": {},
            "terms": None,
            "text": None,
            "db_ID": None,
            "id": None
        })

        for model_name, model_results in all_model_results.items():
            for r in model_results:
                doc_id = r["id"]
                scores_by_id[doc_id]["text"] = r.get("text")
                scores_by_id[doc_id]["normalized_scores"][model_name] = r["similarity_score"]
                scores_by_id[doc_id]["db_ID"] = r.get("db_ID")
                scores_by_id[doc_id]["terms"] = r.get("terms")

        aggregated = []
        for doc_id, entry in scores_by_id.items():
            scores = entry["normalized_scores"]
            terms = entry["terms"]
            db_ID = entry["db_ID"]
            text = entry["text"]
            if weights:
                weighted_avg = sum(scores[m] * weights.get(m, 1) for m in scores) / sum(weights.get(m, 1) for m in scores)
            else:
                weighted_avg = sum(scores.values()) / len(scores)

            confidence = len(scores) / len(all_model_results)
            final_score = weighted_avg * confidence

            aggregated.append({
                "id": doc_id,
                "db_ID": db_ID,
                "text": text,
                "global_score": final_score,
                "confidence": confidence,
                "individual_scores": scores,
                "terms": terms
            })

        return sorted(aggregated, key=lambda x: x["global_score"], reverse=True)[:top_k]

    def get_result_ids(self, results):
        return [result["db_ID"] for result in results if "db_ID" in result]

    def _mongo_direct_querying(self):
        self.prog_messenger.send(f"Performing direct MongoDB search...")

        mongo_results = mongo_text_search(
            collection_metadados=self.collection_metadados,
            search_terms=self.search_terms,
            n_docs=self.n_results
        )
        self._create_analysis_file("mongo_direct_querying.json", mongo_results)

    def _elastic_direct_querying(self):
        self.prog_messenger.send(f"Performing an Elastic search...")

        elastic_results = elastic_query_search(
            search_terms=self.search_terms,
            n_docs=self.n_results
        )
        self._create_analysis_file("elastic_direct_querying.json", elastic_results)
