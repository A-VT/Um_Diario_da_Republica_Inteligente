from utils.retriever.model_type import ModelType
from utils.retriever.retriever_word2vec import Word2VecRetriever
from utils.retriever.retriever_tfidf import TfidfRetriever
from utils.retriever.retriever_bm25 import BM25Retriever
from utils.retriever.retriever_wiki_word2vec import WikiWord2VecRetriever
from utils.mongo_conn import connect_to_mongo
from utils.retriever.process_queries import preprocess_query
from dotenv import load_dotenv
from enum import Enum
import os

from utils.json_file_handler import JSONFileHandler
from utils.progress_messenger import ProgressMessenger
from flask_sse import sse  # Import Flask-SSE for broadcasting

class IRSystem:
    def __init__(self):
        self.client, self.db, self.collection_dados, self.collection_metadados = self._connect_to_db()
        self.documents = self._prep_model()

    def _connect_to_db(self):
        load_dotenv()
        cred_mongo_user, cred_mongo_password = os.getenv("MONGO_USER"), os.getenv("MONGO_PASSWORD")
        return connect_to_mongo(cred_mongo_user, cred_mongo_password)

    def _fetch_documents(self):
        try:
            return list(self.collection_metadados.find({}, {"_id": 1, "db_ID": 1, "Titulo": 1, "Sumario": 1}))
        except Exception as e:
            print(f"Error fetching documents: {e}")
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

    def _init_retrievers(self, user_models):
        self.n_models = len(user_models)
        retrievers = {}

        model_to_retriever = {
            ModelType.TF_IDF: TfidfRetriever,
            ModelType.WORD2VEC: Word2VecRetriever,
            ModelType.BM25: BM25Retriever,
            ModelType.WIKI_WORD2VEC: WikiWord2VecRetriever
        }

        for model_type in user_models:
            retriever_class = model_to_retriever.get(model_type)
            if retriever_class:
                retrievers[model_type] = retriever_class()
            else:
                print(f"Invalid model: {model_type}. Skipping...")

        return retrievers

    def _retrieve_or_create_models(self, retrievers):
        for model_type, retriever in retrievers.items():

            retriever.load_model()
            
            if retriever.model is None:
                print(f"Building model for {model_type}...")
                retriever.build_model(self.documents)
                retriever.save_model()

        return retrievers

    def search(self, user_query, user_models, user_autokeywords, user_nres):
        print("[IR] Selecting documents...")
        self.n_results = user_nres

        retrievers = self._init_retrievers(user_models)
        retrievers = self._retrieve_or_create_models(retrievers)

        search_terms = set()
        search_terms.update(preprocess_query(user_query, user_autokeywords))
        self.search_terms = list(set(search_terms))

        file_handler = JSONFileHandler("IR/results/search_terms.json")
        file_handler.delete_results()
        file_handler.save_results(results=self.search_terms)

        results = []

        for model_type, retriever in retrievers.items():
            
            if model_type == ModelType.TF_IDF:
                temp_results = retriever.find_most_similar(self.search_terms, user_nres)
            elif model_type == ModelType.BM25:
                temp_results = retriever.find_most_similar(self.search_terms, user_nres)
            elif model_type == ModelType.WORD2VEC:
                temp_results = retriever.find_most_similar(self.search_terms, self.documents, user_nres)
            elif model_type == ModelType.WIKI_WORD2VEC:
                temp_results = retriever.find_most_similar(self.search_terms, self.documents, user_nres)
            else:
                print(f"Cannot handle model {model_type}")
                continue
            
            # Aggregate results
            results = self._add_results(results, temp_results, model_type)

        # Balance results by average score and return the top results
        self._global_balance_results(results)
        return results

    def _add_results(self, results, temp_results, model_type):
        results_dict = {i["id"]: i for i in results}
        for temp in temp_results:
            if temp["id"] in results_dict:
                results_dict[temp["id"]]["similarity_score"][model_type.value] = temp["similarity_score"]
            else:
                temp["similarity_score"] = {model_type.value: temp["similarity_score"]}
                results_dict[temp["id"]] = temp
            temp["terms"] = self.search_terms
        return list(results_dict.values())

    def _global_balance_results(self, results):
        for item in results:
            scores = item.get("similarity_score", {}).values()
            item["global_average_score"] = sum(scores) / len(scores) if scores else 0
            item["global_confidence"] = len(scores) / self.n_models
            item["global_balanced_score"] = item["global_average_score"] * item["global_confidence"]

            for key in ["scores", "avg_score", "confidence"]:
                item.pop(key, None)
        results.sort(key=lambda x: x["global_balanced_score"], reverse=True)
        return results[:self.n_results]

    def get_result_ids(self, results):
        return [result["db_ID"] for result in results if "db_ID" in result]
