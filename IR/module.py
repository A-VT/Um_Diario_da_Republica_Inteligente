from enum import Enum
from utils.mongo_conn import *
from utils.retriever.process_queries import preprocess_query
from utils.retriever.retriever_tfidf import TfidfRetriever
from utils.retriever.retriever_word2vec import Word2VecRetriever
from utils.retriever.retriever_bm25 import BM25Retriever
from dotenv import load_dotenv
import json
import os

load_dotenv()

class ModelType(Enum):
    TF_IDF_SIMIL = "TF-IDF_SIMIL"
    WORD2VEC_SIMIL = "WORD2VEC_SIMIL"
    BM25_SIMIL = "BM25_SIMIL"


TYPE_MODEL = [ModelType.WORD2VEC_SIMIL, ModelType.TF_IDF_SIMIL, ModelType.BM25_SIMIL]
SEARCH_W_KEYWORDS = False
OUTPUT_PATH = "./IR/results/results.json"
N_RESULTS = 25

def fetch_documents(collection):
    try:
        documents = list(collection.find({}, {"_id": 1, "Titulo": 1, "Sumario": 1}))
        return documents
    except Exception as e:
        print(f"Error fetching documents: {e}")
        return []


def preprocess_documents(documents):
    processed_docs = []
    for doc in documents:
        searchable_content = doc.get("Sumario", "").strip()
        if len(searchable_content) == 0:
            searchable_content = doc.get("Titulo", "").strip()
        combined_text = f"{searchable_content}"
        processed_docs.append({
            "id": str(doc["_id"]),
            "search_content": combined_text
        })
    return processed_docs

def save_results(results):
    try:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4, ensure_ascii=False)
        print(f"Results saved to {OUTPUT_PATH}")
    except Exception as e:
        print(f"Error saving results to file: {e}")

def delete_results():
    try:
        if os.path.exists(OUTPUT_PATH):
            os.remove(OUTPUT_PATH)
            print(f"Previous results in '{OUTPUT_PATH}' have been deleted.")
        else:
            print(f"Previous results not found.")
    except Exception as e:
        print(f"Error deleting results file: {e}")


def read_results():
        """Reads and returns data from a JSON file."""
        try:
            with open(OUTPUT_PATH, 'r', encoding='utf-8') as file:
                data = json.load(file)
            return data
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error reading file: {e}")
            return []

def prep_model():
    load_dotenv()
    cred_mongo_user, cred_mongo_password = os.getenv("MONGO_USER"), os.getenv("MONGO_PASSWORD")

    client, db, collection_dados, collection_metadados = connect_to_mongo(cred_mongo_user, cred_mongo_password)
    if not client:
        return

    raw_documents = fetch_documents(collection_metadados) #both 'summaries' and 'titles'
    documents = preprocess_documents(raw_documents) #searchable content only - mostly summaries; but when missing titles 
    return documents


def add_results(results, temp_results, model_type:ModelType):
    results_dict = {i["id"]: i for i in results}
    
    for temp in temp_results:
        if temp["id"] in results_dict:
            results_dict[temp["id"]]["similarity_score"][model_type.value] = temp["similarity_score"]
        else:
            temp["similarity_score"] = {model_type.value: temp["similarity_score"]}
            results_dict[temp["id"]] = temp
    
    res = list(results_dict.values())
    return res

def balance_results(results):
    for item in results:
        scores = item.get("similarity_score", {}).values()
        item["average_score"] = sum(scores) / len(scores) if scores else 0
    
    results.sort(key=lambda x: x["average_score"], reverse=True)
    return results[:N_RESULTS]

def main():
    retrievers = {}

    for model_type in TYPE_MODEL:
        if model_type == ModelType.TF_IDF_SIMIL:
            retrievers[model_type] = TfidfRetriever()
        elif model_type == ModelType.WORD2VEC_SIMIL:
            retrievers[model_type] = Word2VecRetriever()
        elif model_type == ModelType.BM25_SIMIL:
            retrievers[model_type] = BM25Retriever()
        else:
            print(f"Cannot handle model: {model_type}")
            return
    
    if not retrievers:
            return

    documents = prep_model()

    for model_type, retriever in retrievers.items():
        retriever.load_model()
        if retriever.model is None:
            retriever.build_model(documents)
            retriever.save_model()

    query = input("Enter your question: ")

    search_terms = [query]
    if SEARCH_W_KEYWORDS:
        search_terms.append(preprocess_query(query))
    print(f'search_terms: {search_terms}')

    results = []
    for model_type, retriever in retrievers.items():
        if model_type  == ModelType.TF_IDF_SIMIL:
            temp_results = retriever.find_most_similar(search_terms, N_RESULTS)
        elif model_type == ModelType.WORD2VEC_SIMIL:
            temp_results = retriever.find_most_similar(search_terms, documents, N_RESULTS)
        elif model_type == ModelType.BM25_SIMIL:
            temp_results = retriever.find_most_similar(search_terms, N_RESULTS)
        else:
            print("Cannot handle model")
            return
             
        results = add_results(results, temp_results, model_type)

    balance_results(results)
    save_results(results)

if __name__ == "__main__":
    main()
