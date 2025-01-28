from enum import Enum
from utils.mongo_conn import *
from utils.retriever.process_queries import preprocess_query
from utils.retriever.retriever_tfidf import TfidfRetriever
from utils.retriever.retriever_word2vec import Word2VecRetriever
from utils.retriever.retriever_bm25 import BM25Retriever
import json
import os


class ModelType(Enum):
    TF_IDF_SIMIL = "TF-IDF_SIMIL"
    WORD2VEC_SIMIL = "WORD2VEC_SIMIL"
    BM25_SIMIL = "BM25_SIMIL"


TYPE_MODEL = ModelType.WORD2VEC_SIMIL
SEARCH_W_KEYWORDS = False
OUTPUT_DIRECTORY_PATH = "./IR/results/"
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


def save_results(results, filename):
    try:
        pathfile = OUTPUT_DIRECTORY_PATH + filename
        os.makedirs(os.path.dirname(pathfile), exist_ok=True)
        with open(pathfile, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4, ensure_ascii=False)
        print(f"Results saved to {pathfile}")
    except Exception as e:
        print(f"Error saving results to file: {e}")


def prep_model():
    client, db, collection_dados, collection_metadados = connect_to_mongo()
    if not client:
        return

    raw_documents = fetch_documents(collection_metadados)
    documents = preprocess_documents(raw_documents)
    return documents


def main():
    retriever = None
    if TYPE_MODEL == ModelType.TF_IDF_SIMIL:
        retriever = TfidfRetriever()
    elif TYPE_MODEL == ModelType.WORD2VEC_SIMIL:
        retriever = Word2VecRetriever()
    elif TYPE_MODEL == ModelType.BM25_SIMIL:
        retriever = BM25Retriever()
    else:
        print("Cannot handle this model")
        return
    

    documents = prep_model()

    retriever.load_model()
    if retriever.model is None:
        retriever.build_model(documents)
        retriever.save_model()

    query = input("Enter your question: ")
    if not SEARCH_W_KEYWORDS:
        search_terms = query
    else:
        max_keywords = max(round(len(query) * 0.2), 1)
        search_terms = preprocess_query(query, max_keywords)
    
    print(f'search_terms: {search_terms}')

    if TYPE_MODEL == ModelType.TF_IDF_SIMIL:
        results = retriever.find_most_similar(search_terms, top_n=N_RESULTS, search_with_keywords=SEARCH_W_KEYWORDS)
        save_results(results, "tf_idf.json")
    elif TYPE_MODEL == ModelType.WORD2VEC_SIMIL:
        results = retriever.find_most_similar(search_terms, documents, top_n=N_RESULTS)
        save_results(results, "word2vec.json")
    elif TYPE_MODEL == ModelType.BM25_SIMIL:
        results = retriever.find_most_similar(search_terms, top_n=N_RESULTS, search_with_keywords=SEARCH_W_KEYWORDS)
        save_results(results, "bm25.json")
    else:
        print("Cannot handle this model")
        return

if __name__ == "__main__":
    main()
