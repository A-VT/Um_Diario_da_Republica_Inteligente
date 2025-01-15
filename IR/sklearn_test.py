from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.mongo_conn import *
from utils.process_queries import preprocess_query
import joblib
import json
import os

MODEL_FILE = "./IR/models/tfidf_model.pkl"
OUTPUT_FILE = "./IR/results/tf-idf_sklearn.json"
TYPE_MODEL = "TF-IDF_SIMIL"
N_RESULTS = 100

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
        titulo = doc.get("Titulo", "").strip()
        sumario = doc.get("Sumario", "").strip()
        combined_text = f"{titulo}. {sumario}"
        processed_docs.append({
            "id": str(doc["_id"]),
            "search_content": combined_text
        })
    return processed_docs

def build_tfidf_model(documents):
    corpus = [doc["search_content"] for doc in documents]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix

def save_model(vectorizer, tfidf_matrix, documents):
    try:
        os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
        joblib.dump((vectorizer, tfidf_matrix, documents), MODEL_FILE)
        print(f"Model and documents saved to {MODEL_FILE}.")
    except Exception as e:
        print(f"Error saving model: {e}")

def load_model():
    if not os.path.exists(MODEL_FILE):
        print(f"Model file {MODEL_FILE} does not exist. A new model will be created.")
        return None, None, None
    try:
        vectorizer, tfidf_matrix, documents = joblib.load(MODEL_FILE)
        print(f"Model and documents loaded from {MODEL_FILE}.")
        return vectorizer, tfidf_matrix, documents
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None


def find_most_similar(query, vectorizer, tfidf_matrix, documents, top_n=5):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            "id": documents[idx]["id"],
            "text": documents[idx]["search_content"],
            "similarity_score": similarities[idx]
        })
    return results

def save_results(results):
    try:
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4, ensure_ascii=False)
        print(f"Results saved to {OUTPUT_FILE}")
    except Exception as e:
        print(f"Error saving results to file: {e}")

def main():

    #TF-IDF Similarity
    if (TYPE_MODEL == "TF-IDF_SIMIL"):  
        vectorizer, tfidf_matrix, documents = load_model()
        if vectorizer is None or tfidf_matrix is None or documents is None:
            client, db, collection_dados, collection_metadados = connect_to_mongo()
            if not client:
                return
            
            raw_documents = fetch_documents(collection_metadados)
            documents = preprocess_documents(raw_documents)
            print(f"Sample of processed documents: {documents[:5]}")

            vectorizer, tfidf_matrix = build_tfidf_model(documents)
            save_model(vectorizer, tfidf_matrix, documents)
    else:
        print("Cannot handle that")
        return

    # Query input
    query = input("Enter your question: ")
    search_terms = preprocess_query(query)
        
    results = find_most_similar(query, vectorizer, tfidf_matrix, documents, top_n=N_RESULTS)

    save_results(results)

if __name__ == "__main__":
    main()
