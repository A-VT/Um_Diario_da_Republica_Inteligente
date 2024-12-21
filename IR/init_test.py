from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
from dotenv import load_dotenv
import os

def connect_to_mongo(mongo_user, mongo_password):
    try:
        mongo_uri = f"mongodb+srv://{mongo_user}:{mongo_password}@drbd.bbw8o.mongodb.net/?retryWrites=true&w=majority&appName=DRbd"
        client = MongoClient(mongo_uri)
        db = client['DiarioRepublica']
        collection = db['metadados']
        print("Connected to MongoDB.")
        return client, db, collection
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None, None, None

def fetch_documents(collection):
    try:
        documents = list(collection.find({}, {"Titulo": 1, "Sumario": 1}))
        return documents
    except Exception as e:
        print(f"Error fetching documents: {e}")
        return []

def preprocess_documents(documents):
    processed_docs = []
    for doc in documents:
        titulo = doc.get("Titulo", "")
        sumario = doc.get("Sumario", "")
        combined_text = f"{titulo}. {sumario}"
        processed_docs.append({
            "id": str(doc["_id"]),
            "text": combined_text
        })
    return processed_docs

def build_tfidf_model(documents):
    texts = [doc["text"] for doc in documents]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix

def find_most_similar(query, vectorizer, tfidf_matrix, documents, top_n=5):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            "id": documents[idx]["id"],
            "text": documents[idx]["text"],
            "similarity_score": similarities[idx]
        })
    return results

def main():

    load_dotenv()
    mongo_user = os.getenv('MONGO_USER')
    mongo_password = os.getenv('MONGO_PASSWORD')

    client, db, collection = connect_to_mongo(mongo_user, mongo_password)
    if not client:
        return

    documents = fetch_documents(collection)
    processed_documents = preprocess_documents(documents)


    vectorizer, tfidf_matrix = build_tfidf_model(processed_documents)

    query = input("Enter your question: ")
    

    top_n = 20  # Number of documents to retrieve
    results = find_most_similar(query, vectorizer, tfidf_matrix, processed_documents, top_n=top_n)

    # Print results
    print("\nTop matching documents:")
    for result in results:
        print(f"ID: {result['id']}")
        print(f"Text: {result['text']}")
        print(f"Similarity Score: {result['similarity_score']:.4f}\n")

if __name__ == "__main__":
    main()
