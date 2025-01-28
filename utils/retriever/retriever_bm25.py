from rank_bm25 import BM25Okapi
import joblib
import os

class BM25Retriever:
    def __init__(self, model_file="./IR/models/bm25_model.pkl"):
        self.model_file = model_file
        self.model = None
        self.documents = None
        self.corpus = None

    def build_model(self, documents):
        """Builds the BM25 model using the provided documents."""
        self.documents = documents
        self.corpus = [doc["search_content"].split() for doc in documents]
        self.model = BM25Okapi(self.corpus)

    def save_model(self):
        """Saves the BM25 model and associated data to a file."""
        try:
            os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
            joblib.dump((self.model, self.documents), self.model_file)
            print(f"Model saved to {self.model_file}.")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self):
        """Loads the BM25 model and associated data from a file."""
        if not os.path.exists(self.model_file):
            print(f"Model file {self.model_file} does not exist. A new model will be created.")
            return
        try:
            self.model, self.documents = joblib.load(self.model_file)
            print(f"Model and documents loaded from {self.model_file}.")
        except Exception as e:
            print(f"Error loading model: {e}")

    def calculate_similarities(self, query, top_n):
        """Calculates similarities between the query and the BM25 model."""
        if self.model is None or self.documents is None:
            print("Model is not built or loaded.")
            return []

        tokenized_query = query.split()
        scores = self.model.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        return [
            {
                "id": self.documents[idx]["id"],
                "text": self.documents[idx]["search_content"],
                "similarity_score": scores[idx],
            }
            for idx in top_indices
        ]

    def find_most_similar(self, search_terms, top_n=5, search_with_keywords=False):
        """Finds the most similar documents for the given search terms."""
        if self.model is None or self.documents is None:
            print("Model is not built or loaded.")
            return []

        results = []

        if search_with_keywords:
            for query in search_terms:
                query_results = self.calculate_similarities(query, top_n)
                results.append({"query": query, "results": query_results})
        else:
            query_results = self.calculate_similarities(search_terms, top_n)
            results.append({"query": search_terms, "results": query_results})

        return results