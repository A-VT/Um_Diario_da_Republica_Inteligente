from rank_bm25 import BM25Okapi
from collections import defaultdict
import joblib
import json
import os

from utils.json_file_handler import JSONFileHandler

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
        else:
            try:
                self.model, self.documents = joblib.load(self.model_file)
                print(f"Model and documents loaded from {self.model_file}.")
            except Exception as e:
                print(f"Error loading model: {e}")

    def calculate_similarities(self, search_terms, top_n):
        """Calculates similarities between the query and the BM25 model."""
        if self.model is None or self.documents is None:
            print("Model is not built or loaded.")
            return []

        #tokenized_query = search_term.split()

        tokenized_query = search_terms
        if isinstance(search_terms, str): #a string an not a list of
            tokenized_query = tokenized_query.split()

        scores = self.model.get_scores(tokenized_query)

        # Normalization of results - 0 to 1
        min_score, max_score = min(scores), max(scores)

        if max_score == min_score:
            normalized_scores = [1.0] * len(scores)
        else:
            normalized_scores = [(s - min_score) / (max_score - min_score) for s in scores]

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]

        return [
                {
                    "id": self.documents[idx]["id"],
                    "db_ID": self.documents[idx]["db_ID"],
                    "text": self.documents[idx]["search_content"],
                    "similarity_score": normalized_scores[idx],
                    "terms": tokenized_query
                }
                for idx in top_indices
            ]

    def find_most_similar(self, search_terms, top_n):
        """Finds the most similar documents for the given search terms."""
        self.n_terms = len(search_terms)

        if self.model is None or self.documents is None:
            print("Model is not built or loaded.")
            return []

        query_results = self.calculate_similarities(search_terms, top_n)

        print("Results obtained for BM25.")

        file_handler = JSONFileHandler("IR/results/bm25_results.json")
        file_handler.delete_results()
        file_handler.save_results(results=query_results)

        return query_results