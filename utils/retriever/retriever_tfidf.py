from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.json_file_handler import JSONFileHandler
from collections import defaultdict
import joblib
import os

class TfidfRetriever:
    def __init__(self, model_file="./IR/models/tfidf_model.pkl"):
        self.model_file = model_file
        self.model = None
        self.tfidf_matrix = None
        self.documents = None
        self.corpus = None

    def build_model(self, documents):
        """Builds the TF-IDF model using the provided documents."""
        self.documents = documents
        self.corpus = [doc["search_content"] for doc in documents]
        self.model = TfidfVectorizer()
        self.tfidf_matrix = self.model.fit_transform(self.corpus)

    def save_model(self):
        """Saves the TF-IDF model and associated data to a file."""
        try:
            os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
            joblib.dump((self.model, self.tfidf_matrix, self.documents), self.model_file)
            print(f"Model saved to {self.model_file}.")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self):
        """Loads the TF-IDF model and associated data from a file."""
        if not os.path.exists(self.model_file):
            print(f"Model file {self.model_file} does not exist. A new model will be created.")
            return
        else:
            try:
                self.model, self.tfidf_matrix, self.documents = joblib.load(self.model_file)
                print(f"Model and documents loaded from {self.model_file}.")
            except Exception as e:
                print(f"Error loading model: {e}")

    def calculate_similarities(self, query_vector, top_n, search_terms):
        """Calculates similarities between the query vector and the TF-IDF matrix."""
        if self.tfidf_matrix is None or self.documents is None:
            print("Model is not built or loaded.")
            return []

        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        top_indices = similarities.argsort()[-top_n:][::-1]
        return [
            {
                "id": self.documents[idx]["id"],
                "db_ID": self.documents[idx]["db_ID"],
                "text": self.documents[idx]["search_content"],
                "similarity_score": similarities[idx],
                "terms": search_terms
            }
            for idx in top_indices
        ]
    
    def _balance_results(self, query_results):
        merged_results = defaultdict(lambda: {
            "id": None,
            "db_ID": None,
            "text": None,
            "terms": [],
            "similarity_scores": []
        })

        # Flatten the nested list and merge entries
        for result_list in query_results:
            for item in result_list:
                doc_id = item["id"]
                merged = merged_results[doc_id]

                # Set static values if not already set
                if merged["id"] is None:
                    merged["id"] = item["id"]
                    merged["db_ID"] = item["db_ID"]
                    merged["text"] = item["text"]

                # Add similarity score and search term
                merged["similarity_scores"].append(item["similarity_score"])
                merged["terms"].append(item["terms"])

        # Final balanced result with averaged similarity score
        balanced = []
        for doc in merged_results.values():
            avg_score = sum(doc["similarity_scores"]) / len(doc["similarity_scores"])
            confidence = len(doc["similarity_scores"]) / self.n_terms
            similarity_score = avg_score * confidence
            balanced.append({
                "id": doc["id"],
                "db_ID": doc["db_ID"],
                "text": doc["text"],
                "terms": doc["terms"],
                "scores": doc["similarity_scores"],
                "avg_score": avg_score,
                "confidence": confidence,
                "similarity_score": similarity_score
            })
        balanced.sort(key=lambda x: x["similarity_score"], reverse=True)
        return balanced

    def find_most_similar(self, search_terms, top_n):
        """Finds the most similar documents for the given search terms."""
        self.n_terms = len(search_terms)

        if self.model is None or self.tfidf_matrix is None or self.documents is None:
            print("Model is not built or loaded.")
            return []

        full_query = " ".join(search_terms)
        full_query = " ".join(dict.fromkeys(full_query.split()))

        query_vector = self.model.transform([full_query])
        query_results = self.calculate_similarities(query_vector, top_n, search_terms)

        #query_results = self._balance_results(full_results)

        print("Results obtained for TF-IDF.")

        output_file = "IR_analysis/parl_europeu" #IR/results
        file_handler = JSONFileHandler(f"{output_file}/tfidf_results.json")
        file_handler.delete_results()
        file_handler.save_results(results=query_results)

        return query_results