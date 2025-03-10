from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

    def calculate_similarities(self, query_vector, top_n):
        """Calculates similarities between the query vector and the TF-IDF matrix."""
        if self.tfidf_matrix is None or self.documents is None:
            print("Model is not built or loaded.")
            return []

        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        top_indices = similarities.argsort()[-top_n:][::-1]
        return [
            {
                "id": self.documents[idx]["id"],
                "text": self.documents[idx]["search_content"],
                "similarity_score": similarities[idx],
            }
            for idx in top_indices
        ]

    def find_most_similar(self, search_terms, top_n):
        """Finds the most similar documents for the given search terms."""
        if self.model is None or self.tfidf_matrix is None or self.documents is None:
            print("Model is not built or loaded.")
            return []

        for term in search_terms:
            query_vector = self.model.transform([term])
            query_results = self.calculate_similarities(query_vector, top_n)

        print("Results obtained for TF-IDF.")

        return query_results