from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import numpy as np
import os

class WikiWord2VecRetriever:
    def __init__(self, model_file="./IR/models/model_300_20_sg.wv"):
        self.nlp = spacy.load("pt_core_news_md")
        self.model_file = model_file
        self.model = None
        self.documents = None
    
    def load_model(self):
        """Loads the Word2Vec model from the given file."""
        if not os.path.exists(self.model_file):
            raise FileNotFoundError(f"Model file {self.model_file} does not exist.")
        try:
            self.model = KeyedVectors.load(self.model_file, mmap='r')
            print(f"Model loaded from {self.model_file}.")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def tokenize(self, text):
        """Tokenizes text using spaCy's Portuguese model."""
        doc = self.nlp(text.lower())
        return [token.text for token in doc if token.is_alpha and not token.is_stop]
    
    def calculate_similarities(self, search_term, top_n):
        """Calculates similarities between the query and stored documents."""
        if self.model is None or self.documents is None:
            print("Model is not loaded or documents are missing.")
            return []

        tokenized_query = self.tokenize(search_term)
        valid_tokens = [self.model[token] for token in tokenized_query if token in self.model]
        query_vector = np.mean(valid_tokens, axis=0) if valid_tokens else np.zeros(self.model.vector_size)

        similarities = []
        for doc in self.documents:
            doc_tokens = self.tokenize(doc["search_content"])
            doc_vectors = [self.model[token] for token in doc_tokens if token in self.model]
            doc_vector = np.mean(doc_vectors, axis=0) if doc_vectors else np.zeros(self.model.vector_size)

            similarity_score = cosine_similarity([query_vector], [doc_vector])[0][0]

            similarities.append({
                "id": doc["id"],
                "text": doc["search_content"],
                "similarity_score": float(similarity_score),
            })

        return sorted(similarities, key=lambda x: x["similarity_score"], reverse=True)[:top_n]

    def find_most_similar(self, search_terms, documents, top_n):
        """Finds the most similar documents for the given search terms."""
        if self.documents is None:
            self.documents = documents

        if self.model is None or self.documents is None:
            print("Model is not loaded or documents are missing.")
            return []

        results = []
        for term in search_terms:
            results.extend(self.calculate_similarities(term, top_n))
        print("Results obtained for Word2Vec.")
        return results
