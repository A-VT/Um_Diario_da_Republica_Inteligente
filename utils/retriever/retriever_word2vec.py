from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
import os
import numpy as np

nltk.download('punkt')

class Word2VecRetriever:
    def __init__(self, model_file="./IR/models/word2vec_model.model"):
        self.model_file = model_file
        self.model = None

    def build_model(self, documents, vector_size=100, window=5, min_count=1, workers=4):
        """Builds the Word2Vec model using the provided documents."""
        corpus = [doc["search_content"] for doc in documents]
        tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]
        self.model = Word2Vec(sentences=tokenized_corpus, vector_size=vector_size, window=window, min_count=min_count, workers=workers)

    def save_model(self):
        """Saves the Word2Vec model to a file."""
        try:
            os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
            self.model.save(self.model_file)
            print(f"Model saved to {self.model_file}.")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self):
        """Loads the Word2Vec model from a file."""
        try:
            if os.path.exists(self.model_file):
                self.model = Word2Vec.load(self.model_file)
                print(f"Model loaded from {self.model_file}.")
            else:
                print(f"Model file not found at {self.model_file}.")
        except Exception as e:
            print(f"Error loading Word2Vec model: {e}")

    def _calculate_similarity(self, query_vector, document):
        """Calculates the cosine similarity between the query vector and a document vector."""
        doc_tokens = word_tokenize(document["search_content"].lower())
        valid_tokens = [self.model.wv[token] for token in doc_tokens if token in self.model.wv]

        if valid_tokens:
            # You can use a weighted average of word vectors or another strategy
            doc_vector = np.mean(valid_tokens, axis=0)  # Mean of word vectors
        else:
            doc_vector = np.zeros(self.model.vector_size)  # Zero vector if no valid tokens

        return cosine_similarity([query_vector], [doc_vector])[0][0]

    def find_most_similar(self, search_terms, documents, top_n=5):
        """Finds the most similar documents for the given search terms."""
        if self.model is None:
            print("Model is not loaded or built.")
            return []

        tokens = word_tokenize(search_terms.lower())
        valid_tokens = [self.model.wv[token] for token in tokens if token in self.model.wv]

        if valid_tokens:
            query_vector = np.mean(valid_tokens, axis=0)  # Mean of word vectors for the query
        else:
            query_vector = np.zeros(self.model.vector_size)  # Zero vector if no valid tokens

        similarities = [
            {
                "id": doc["id"],
                "text": doc["search_content"],
                "similarity_score": float(self._calculate_similarity(query_vector, doc))  # Convert to float
            }
            for doc in documents
        ]

        similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
        top_results = similarities[:top_n]

        # Return the structured result as per the desired JSON format
        return [{
            "query": search_terms,
            "results": top_results
        }]
