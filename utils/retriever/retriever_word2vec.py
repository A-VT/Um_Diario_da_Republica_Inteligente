from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import spacy
import os
import numpy as np

class Word2VecRetriever:
    def __init__(self, model_file="./IR/models/word2vec_model.model"):
        self.nlp = spacy.load("pt_core_news_md")
        self.model_file = model_file
        self.model = None
        self.documents = None
        self.corpus = None

    def _tokenize(self, text):
        """Tokenizes Portuguese text using spaCy and removes stopwords."""
        doc = self.nlp(text.lower())
        return [token.text for token in doc if token.is_alpha and token.text not in self.nlp.Defaults.stop_words]

    def build_model(self, documents, vector_size=100, window=5, min_count=1, workers=4):
        """Builds the Word2Vec model using Portuguese documents."""
        self.documents = documents
        self.corpus = [doc["search_content"] for doc in documents]
        tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]
        self.model = Word2Vec(sentences=tokenized_corpus, sg=0, vector_size=vector_size, min_count=min_count, workers=workers, epochs=5) #window=window

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
        if not os.path.exists(self.model_file):
            print(f"Model file {self.model_file} does not exist. A new model will be created.")
            return
        else:
            try:
                self.model = Word2Vec.load(self.model_file)
                print(f"Model and documents loaded from {self.model_file}.")
            except Exception as e:
                print(f"Error loading model: {e}")

    def _tokenize(self, text):
        """Tokenizes text using spaCy's Portuguese model."""
        doc = self.nlp(text)
        return [token.text for token in doc if not token.is_stop and not token.is_punct]

    def calculate_similarities(self, search_term, documents, top_n):
        """Calculates similarities between the query and the Word2Vec model."""
        if self.model is None or self.documents is None:
            print("Model is not built or loaded.")
            return []

        tokenized_query = self._tokenize(search_term)

        valid_tokens = [self.model.wv[token] for token in tokenized_query if token in self.model.wv]

        query_vector = np.mean(valid_tokens, axis=0) if valid_tokens else np.zeros(self.model.vector_size)

        similarities = []
        for doc in documents:
            doc_tokens = self._tokenize(doc["search_content"])
            doc_vectors = [self.model.wv[token] for token in doc_tokens if token in self.model.wv]

            doc_vector = np.mean(doc_vectors, axis=0) if doc_vectors else np.zeros(self.model.vector_size)

            similarity_score = cosine_similarity([query_vector], [doc_vector])[0][0]

            similarities.append({
                "id": doc["id"],
                "text": doc["search_content"],
                "similarity_score": float(similarity_score),
            })

        top_results = sorted(similarities, key=lambda x: x["similarity_score"], reverse=True)[:top_n]
        return top_results

    def find_most_similar(self, search_terms, documents, top_n):
        """Finds the most similar documents for the given Portuguese search terms."""
        if self.documents == None:
            self.documents = documents

        if self.model is None or self.documents is None:
            print("Model is not built or loaded.")
            return []

        for term in search_terms:
            query_results = self.calculate_similarities(term, self.documents, top_n)
        print("Results obtained for Word2vec.")

        return query_results
    
    def __del__(self):
        """Destructor to clean up resources."""
        if hasattr(self, 'nlp') and self.nlp is not None:
            del self.nlp
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'documents'):
            self.documents = None
            self.corpus = None
        self.model_file = None
        print(f"Instance of {self.__class__.__name__} cleaned up.")