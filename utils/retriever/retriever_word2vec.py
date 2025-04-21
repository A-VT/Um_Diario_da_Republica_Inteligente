from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from utils.json_file_handler import JSONFileHandler
from collections import defaultdict
from gensim.models import Word2Vec
import spacy
import os
import numpy as np
import logging

class Word2VecRetriever:
    def __init__(self, model_file="./IR/models/word2vec_model.model"):
        self.nlp = spacy.load("pt_core_news_md")
        self.model_file = model_file
        self.model = None
        self.documents = None
        self.corpus = None
        self.document_vectors = {}

    def _tokenize(self, text):
        """Tokenizes text using spaCy's Portuguese model."""
        doc = self.nlp(text.lower())
        return [token.text for token in doc if token.is_alpha and not token.is_stop]

    def build_model(self, documents, vector_size=25, window=3, min_count=1, workers=4):
        logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
        self.documents = documents
        self.corpus = [doc["search_content"] for doc in documents]
        tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]
        self.model = Word2Vec(
            sentences=tokenized_corpus,
            sg=0,
            vector_size=vector_size,
            min_count=min_count,
            workers=workers,
            window=window,
            epochs=20,
            compute_loss=True
        )
        self._cache_document_vectors()

    def _cache_document_vectors(self):
        """Precompute and store document vectors."""
        if self.model is None or self.documents is None:
            return
        self.document_vectors = {
            doc["id"]: np.mean(
                [self.model.wv[token] for token in self._tokenize(doc["search_content"]) if token in self.model.wv] or [np.zeros(self.model.vector_size)],
                axis=0
            ) for doc in self.documents
        }

    def model_evaluation(self, show_examples=True):
        if self.model is None or self.corpus is None:
            print("Model or corpus not loaded.")
            return

        try:
            final_loss = self.model.get_latest_training_loss()
            print(f"\nFinal Training Loss: {final_loss:.4f}")
            vocab_size = len(self.model.wv)
            print(f"Vocabulary Size: {vocab_size} tokens")

            tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]
            all_tokens = set(token for doc in tokenized_corpus for token in doc)
            in_vocab = [token for token in all_tokens if token in self.model.wv]
            coverage = len(in_vocab) / len(all_tokens) * 100 if all_tokens else 0
            print(f"Vocabulary Coverage: {coverage:.2f}%")

            print("\nSemantic Similarity Checks:")
            word_pairs = [
                ("governo", "estado"),
                ("empresa", "negócio"),
                ("candidato", "eleição"),
                ("juiz", "tribunal"),
                ("multar", "coima")
            ]
            for w1, w2 in word_pairs:
                if w1 in self.model.wv and w2 in self.model.wv:
                    sim = self.model.wv.similarity(w1, w2)
                    print(f"   - ({w1}, {w2}): {sim:.4f}")
                else:
                    print(f"   - ({w1}, {w2}): [words not in vocab]")

            if show_examples:
                print("\nTop 10 Similar Words:")
                example_words = ["cargos", "empresa", "eleição", "privado", "coima"]
                for word in example_words:
                    if word in self.model.wv:
                        similar = self.model.wv.most_similar(word, topn=5)
                        print(f"   - {word}: {[f'{w} ({s:.2f})' for w, s in similar]}")
                    else:
                        print(f"   - {word}: [not in vocab]")

        except Exception as e:
            print(f"Error during evaluation: {e}")

    def save_model(self):
        try:
            os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
            self.model.save(self.model_file)
            print(f"Model saved to {self.model_file}.")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self):
        if not os.path.exists(self.model_file):
            print(f"Model file {self.model_file} does not exist. A new model will be created.")
            return
        try:
            self.model = Word2Vec.load(self.model_file)
            print(f"Model loaded from {self.model_file}.")
            if self.documents:
                self._cache_document_vectors()
        except Exception as e:
            print(f"Error loading model: {e}")

    def calculate_similarities_for_term(self, search_term, top_n):
        if self.model is None or not self.document_vectors:
            print("Model or document vectors not initialized.")
            return []

        tokenized_query = self._tokenize(search_term)
        valid_tokens = [self.model.wv[token] for token in tokenized_query if token in self.model.wv]
        if not valid_tokens:
            return []

        query_vector = np.mean(valid_tokens, axis=0).reshape(1, -1)
        query_vector = normalize(query_vector)

        doc_ids = list(self.document_vectors.keys())
        doc_matrix = np.array([self.document_vectors[doc_id] for doc_id in doc_ids])
        doc_matrix = normalize(doc_matrix)

        similarity_scores = cosine_similarity(query_vector, doc_matrix)[0]

        similarities = []
        for i, score in enumerate(similarity_scores):
            doc = next(doc for doc in self.documents if doc["id"] == doc_ids[i])
            similarities.append({
                "id": doc["id"],
                "db_ID": doc["db_ID"],
                "text": doc["search_content"],
                "similarity_score": float(score),
                "terms": search_term
            })

        return sorted(similarities, key=lambda x: x["similarity_score"], reverse=True)[:top_n]
    
    def calculate_similarity_for_query(self, query_text, top_n):
        """
        Calculate cosine similarities between query vector and all document vectors.
        `query_text` is expected to be a string containing multiple terms.
        """
        if self.model is None or not self.document_vectors:
            print("Model or document vectors not initialized.")
            return []

        tokenized_query = self._tokenize(query_text)
        valid_tokens = [self.model.wv[token] for token in tokenized_query if token in self.model.wv]
        if not valid_tokens:
            return []

        query_vector = np.mean(valid_tokens, axis=0).reshape(1, -1)
        query_vector = normalize(query_vector)

        doc_ids = list(self.document_vectors.keys())
        doc_matrix = np.array([self.document_vectors[doc_id] for doc_id in doc_ids])
        doc_matrix = normalize(doc_matrix)

        similarity_scores = cosine_similarity(query_vector, doc_matrix)[0]

        similarities = []
        for i, score in enumerate(similarity_scores):
            doc = next(doc for doc in self.documents if doc["id"] == doc_ids[i])
            similarities.append({
                "id": doc["id"],
                "db_ID": doc["db_ID"],
                "text": doc["search_content"],
                "similarity_score": float(score),
                "terms": query_text
            })

        return sorted(similarities, key=lambda x: x["similarity_score"], reverse=True)[:top_n]


    def _balance_results(self, query_results):
        merged_results = defaultdict(lambda: {
            "id": None,
            "db_ID": None,
            "text": None,
            "terms": [],
            "similarity_scores": []
        })

        for result_list in query_results:
            for item in result_list:
                doc_id = item["id"]
                merged = merged_results[doc_id]
                if merged["id"] is None:
                    merged["id"] = item["id"]
                    merged["db_ID"] = item["db_ID"]
                    merged["text"] = item["text"]
                merged["similarity_scores"].append(item["similarity_score"])
                merged["terms"].append(item["terms"])

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

    def find_most_similar(self, search_terms, documents, top_n):
        self.n_terms = len(search_terms)
        
        if self.documents is None:
            self.documents = documents
            self._cache_document_vectors()

        """
        query_results = []
        for term in search_terms:
            results = self.calculate_similarities(term, top_n)
            if results:
                query_results.append(results)

        balanced = self._balance_results(query_results)
        """

        full_query = " ".join(search_terms)
        full_query = " ".join(dict.fromkeys(full_query.split()))
        results = self.calculate_similarity_for_query(full_query, top_n)

        
        print("Results obtained for Word2Vec.")


        file_handler = JSONFileHandler("IR/results/word2vec_results.json")
        file_handler.delete_results()
        file_handler.save_results(results=results)

        return results

