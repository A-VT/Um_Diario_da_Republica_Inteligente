from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from utils.json_file_handler import JSONFileHandler
from collections import defaultdict
import spacy
import numpy as np
import os

class WikiWord2VecRetriever:
    def __init__(self, model_file="./IR/models/model_300_20_sg.wv"):
        self.nlp = spacy.load("pt_core_news_md")
        self.model_file = model_file
        self.model = None
        self.documents = None
        self.document_vectors = {}
        self.n_terms = 1

    def _tokenize(self, text):
        doc = self.nlp(text.lower())
        return [token.text for token in doc if token.is_alpha and not token.is_stop]

    def load_model(self):
        if not os.path.exists(self.model_file):
            raise FileNotFoundError(f"Model file {self.model_file} does not exist.")
        try:
            self.model = KeyedVectors.load(self.model_file, mmap='r')
            print(f"Model loaded from {self.model_file}.")
        except Exception as e:
            print(f"Error loading model: {e}")

    def _cache_document_vectors(self):
        if self.model is None or self.documents is None:
            return
        self.document_vectors = {
            doc["id"]: np.mean(
                [self.model[token] for token in self._tokenize(doc["search_content"]) if token in self.model] or [np.zeros(self.model.vector_size)],
                axis=0
            ) for doc in self.documents
        }

    def calculate_similarities_for_term(self, search_term, top_n):
        if self.model is None or not self.document_vectors:
            print("Model or document vectors not initialized.")
            return []

        tokenized_query = self._tokenize(search_term)
        valid_tokens = [self.model[token] for token in tokenized_query if token in self.model]
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

    def calculate_similarity_for_query(self, query_text, top_n):
        if self.model is None or not self.document_vectors:
            print("Model or document vectors not initialized.")
            return []

        tokenized_query = self._tokenize(query_text)
        valid_tokens = [self.model[token] for token in tokenized_query if token in self.model]
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


    def find_most_similar(self, search_terms, documents, top_n):
        self.n_terms = len(search_terms)
        self.documents = documents

        if self.model is None or self.documents is None:
            print("Model is not loaded or documents are missing.")
            return []

        self._cache_document_vectors()

        full_query = " ".join(search_terms)
        full_query = " ".join(dict.fromkeys(full_query.split()))

        results = self.calculate_similarity_for_query(full_query, top_n)
                

        #balanced = self._balance_results(query_results)
        print("Results obtained for Wiki_Word2Vec.")

        output_file = "IR_analysis/parl_europeu" #IR/results
        file_handler = JSONFileHandler(f"{output_file}/wiki_word2vec_results.json")
        file_handler.delete_results()
        file_handler.save_results(results=results)

        return results
