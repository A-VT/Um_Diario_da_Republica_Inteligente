import faiss
import numpy as np
import os
import joblib
from utils.json_file_handler import JSONFileHandler
from sentence_transformers import SentenceTransformer
from collections import defaultdict


class SbertRetriever: #using L2 distance
    def __init__(self, directory_answers:str, model_file="./IR/models/faiss_index", embedding_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):  #sentence-transformers/all-MiniLM-L6-v2
        self.model_file = model_file
        self.directory_answers = directory_answers
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.model = None
        self.documents = None
        self.doc_id_map = {}
        self.dimension = None

    def _embed_text(self, text):
        return self.embedding_model.encode([text])[0]

    def build_model(self, documents):
        self.documents = documents
        embeddings = []

        for i, doc in enumerate(documents):
            embedding = self._embed_text(doc["search_content"])
            if self.dimension is None:
                self.dimension = len(embedding)
            embeddings.append(embedding)
            self.doc_id_map[i] = doc

        embeddings_np = np.array(embeddings).astype("float32")
        self.model = faiss.IndexFlatIP(self.dimension) #IndexFlatL2
        self.model.add(embeddings_np)
        print(f"FAISS index built with {len(embeddings_np)} vectors.")

    def save_model(self):
        try:
            os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
            faiss.write_index(self.model, self.model_file + ".index")
            joblib.dump((self.documents, self.doc_id_map, self.dimension), self.model_file + ".meta")
            print(f"Index saved to {self.model_file}.index and metadata to {self.model_file}.meta")
        except Exception as e:
            print(f"Error saving FAISS index: {e}")

    def load_model(self):
        try:
            if not os.path.exists(self.model_file + ".index"):
                print("Index file does not exist. You need to build it first.")
                return
            self.model = faiss.read_index(self.model_file + ".index")
            self.documents, self.doc_id_map, self.dimension = joblib.load(self.model_file + ".meta")
            print("FAISS index and metadata loaded.")
        except Exception as e:
            print(f"Error loading FAISS index: {e}")

    def find_most_similar(self, search_terms, top_n):
        if self.model is None or self.documents is None:
            print("Index not loaded or built.")
            return []

        query_text = " ".join(dict.fromkeys(search_terms))
        query_embedding = self._embed_text(query_text).astype("float32").reshape(1, -1)

        D, I = self.model.search(query_embedding, top_n)  # D: distances, I: indices

        distances = D[0]
        indices = I[0]

        min_dist = np.min(distances)
        max_dist = np.max(distances)
        range_dist = max_dist - min_dist if max_dist != min_dist else 1.0  # avoid division by zero


        results = []
        for i, idx in enumerate(I[0]): #I[0]
            doc = self.doc_id_map[idx]
            norm_score = 1.0 - ((distances[i] - min_dist) / range_dist)
            results.append({
                "id": doc["id"],
                "db_ID": doc["db_ID"],
                "text": doc["search_content"],
                "similarity_score": float(norm_score),  # invert distance for similarity feel - -D[0][i]
                "terms": search_terms
            })

        print("Results obtained for FAISS.")
        file_handler = JSONFileHandler(f"{self.directory_answers}/faiss_results.json")
        file_handler.delete_results()
        file_handler.save_results(results=results)

        return results
