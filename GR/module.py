import os
import requests
import faiss
import numpy as np
from dotenv import load_dotenv
from enum import Enum
from bson.objectid import ObjectId
from bson.errors import InvalidId
from sentence_transformers import SentenceTransformer

from utils.json_file_handler import JSONFileHandler
from utils.mongo_conn import connect_to_mongo
from transformers import AutoTokenizer

class ModelType(Enum):
    QWEN2_5__14b = "t3q-qwen2.5-14b-v1.0-e3"


class GRSystem:
    def __init__(self, prog_messenger, model=ModelType.QWEN2_5__14b, max_tokens=6000, list_doc_ids=None, k=10, chunk_size=200, chunk_overlap=50, directory_answers="GR/results"):
        self.prog_messenger = prog_messenger

        load_dotenv()
        self.LLM_url = os.getenv("API_ENDPOINT")
        self.mongo_user = os.getenv("MONGO_USER")
        self.mongo_password = os.getenv("MONGO_PASSWORD")

        self.model = model
        self.max_tokens = max_tokens
        self.k = k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.list_doc_ids = list_doc_ids or []
        self.directory_answers = directory_answers

        self.retriever = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.tokenizer = AutoTokenizer.from_pretrained("JungZoona/T3Q-qwen2.5-14b-v1.0-e3")

        # Step 1: Load and chunk documents
        raw_docs = self._get_contents()
        self.chunks = self._chunk_documents(raw_docs)
        self.embeddings = self.retriever.encode(self.chunks, convert_to_tensor=False)
        self.index = self._build_faiss_index(self.embeddings)

    def _connect_to_db(self):
        return connect_to_mongo(self.mongo_user, self.mongo_password)

    def _get_contents(self):
        client, db, collection_dados, _ = self._connect_to_db()
        docs = []

        for doc_id in self.list_doc_ids:
            try:
                object_id = ObjectId(doc_id)
                result = collection_dados.find_one({"_id": object_id})
                content = result.get("Content") if result else None
                if content:
                    docs.append(content)
                else:
                    print(f"[WARN] No content found for ID: {doc_id}")
                    self.prog_messenger.send(f"No content found for ID: {doc_id}")
            except (InvalidId, Exception) as e:
                print(f"[ERROR] {doc_id} - {e}")
                self.prog_messenger.send(f"{doc_id} - {e}")

        # Optional save
        handler = JSONFileHandler(f"{self.directory_answers}/docs")
        handler.delete_results()
        handler.save_results(docs)
        return docs

    def _chunk_documents(self, documents):
        chunks = []
        for doc in documents:
            tokens = doc.split()
            for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
                chunk = " ".join(tokens[i:i + self.chunk_size])
                chunks.append(chunk)
        return chunks
    
    def _retrieve_chunks(self, query):
        query_vec = self.retriever.encode([query])[0]
        D, I = self.index.search(np.array([query_vec]).astype("float32"), self.k)
        return [self.chunks[i] for i in I[0]]

    def _build_faiss_index(self, embeddings):
        dim = len(embeddings[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings).astype("float32"))
        return index
    
    def _count_tokens(text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def get_summaries(self, user_query):
        print("[GR] Preparing LLM response...")

        relevant_chunks = self._retrieve_chunks(user_query)
        context = "\n\n".join(relevant_chunks)

        processed_user_query = ""
        if "?" in user_query:
            processed_user_query = user_query
        else:
            processed_user_query = f"O que sabes sobre o tema '{user_query}'?"

        system_prompt = f"""
És um assistente jurídico altamente qualificado.

Deverás responder de forma detalhada à pergunta dada, de forma estruturada e profunda, e utilizando o contexto legal fornecido. Inclui:

- Definições legais relevantes
- Citações completas de artigos e cláusulas
- Interpretações e implicações legais
- Diferenças entre normas similares
- Jurisprudência relevante (se aplicável)

Mantém uma linguagem técnica e clara, em Português de Portugal.

Contexto:
{context}

Pergunta: {processed_user_query}"""

        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model.value,
            "messages": [{"role": "system", "content": system_prompt}],
            "temperature": 0.7,
            "max_tokens": self.max_tokens,
            "top_p": 0.95,
            "stream": False
        }

        file_handler = JSONFileHandler(f"{self.directory_answers}/llm_content.json")
        file_handler.delete_results()
        file_handler.save_results(results=system_prompt)

        response = requests.post(self.LLM_url, json=data, headers=headers)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            return f"Error: {response.status_code} - {response.text}"
