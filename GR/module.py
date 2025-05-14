import os
import requests
import faiss
import numpy as np
from dotenv import load_dotenv
from enum import Enum
from bson.objectid import ObjectId
from bson.errors import InvalidId
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from utils.json_file_handler import JSONFileHandler
from utils.mongo_conn import connect_to_mongo

class ModelType(Enum):
    QWEN2_5__14b = "t3q-qwen2.5-14b-v1.0-e3"

class GRSystem:
    def __init__(self, prog_messenger, model=ModelType.QWEN2_5__14b, max_tokens=6000,
                 list_doc_ids=None, k=10, chunk_size=200, chunk_overlap=50,
                 directory_answers="GR/results"):
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

        self.parent_docs = self._get_contents()
        self.child_chunks, self.child_to_parent = self._chunk_documents(self.parent_docs)
        self.embeddings = self.retriever.encode(self.child_chunks, convert_to_tensor=False)
        self.index = self._build_faiss_index(self.embeddings)

    def _connect_to_db(self):
        return connect_to_mongo(self.mongo_user, self.mongo_password)

    def _get_contents(self):
        client, db, collection_dados, _ = self._connect_to_db()
        docs = []
        for doc_id in self.list_doc_ids:
            try:
                oid = ObjectId(doc_id)
                res = collection_dados.find_one({"_id": oid})
                if res and res.get("Content"):
                    docs.append(res["Content"])
                else:
                    self.prog_messenger.send(f"No content for ID: {doc_id}")
            except Exception as e:
                self.prog_messenger.send(f"Error fetching {doc_id}: {e}")
        JSONFileHandler(f"{self.directory_answers}/docs").save_results(docs)
        return docs

    def _chunk_documents(self, documents):
        child_chunks, child_to_parent = [], []
        for idx, doc in enumerate(documents):
            tokens = doc.split()
            for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
                chunk = " ".join(tokens[i:i + self.chunk_size])
                child_chunks.append(chunk)
                child_to_parent.append(idx)
        return child_chunks, child_to_parent

    def _build_faiss_index(self, embeddings):
        dim = embeddings.shape[1] if hasattr(embeddings, 'shape') else len(embeddings[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings).astype("float32"))
        return index

    def _count_tokens(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _retrieve_parent_docs(self, query):
        vec = self.retriever.encode([query])[0]
        D, I = self.index.search(np.array([vec]).astype("float32"), self.k)
        parents = {self.child_to_parent[i] for i in I[0]}
        return [self.parent_docs[i] for i in parents]

    def _build_context(self, parent_docs):
        max_model_ctx = 4096
        reserve = 1000
        available = max_model_ctx - reserve
        context_texts, total = [], 0
        for doc in parent_docs:
            tok = self._count_tokens(doc)
            if total + tok > available:
                break
            context_texts.append(doc)
            total += tok
        return "\n\n".join(context_texts)

    def get_summaries(self, user_query):
        print("[GR] Generating with token-aware Parent-Document RAG...")
        parents = self._retrieve_parent_docs(user_query)
        context = self._build_context(parents)

        q = user_query if "?" in user_query else f"O que sabes sobre o tema '{user_query}'?"
        prompt = f"""
És um assistente jurídico altamente qualificado, especializado em análise legal detalhada.

Deverás responder de forma exaustiva e meticulosa, fornecendo:
- Todas as definições legais relevantes
- Citações completas de normas, artigos e cláusulas
- Interpretações jurídicas e implicações práticas
- Distinções entre normas parecidas
- Jurisprudência relevante (se aplicável)
- Exemplos concretos, quando apropriado

Mantém uma linguagem técnica e clara, em Português de Portugal. Não assumes nem inventas factos não contidos no contexto.
Se necessário, a resposta poderá conter tags html para headers, bold, italics e underline. 

Contexto:
{context}

Pergunta: {q}"""

        JSONFileHandler(f"{self.directory_answers}/llm_content.json").save_results(prompt)
        headers = {"Content-Type": "application/json"}
        data = {"model": self.model.value,
                "messages": [{"role": "system", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": self.max_tokens,
                "top_p": 0.95,
                "stream": False}
        res = requests.post(self.LLM_url, json=data, headers=headers)
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"].strip()
        return f"Error: {res.status_code} - {res.text}"
