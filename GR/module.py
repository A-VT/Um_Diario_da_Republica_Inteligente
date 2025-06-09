# GR/module.py
import os
import json
import faiss
import torch
import numpy as np
from bson.objectid import ObjectId
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer
from dotenv import load_dotenv

from utils.mongo_conn import connect_to_mongo
from utils.progress_messenger import ProgressMessenger
from utils.llm_communicator import LLMCommunicator

load_dotenv()

class GRSystem:
    def __init__(
        self,
        directory_answers: str,
        prog_messenger: ProgressMessenger,
        llm_client: LLMCommunicator,
        list_doc_ids=None,
        k=10,
        chunk_size=200,
        chunk_overlap=50,
    ):
        self.prog_messenger = prog_messenger
        self.llm = llm_client
        self.k = k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.list_doc_ids = list_doc_ids or []
        self.directory_answers = directory_answers

        # Connect once and store both collections
        self.client, self.db, self.coll, self.coll_meta = self._connect_to_db()

        # Device selection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prog_messenger.send(
            f"{'CUDA available' if self.device=='cuda' else 'CUDA not available'}. "
            f"Using {torch.cuda.get_device_name(0) if self.device=='cuda' else 'CPU'}."
        )

        # Load encoder & tokenizer
        self.retriever = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            device=self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "JungZoona/T3Q-qwen2.5-14b-v1.0-e3"
        )

        # IR prep: load, chunk, embed, index
        self.prog_messenger.send("Processing contents of documents.")
        self.parent_docs = self._get_contents()
        self.child_chunks, self.child_to_parent = self._chunk_documents(self.parent_docs)
        self.embeddings = self._encode_chunks(self.child_chunks)
        self.index = self._build_faiss_index(self.embeddings)

    def _connect_to_db(self):
        return connect_to_mongo(
            os.getenv("MONGO_USER"), 
            os.getenv("MONGO_PASSWORD")
        )

    def _get_contents(self):
        docs = []
        def fetch_doc(d_id):
            try:
                doc = self.coll.find_one(
                    {"_id": ObjectId(d_id)}, 
                    {"Content": 1}
                )
                return doc.get("Content") if doc else None
            except Exception as e:
                self.prog_messenger.send(f"Error fetching {d_id}: {e}")

        with ThreadPoolExecutor(max_workers=10) as pool:
            for content in pool.map(fetch_doc, self.list_doc_ids):
                if content:
                    docs.append(content)

        os.makedirs(self.directory_answers, exist_ok=True)
        with open(
            os.path.join(self.directory_answers, "docs.json"), 
            "w", encoding="utf-8"
        ) as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)

        return docs

    def _chunk_documents(self, documents):
        chunks, mapping = [], []
        step = self.chunk_size - self.chunk_overlap
        for doc_idx, doc in enumerate(documents):
            tokens = doc.split()
            for i in range(0, len(tokens), step):
                chunks.append(" ".join(tokens[i : i + self.chunk_size]))
                mapping.append(doc_idx)
        return chunks, mapping

    def _encode_chunks(self, chunks):
        arr = self.retriever.encode(
            chunks,
            convert_to_tensor=False,
            batch_size=32,
            show_progress_bar=True,
            device=self.device
        )
        return np.array(arr, dtype="float32")

    def _build_faiss_index(self, embeddings_np):
        dim = embeddings_np.shape[1]
        idx = faiss.IndexFlatL2(dim)
        idx.add(embeddings_np)
        self.prog_messenger.send(f"FAISS index built with {embeddings_np.shape[0]} vectors.")
        return idx

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _build_context(self, parent_docs):
        max_ctx, reserve = 4096, 1000
        avail = max_ctx - reserve
        ctx, total_tok = [], 0
        for doc in parent_docs:
            tok = self._count_tokens(doc)
            if total_tok + tok > avail:
                break
            ctx.append(doc)
            total_tok += tok
        return "\n\n".join(ctx)

    def get_summaries(self, user_query: str, previous_responses: str):
        # 1) Encode & search once
        vec = self.retriever.encode([user_query], device=self.device)[0]
        D, I = self.index.search(np.array([vec]), self.k)
        parent_idxs = sorted({self.child_to_parent[i] for i in I[0]})

        # 2) Collect parent docs & their ORIGINAL dados _ids
        parent_docs = [self.parent_docs[i] for i in parent_idxs]
        dados_ids   = [self.list_doc_ids[i]  for i in parent_idxs]

        # 3) Build the context for the LLM
        context = self._build_context(parent_docs)

        # 4) Fetch your metadata-key ID for each dados_id
        metadata_keys = []
        for d_id in dados_ids:
            try:
                meta = self.coll_meta.find_one(
                    {"db_ID": ObjectId(d_id)}, 
                    {"ID": 1}
                )
                if meta and meta.get("ID"):
                    metadata_keys.append(str(meta["ID"]))
                else:
                    metadata_keys.append(f"(no metadata for {d_id})")
            except Exception as e:
                self.prog_messenger.send(f"Error fetching metadata for {d_id}: {e}")
                metadata_keys.append(f"(error {d_id})")

        # 5) Prepare & dump the prompt
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
Se necessário, a resposta poderá apenas conter tags em html para headers, bold, italics e underline.

Evita repetir isto:
{previous_responses}

Contexto:
{context}

Pergunta: {user_query}"""

        with open(
            os.path.join(self.directory_answers, "llm_content.json"), 
            "w", encoding="utf-8"
        ) as f:
            json.dump({"prompt": prompt}, f, ensure_ascii=False, indent=2)

        # 6) Generate
        summary = self.llm.generate(prompt) or "Error generating summary."

        # 7) Return summary + your metadata keys as `document_ids`
        return {
            "summary": summary,
            "document_ids": metadata_keys
        }
