import os
import json
import faiss
import numpy as np
from bson.objectid import ObjectId
from dotenv import load_dotenv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch

from utils.mongo_conn import connect_to_mongo

load_dotenv()

# ==== Configuration ====
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
SAVE_DIR = "./faiss_index/"
INDEX_FILE = os.path.join(SAVE_DIR, "index.faiss")
META_FILE = os.path.join(SAVE_DIR, "chunk_metadata.json")


# ==== Setup ====
os.makedirs(SAVE_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
retriever = SentenceTransformer(MODEL_NAME, device=device)

client, db, coll, coll_meta = connect_to_mongo(
    os.getenv("MONGO_USER"),
    os.getenv("MONGO_PASSWORD")
)

print("Fetching all documents...")
#all_docs_cursor = coll.find({}, {"_id": 1, "TipoLegislacao":1, "Content": 1, "FragmentoDiploma":1, "AlteracoesGlobais":1})
#all_docs = [(str(doc["_id"]), doc["Content"]) for doc in all_docs_cursor if doc.get("Content")]

all_docs_cursor = coll.find({}, {
    "_id": 1,
    "TipoLegislacao": 1,
    "Content": 1,
    "FragmentoDiploma": 1,
    "AlteracoesGlobais": 1
})

all_docs = []
for doc in tqdm(all_docs_cursor, desc="Retrieving documents"):
    content = doc.get("Content")
    if not content:
        continue

    doc_id = str(doc["_id"])

    # Get corresponding metadata
    meta = coll_meta.find_one({"db_ID": doc["_id"]}, {
        "Modificacao": 1,
        "Url": 1,
        "ID": 1,
        "Titulo": 1,
        "Sumario": 1
    }) or {}

    all_docs.append({
        "doc_id": doc_id,
        "content": content,
        "tipo_legislacao": doc.get("TipoLegislacao"),
        "fragmento_diploma": doc.get("FragmentoDiploma"),
        "alteracoes_globais": doc.get("AlteracoesGlobais"),
        "modificacao": meta.get("Modificacao"),
        "url": meta.get("Url"),
        "id": meta.get("ID"),
        "titulo": meta.get("Titulo"),
        "sumario": meta.get("Sumario")
    })


# ==== Helpers ====
def chunk_document(text, parent_doc_id, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    tokens = text.split()
    step = chunk_size - overlap
    chunks = []
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = " ".join(chunk_tokens)
        chunks.append((chunk_text, parent_doc_id))
    return chunks

# ==== Step 1: Chunk All Docs ====
print("Chunking documents...")
all_chunks = []
for doc in tqdm(all_docs, desc="Chunking documents"):
    content = doc.get("content")
    if not content:
        continue
    doc_id = doc["doc_id"]
    chunks = chunk_document(content, doc_id)
    all_chunks.extend(chunks)

# ==== Step 2: Encode ====
texts = [c[0] for c in all_chunks]
print(f"Encoding {len(texts)} chunks...")
embeddings = retriever.encode(
    texts,
    convert_to_tensor=False,
    batch_size=32,
    show_progress_bar=True,
    device=device
)
embeddings = np.array(embeddings).astype("float32")

# ==== Step 3: Build FAISS Index with ID Mapping ====
print("Building FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))

# Generate unique numeric IDs for chunks
chunk_ids = np.arange(len(texts)).astype("int64")
index.add_with_ids(embeddings, chunk_ids)

# ==== Step 4: Build Metadata ====
print("Building metadata...")
chunk_metadata = {
    int(i): {
        "parent_doc_id": all_chunks[i][1],
        "chunk_text": all_chunks[i][0]
    }
    for i in range(len(texts))
}

# ==== Step 5: Save ====
print(f"Saving FAISS index to {INDEX_FILE}")
faiss.write_index(index, INDEX_FILE)

print(f"Saving metadata to {META_FILE}")
with open(META_FILE, "w", encoding="utf-8") as f:
    json.dump(chunk_metadata, f, ensure_ascii=False, indent=2)

print("✅ Done.")
