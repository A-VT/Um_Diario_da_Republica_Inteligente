from bson.objectid import ObjectId
from utils.mongo_conn import *
from dotenv import load_dotenv
from enum import Enum
import requests
import os

load_dotenv()
LLM_url = os.getenv("API_ENDPOINT")

class ModelType(Enum):
    MISTRAL_7B_INSTRUCT = "mistral-7b-instruct-v0.3"
    MISTRAL_7B = "mistral-7b-v0.1"

LLM_MODEL = ModelType.MISTRAL_7B
MAX_TOKENS = 10000

LIST_DOC_IDS = ["6765b82130d2176f56d7974c"]

def get_documents(list_doc_ids):
    docs= []
    client, db, collection_dados, collection_metadados = connect_to_mongo()
    for doc_id in list_doc_ids:
        document = collection_dados.find_one({"_id": ObjectId(doc_id)}, {"Content": 1, "_id": 0})
        if document:
            docs.append(document["Content"])
    return docs

def create_base_prompt(docs):
    if LLM_MODEL in [ModelType.MISTRAL_7B, ModelType.MISTRAL_7B_INSTRUCT] :
        role = "system"

    prompt = "Use the information in these documents:"
    for doc in iter(docs):
        prompt += "\n --- \n {doc} "
    return {"role": role, "content": prompt}

def chat_with_model(user_query, base_prompt, model):
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [base_prompt, {"role": "user", "content": user_query}],
        "temperature": 0.7,
        "max_tokens": MAX_TOKENS,
        "stream": False
    }
    
    response = requests.post(LLM_url, json=data, headers=headers)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        return f"Error: {response.status_code} - {response.text}"


documents_to_consider = get_documents(LIST_DOC_IDS)
base_prompt = create_base_prompt(documents_to_consider)
print(chat_with_model("How does photosynthesis work?", base_prompt, model= LLM_MODEL.value))


