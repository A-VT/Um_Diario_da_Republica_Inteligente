from utils.json_file_handler import JSONFileHandler
from utils.mongo_conn import connect_to_mongo
from bson.objectid import ObjectId
from bson.errors import InvalidId
from dotenv import load_dotenv
from enum import Enum
import requests
import os


class ModelType(Enum):
    QWEN2_5__14b = "t3q-qwen2.5-14b-v1.0-e3"

class GRSystem:
    def __init__(self, model = ModelType.QWEN2_5__14b, max_tokens=100000, list_doc_ids= []):
        self.model = model
        self.max_tokens = max_tokens
        self.list_doc_ids = list_doc_ids

        load_dotenv()
        self.LLM_url = os.getenv("API_ENDPOINT")
        self.cred_mongo_user = os.getenv("MONGO_USER")
        self.cred_mongo_password = os.getenv("MONGO_PASSWORD")
    
    def _connect_to_db(self):
        return connect_to_mongo(self.cred_mongo_user, self.cred_mongo_password)

    def _get_contents(self):
        self.client, self.db, self.collection_dados, self.collection_metadados = self._connect_to_db()
        docs = []

        for doc_id in self.list_doc_ids:
            try:
                object_id = ObjectId(doc_id)
                result = self.collection_dados.find_one({"_id": object_id})
                if result:
                    content = result.get("Content")
                    if content:
                        docs.append(content)
                    else:
                        print(f"[WARN] Document found but no 'Content' field in ID: {doc_id}")
                else:
                    print(f"[WARN] No document found for ID: {doc_id}")
            except InvalidId:
                print(f"[ERROR] Invalid ObjectId: {doc_id}")
            except Exception as e:
                print(f"[ERROR] Unexpected error for ID {doc_id}: {e}")

        handler = JSONFileHandler("./GR/results/docs")
        handler.delete_results()
        handler.save_results(docs)
        return docs


    def send_context_query(self):
        
        prompt = """És um assistente jurídico responsável por analisar e resumir documentos legais. O utilizador fornecerá uma pergunta jurídica específica ou um conjunto de palavras-chave. Utiliza extritamente os documentos que irão ser forcenidos nas próximas queries de role 'system'.

Cumpre estas regras:  
0. Se não houver informação suficiente - informa isso e não cries informação.
1. Extrai e resume os artigos, princípios, definições, regras-chave, precedentes e cláusulas mais relevantes na documentação fornecida.  
2. Se o utilizador fornecer uma pergunta, responde-a de forma concisa, citando e resumindo a informação legal pertinente.  
3. Se o utilizador fornecer palavras-chave, deverás identifica e resume as seções nos documentos mais relevantes para esses termos.
5. Menciona sempre a fonte da informação apresentada.
6. Sempre que encontrares as palavras exatas 'VER ALTERAÇÕES' seguidas por várias linhas começadas por 'VER ALTERAÇÕES', deverás interpretá-las como uma secção adicional ao documento.
7. A primeira secção de 'VER ALTERAÇÕES' remete para o documento inteiro; as seguintes secções do mesmo estilo referem-se à informação imediatamente anterior a essa secção.
8. As secções 'VER ALTERAÇÕES' deverão ser omitidas sempre que possível.
8. As respostas deverão ser escritas em Português de Portugal.

Estruture a resposta de forma clara, utilizando:  
- **Tópicos** para facilitar a leitura  
- **Seções numeradas** quando apropriado
- **Títulos categorizados**, se necessário  
"""

        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model.value,
            "messages": [{"role": "system", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": self.max_tokens,
            "stream": False
        }

        response = requests.post(self.LLM_url, json=data, headers=headers)


    def send_docs_queries(self, docs):
        for doc in docs:
            prompt = doc

            headers = {"Content-Type": "application/json"}
            data = {
                "model": self.model.value,
                "messages": [{"role": "system", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": self.max_tokens,
                "stream": False
            }

            response = requests.post(self.LLM_url, json=data, headers=headers)


    def _chat_with_model(self, user_query):
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model.value,
            "messages": [{"role": "user", "content": user_query}],
            "temperature": 0.7,
            "max_tokens": self.max_tokens,
            "stream": False
        }

        response = requests.post(self.LLM_url, json=data, headers=headers)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            return f"Error: {response.status_code} - {response.text}"


    def get_summaries(self, user_query):
        print("[GR] Preparing LLM response...")

        docs = self._get_contents()
        base_prompt = """És um assistente jurídico responsável por analisar e resumir documentos legais. O utilizador fornecerá uma pergunta jurídica específica ou um conjunto de palavras-chave. Utiliza extritamente os documentos que irão ser forcenidos nas próximas queries de role 'system'.

Cumpre estas regras:  
0. Se não houver informação suficiente - informa isso e não cries informação.
1. Extrai e resume os artigos, princípios, definições, regras-chave, precedentes e cláusulas mais relevantes na documentação fornecida.  
2. Se o utilizador fornecer uma pergunta, responde-a de forma concisa, citando e resumindo a informação legal pertinente.  
3. Se o utilizador fornecer palavras-chave, deverás identifica e resume as seções nos documentos mais relevantes para esses termos.
5. Menciona sempre a fonte da informação apresentada.
6. Sempre que encontrares as palavras exatas 'VER ALTERAÇÕES' seguidas por várias linhas começadas por 'VER ALTERAÇÕES', deverás interpretá-las como uma secção adicional ao documento.
7. A primeira secção de 'VER ALTERAÇÕES' remete para o documento inteiro; as seguintes secções do mesmo estilo referem-se à informação imediatamente anterior a essa secção.
8. As secções 'VER ALTERAÇÕES' deverão ser omitidas sempre que possível.
9. As respostas deverão ser escritas em Português de Portugal.
10. Escreve de forma clara.
"""

        messages = [{"role": "system", "content": base_prompt}]
        
        for doc in docs:
            messages.append({"role": "system", "content": doc})

        messages.append({"role": "user", "content": user_query})

        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model.value,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": self.max_tokens,
            "stream": False
        }

        response = requests.post(self.LLM_url, json=data, headers=headers)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            return f"Error: {response.status_code} - {response.text}"
