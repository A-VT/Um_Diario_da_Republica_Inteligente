from dotenv import load_dotenv
from enum import Enum
import requests
import os

class LLMModelType(Enum):
    QWEN2_5__14b = "t3q-qwen2.5-14b-v1.0-e3"


class LLMCommunicator:
    def __init__(self, model: LLMModelType, max_tokens: int, prog_messenger):
        load_dotenv()
        self.LLM_url = os.getenv("API_ENDPOINT")
    
        self.model = model.value
        self.max_tokens = max_tokens
        self.prog_messenger = prog_messenger

    def generate(self, prompt: str, temperature: float = 0.7, top_p: float = 0.95):
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": prompt}],
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "top_p": top_p,
            "stream": False
        }
        try:
            res = requests.post(self.LLM_url, json=payload, headers=headers)
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            self.prog_messenger.send(f"LLM request failed: {e}")
            return None

