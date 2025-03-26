from enum import Enum
from pydantic import BaseModel
from utils.retriever.model_type import ModelType

class QueryResponse(BaseModel):
    answer: str

class QueryRequest(BaseModel):
    text: str
    models: list[ModelType]
    n_docs: int
    auto_select_keywords: bool