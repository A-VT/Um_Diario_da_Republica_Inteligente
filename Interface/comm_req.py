from enum import Enum
from pydantic import BaseModel
from utils.retriever.model_type import ModelType
from typing import Optional

class QueryResponse(BaseModel):
    summary: str
    document_ids: list[str]

class QueryRequest(BaseModel):
    text: str
    models: list[ModelType]
    n_docs: int
    auto_select_keywords: bool
    previous_responses: Optional[str] = None
