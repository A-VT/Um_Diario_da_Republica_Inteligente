from enum import Enum
from .retriever_bm25 import BM25Retriever
from .retriever_word2vec import Word2VecRetriever
from .retriever_tfidf import TfidfRetriever
from .retriever_wiki_word2vec import WikiWord2VecRetriever

class ModelType(str, Enum):
    """
    USAGE:
    retriever_type = ModelType.BM25  # Choose a retriever type
    retriever = retriever_type.get_retriever()  # Get an instance of BM25Retriever
    print(type(retriever))  # Output: <class 'retriever_bm25.BM25Retriever'>
    """

    TF_IDF = "TF-IDF"
    WORD2VEC = "WORD2VEC"
    BM25 = "BM25"
    WIKI_WORD2VEC = "WIKI_WORD2VEC"

    def get_retriever(self):
        retriever_classes = {
            ModelType.TF_IDF: TfidfRetriever(),
            ModelType.WORD2VEC: Word2VecRetriever(),
            ModelType.BM25: BM25Retriever(),
            ModelType.WIKI_WORD2VEC: WikiWord2VecRetriever(),
        }
        return retriever_classes[self]()