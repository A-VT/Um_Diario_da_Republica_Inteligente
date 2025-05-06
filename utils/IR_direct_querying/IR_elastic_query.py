from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import os


def elastic_query_search(search_terms, n_docs):
    """
    Perform a fuzzy multi-field search in Elasticsearch across 'Titulo' and 'Sumario',
    handling typos and boosting 'Titulo' relevance.

    Args:
        search_terms (list[str]): List of search terms (words/phrases).
        n_docs (int): Max number of documents to return.

    Returns:
        List[dict]: Matching documents (with id, db_ID, Titulo, Sumario, score).
    """
    load_dotenv()
    api_key = os.getenv("ELASTIC_API_KEY")
    cloud_id = os.getenv("ELASTIC_CLOUD_ID")
    index_name = os.getenv("ELASTICSEARCH_INDEXNAME")

    es = Elasticsearch(cloud_id=cloud_id, api_key=api_key)

    # Join all search terms into a single string for multi_match
    search_query = " ".join(search_terms)

    query = {
        "multi_match": {
            "query": search_query,
            "fields": ["Titulo", "Sumario^2"],
            "fuzziness": "AUTO",
            "type": "best_fields"
        }
    }

    try:
        response = es.search(index=index_name, query=query, size=n_docs)
        results = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            results.append({
                "id": hit["_id"],
                "db_ID": str(source.get("db_ID", "")),
                "Titulo": source.get("Titulo", ""),
                "Sumario": source.get("Sumario", ""),
                "score": hit["_score"]
            })
        return results
    except Exception as e:
        print(f"[ElasticQuery] Error during Elasticsearch search: {e}")
        return []