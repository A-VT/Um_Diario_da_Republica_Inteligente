def mongo_regex_search(collection_metadados, search_terms, n_docs):
    """
    Perform a pure MongoDB regex search in the collection based on the search terms (substring match).

    Args:
        collection_metadados: MongoDB collection object to search in.
        search_terms: List of search terms (strings).
        n_docs: Maximum number of documents to retrieve.

    Returns:
        List of matching documents (dicts).
    """
    or_clauses = []
    for term in search_terms:
        or_clauses.append({"Sumario": {"$regex": term, "$options": "i"}})  # Case insensitive
        or_clauses.append({"Titulo": {"$regex": term, "$options": "i"}})

    query = {"$or": or_clauses}

    try:
        cursor = collection_metadados.find(query, {"_id": 1, "db_ID": 1, "Titulo": 1, "Sumario": 1}).limit(n_docs)
        results = []
        for doc in cursor:
            results.append({
                "id": str(doc["_id"]),
                "db_ID": str(doc.get("db_ID", "")),
                "Titulo": doc.get("Titulo", ""),
                "Sumario": doc.get("Sumario", "")
            })
        return results
    except Exception as e:
        print(f"[MongoQuery] Error during Mongo search: {e}")
        return []
    
def mongo_text_search(collection_metadados, search_terms, n_docs):
    """
    Perform a full-text search using MongoDB text indexes.
    Automatically creates a text index on 'Titulo' and 'Sumario' if it doesn't exist.

    Args:
        collection_metadados: MongoDB collection object.
        search_terms: List of search terms (strings).
        n_docs: Maximum number of documents to retrieve.

    Returns:
        List of matching documents with relevance scores.
    """
    # Ensure text index exists
    index_exists = False
    for index in collection_metadados.list_indexes():
        if index.get("weights") == {"Titulo": 1, "Sumario": 1} and index.get("key", {}).get("_fts") == "text":
            index_exists = True
            break

    if not index_exists:
        try:
            collection_metadados.create_index(
                [("Titulo", "text"), ("Sumario", "text")],
                name="text_search_index",
                default_language="portuguese",  # Optional: improve stemming/stopword handling
                weights={"Titulo": 10, "Sumario": 2}  # Optional: boost title matches
            )
        except Exception as e:
            print(f"[MongoTextSearch] Failed to create text index: {e}")
            return []

    # Convert list of search terms to a single space-separated string
    if isinstance(search_terms, list):
        search_string = " ".join(search_terms)
    else:
        search_string = str(search_terms)

    # Perform the text search
    try:
        cursor = collection_metadados.find(
            {"$text": {"$search": search_string}},
            {
                "score": {"$meta": "textScore"},
                "_id": 1,
                "db_ID": 1,
                "Titulo": 1,
                "Sumario": 1
            }
        ).sort([("score", {"$meta": "textScore"})]).limit(n_docs)

        results = []
        for doc in cursor:
            results.append({
                "id": str(doc["_id"]),
                "db_ID": str(doc.get("db_ID", "")),
                "Titulo": doc.get("Titulo", ""),
                "Sumario": doc.get("Sumario", ""),
                "score": doc.get("score", 0)
            })
        return results
    except Exception as e:
        print(f"[MongoTextSearch] Error during text search: {e}")
        return []