from vector_store import qdrant, COLLECTION, get_embedding

def retrieve(query, k=3):
    query_vector = get_embedding(query)
    results = qdrant.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=k
    )
    return [r.payload["text"] for r in results]
