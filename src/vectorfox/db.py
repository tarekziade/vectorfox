from qdrant_client import QdrantClient, models


QDRANT_PATH = "qdrant_data"
COLLECTION_NAME = "firefox_docs"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


client = QdrantClient(path=QDRANT_PATH)


def search_contexts(query_text: str, top_k: int = 5):
    query_doc = models.Document(text=query_text, model=MODEL_NAME)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_doc,
        limit=top_k,
        with_payload=True,
    )
    contexts = []
    sources = []
    for point in results.points:
        payload = point.payload or {}
        text = (
            payload["text"]
            if isinstance(payload["text"], str)
            else payload["text"].get("text", "")
        )
        url = payload.get("url", "N/A")
        if text:
            contexts.append(text.strip())
            sources.append(url)
    return zip(sources, contexts)
