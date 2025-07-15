from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import TextVectorizerConfig


COLLECTION_NAME = "firefox_docs"

client = QdrantClient(host="localhost", port=6333)

vectorizer_config = {
    "text": TextVectorizerConfig(
        type="text", source="text", model="qdrant-text", use_cache=True
    )
}
if COLLECTION_NAME in [c.name for c in client.get_collections().collections]:
    print(f"[!] Collection '{COLLECTION_NAME}' already exists.")
else:
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"text": VectorParams(size=384, distance=Distance.COSINE)},
        vectorizer_config=vectorizer_config,
    )
    print(f"[+] Collection '{COLLECTION_NAME}' created with built-in text vectorizer.")
