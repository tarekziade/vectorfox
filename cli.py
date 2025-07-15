import argparse
from qdrant_client import QdrantClient, models

COLLECTION_NAME = "firefox_docs"
QDRANT_PATH = "qdrant_data"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

client = QdrantClient(path=QDRANT_PATH)


def search(query_text: str, top_k: int = 5):
    query_doc = models.Document(text=query_text, model=MODEL_NAME)
    results = client.query_points(
        collection_name=COLLECTION_NAME, query=query_doc, limit=top_k, with_payload=True
    )

    print(f'\n🔍 Top {top_k} results for: "{query_text}"\n' + "-" * 60)
    for i, point in enumerate(results.points, 1):
        payload = point.payload or {}
        text_snippet = payload.get("text", "").strip().replace("\n", " ")
        url = payload.get("url", "N/A")
        print(f"{i}. [URL] {url}\n   [Text] {text_snippet[:250]}...\n")


def main():
    parser = argparse.ArgumentParser(
        description="Semantic search in Firefox docs Qdrant DB"
    )
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of results to return"
    )

    args = parser.parse_args()
    search(args.query, top_k=args.top_k)


if __name__ == "__main__":
    main()
