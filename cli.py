import argparse
from qdrant_client import QdrantClient, models
from openai import OpenAI
import os


def strip_think_section(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>")[1].strip()
    return text


LLM_MODEL = "qwen3:0.6b"
COLLECTION_NAME = "firefox_docs"
QDRANT_PATH = "qdrant_data"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_API_BASE = "http://localhost:11434/v1"
OPENAI_API_KEY = "ollama"

client = QdrantClient(path=QDRANT_PATH)
openai_client = OpenAI(
    base_url=OPENAI_API_BASE,
    api_key=OPENAI_API_KEY,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def build_rag_prompt(context_blocks, query):
    context_text = "\n\n".join(context_blocks)
    return f"""You are a helpful assistant answering technical questions about the Firefox codebase and its documentation.

Answer the following question based on the context provided. Be accurate and concise. Do not show your thinking or reasoning process.

---

Context:
{context_text}

---

Question: {query}
Answer:"""


def search(query_text: str, top_k: int = 5):
    print("Searching Documentation...\n")
    query_doc = models.Document(text=query_text, model=MODEL_NAME)
    results = client.query_points(
        collection_name=COLLECTION_NAME, query=query_doc, limit=top_k, with_payload=True
    )

    # Extract top-k context blocks and source URLs
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

    # Build and run RAG prompt
    prompt = build_rag_prompt(contexts, query_text)
    print("\nThinking...\n")

    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a concise assistant answering technical questions about the Firefox codebase. "
                    "Do not show internal reasoning or say 'thinking'. Just answer clearly and directly."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    print("Response:\n")
    print(strip_think_section(response.choices[0].message.content.strip()))
    print("\nSources:\n" + "\n".join(f"- {url}" for url in sources))


def main():
    parser = argparse.ArgumentParser(
        description="RAG over Firefox docs using Qdrant + Ollama"
    )
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of results to return"
    )

    args = parser.parse_args()
    search(args.query, top_k=args.top_k)


if __name__ == "__main__":
    main()
