import argparse
import sys

from qdrant_client import QdrantClient, models
from vectorfox.db import search_contexts
from vectorfox.llm import chat_completion, build_rag_prompt


def search(query_text: str, top_k: int = 5):
    print("Searching Documentation...\n")
    contexts = search_contexts(query_text, top_k=top_k)

    print("\nCalling Ollama LLM...\n")
    prompt = build_rag_prompt(contexts, query_text)

    for line in chat_completion(prompt, sse=False):
        sys.stdout.write(line)
        sys.stdout.flush()

    print("\nSources:\n" + "\n".join(f"- {context[1]}" for context in contexts))


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
