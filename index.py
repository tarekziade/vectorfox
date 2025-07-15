import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from qdrant_client import QdrantClient, models
from tqdm import tqdm
import hashlib
import os

BASE_URL = "https://firefox-source-docs.mozilla.org/"
COLLECTION_NAME = "firefox_docs"
QDRANT_PATH = "qdrant_data"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_PAGES = 200
VISITED = set()

client = QdrantClient(path=QDRANT_PATH)


def hash_id(url: str, i: int) -> int:
    return int(hashlib.sha256(f"{url}#{i}".encode()).hexdigest(), 16) % (10**18)


def extract_links_and_text(url: str):
    try:
        r = requests.get(url, timeout=10)
        if not r.ok or "text/html" not in r.headers.get("Content-Type", ""):
            return [], None

        soup = BeautifulSoup(r.text, "html.parser")
        main = soup.select_one("main") or soup.body
        text = main.get_text(separator="\n") if main else ""

        links = [
            urljoin(url, a["href"])
            for a in soup.find_all("a", href=True)
            if urlparse(a["href"]).netloc in ("", urlparse(BASE_URL).netloc)
        ]
        return links, text.strip()
    except Exception as e:
        print(f"[!] Error fetching {url}: {e}")
        return [], None


def chunk_text(text: str, max_tokens: int = 500):
    lines = text.split("\n")
    chunks, current, count = [], [], 0
    for line in lines:
        tokens = len(line.split())
        if count + tokens > max_tokens:
            chunks.append(" ".join(current))
            current, count = [line], tokens
        else:
            current.append(line)
            count += tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


def ensure_collection():
    if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
        dim = client.get_embedding_size(MODEL_NAME)
        client.create_collection(
            COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=dim, distance=models.Distance.COSINE
            ),
        )


def crawl_and_index(start_url: str):
    queue = [start_url]
    ensure_collection()

    with tqdm(total=MAX_PAGES) as pbar:
        while queue and len(VISITED) < MAX_PAGES:
            url = queue.pop(0)
            if url in VISITED:
                continue
            VISITED.add(url)

            links, text = extract_links_and_text(url)
            queue.extend([l for l in links if l not in VISITED])

            if text:
                chunks = chunk_text(text)
                documents = [
                    models.Document(text=chunk, model=MODEL_NAME) for chunk in chunks
                ]
                payloads = [{"url": url, "text": chunk} for chunk in chunks]
                ids = [hash_id(url, i) for i in range(len(chunks))]

                client.upload_collection(
                    collection_name=COLLECTION_NAME,
                    vectors=documents,
                    ids=ids,
                    payload=payloads,
                )
            pbar.update(1)


if __name__ == "__main__":
    os.makedirs(QDRANT_PATH, exist_ok=True)
    crawl_and_index(BASE_URL)
