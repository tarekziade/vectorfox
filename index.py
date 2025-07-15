import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from tqdm import tqdm
import hashlib

BASE_URL = "https://firefox-source-docs.mozilla.org/"
COLLECTION_NAME = "firefox_docs"
VISITED = set()
MAX_PAGES = 200

qdrant = QdrantClient(host="localhost", port=6333)


def hash_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def crawl_and_index(start_url: str):
    queue = [start_url]

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
                points = [
                    PointStruct(
                        id=hash_id(url + f"#{i}"),
                        vector=None,  # Will be created automatically
                        payload={"url": url, "text": chunk},
                    )
                    for i, chunk in enumerate(chunks)
                ]
                qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            pbar.update(1)


if __name__ == "__main__":
    crawl_and_index(BASE_URL)
