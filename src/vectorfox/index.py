import requests
import re
from urllib.parse import urljoin, urlparse
from qdrant_client import QdrantClient, models
from tqdm import tqdm
import hashlib
import os
from bs4 import BeautifulSoup
from urllib.parse import urldefrag

BASE_URL = "https://firefox-source-docs.mozilla.org/"
COLLECTION_NAME = "firefox_docs"
QDRANT_PATH = "qdrant_data"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_PAGES = 2000
VISITED = set()

client = QdrantClient(path=QDRANT_PATH)


def normalize_url(url: str) -> str:
    """Removes URL fragments (e.g., #section-name)"""
    return urldefrag(url)[0]


def hash_id(url: str) -> int:
    return int(hashlib.sha256(url.encode()).hexdigest(), 16) % (10**18)


def get_source_link_from_html(html: str, base_url: str) -> str | None:
    """
    Parses the HTML and returns the absolute URL to the .rst.txt source
    if the "View page source" link is present.
    """
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        if a.text.strip() == "View page source":
            return urljoin(base_url, a["href"])
    return None


def has_view_page_source(html: str) -> bool:
    return "View page source" in html


def extract_rst_links(rst_text: str) -> list[str]:
    link_pattern = re.compile(r"`[^`<]*<([^>]+)>`_")
    return [
        urljoin(BASE_URL, match)
        for match in link_pattern.findall(rst_text)
        if urlparse(match).scheme in ("", "https")
    ]


def extract_html_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    return [
        urljoin(base_url, a["href"])
        for a in soup.find_all("a", href=True)
        if urlparse(a["href"]).netloc in ("", urlparse(BASE_URL).netloc)
    ]


def fetch_page(url: str) -> tuple[str | None, str | None, list[str]]:
    try:
        html_response = requests.get(url, timeout=10)
        if not html_response.ok or "text/html" not in html_response.headers.get(
            "Content-Type", ""
        ):
            return None, None, []

        html = html_response.text
        source_link = get_source_link_from_html(html, base_url=url)
        rst_text = None
        links = []
        html_text = None
        if source_link:
            rst_response = requests.get(source_link, timeout=10)
            if rst_response.ok and rst_response.text.strip():
                rst_text = rst_response.text.strip()

        # use HTML to extract links
        soup = BeautifulSoup(html, "html.parser")
        main = soup.select_one("main") or soup.body
        if main:
            clean_text = main.get_text(separator="\n")
            links = [
                normalize_url(link) for link in extract_html_links(html, base_url=url)
            ]
            html_text = clean_text.strip()

        return rst_text, html_text, links
    except Exception as e:
        print(f"[!] Failed to fetch {url}: {e}")
        return None, None, []


def chunk_text(text: str, max_tokens: int = 500):
    lines = text.split("\n")
    chunks, current, count = [], [], 0
    for line in lines:
        tokens = len(line.split())
        if count + tokens > max_tokens:
            chunks.append("\n".join(current))
            current, count = [line], tokens
        else:
            current.append(line)
            count += tokens
    if current:
        chunks.append("\n".join(current))
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


def delete_previous(url: str):
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.Filter(
            must=[models.FieldCondition(key="url", match=models.MatchValue(value=url))]
        ),
    )


def crawl_and_index(start_url: str):
    queue = [start_url]
    ensure_collection()

    with tqdm(total=MAX_PAGES) as pbar:
        while queue and len(VISITED) < MAX_PAGES:
            url = normalize_url(queue.pop(0))
            if url in VISITED or not url.startswith(BASE_URL):
                continue
            VISITED.add(url)

            rst_text, html_text, new_links = fetch_page(url)
            if rst_text is None and html_text is None:
                continue

            delete_previous(url)

            source_text = rst_text or html_text

            documents = [models.Document(text=source_text, model=MODEL_NAME)]
            payloads = [{"url": url, "text": documents[0]}]
            ids = [hash_id(url)]

            client.upload_collection(
                collection_name=COLLECTION_NAME,
                vectors=documents,
                ids=ids,
                payload=payloads,
            )
            queue.extend(
                [normalize_url(l) for l in new_links if normalize_url(l) not in VISITED]
            )

            pbar.update(1)


if __name__ == "__main__":
    os.makedirs(QDRANT_PATH, exist_ok=True)
    crawl_and_index(BASE_URL)
