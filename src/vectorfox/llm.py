import os
import requests
import json
from fastapi.responses import StreamingResponse

from typing import List, Tuple

with open(os.path.join(os.path.dirname(__file__), "config.json")) as f:
    config = json.load(f)

PROVIDER = config.get("provider", "ollama")

# Ollama settings
OLLAMA_MODEL = config["ollama"]["model"]
OLLAMA_CHAT_URL = config["ollama"]["url"]


VERTEX_REGION = config["vertex"]["region"]
VERTEX_PROJECT_ID = config["vertex"]["project_id"]
VERTEX_MODEL_ID = config["vertex"]["model_id"]


# OpenAI settings
if PROVIDER == "openai":
    import openai

    OPENAI_MODEL = config["openai"]["model"]
    openai.api_key = config["openai"]["api_key"]


def build_rag_prompt(context_blocks: List[Tuple[str, str]], query: str) -> str:
    labeled_context = "\n\n".join(
        f"[{source}]\n{text}" for source, text in context_blocks
    )

    return f"""You are a helpful and concise assistant answering technical questions about the Firefox codebase and its documentation.

Use only the information provided in the context blocks below to answer the question. Prefer information from earlier blocks when multiple sources overlap or conflict.

Be precise and technical. Format your answer in markdown, using bullet points, code blocks, and links where relevant. Do not show your reasoning or internal thoughts. Do not make up information or guess.

---

Context Blocks:
{labeled_context}

---

Question: {query}

Answer:"""


def stream_ollama(
    model: str, system_prompt: str, user_prompt: str, sse: bool
) -> Generator[str, None, None]:
    def wrap(msg):
        return f"data: {msg}\n\n" if sse else msg

    response = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": True,
        },
        stream=True,
    )

    if response.status_code != 200:
        yield wrap(f"Ollama error: {response.status_code} {response.text}")
        return

    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            content = data.get("message", {}).get("content", "")
            yield wrap(content)

    yield wrap("[DONE]")


def stream_openai(model: str, system_prompt: str, user_prompt: str, sse: bool):
    import openai

    def wrap(msg):
        return f"data: {msg}\n\n" if sse else msg

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
        )

        for chunk in response:
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                yield wrap(delta["content"])

        yield wrap("[DONE]")

    except Exception as e:
        yield wrap(f"OpenAI error: {str(e)}")


def stream_vertexai(prompt: str, sse: bool) -> Generator[str, None, None]:
    def wrap(msg):
        return f"data: {msg}\n\n" if sse else msg

    # Get access token via gcloud subprocess
    process = subprocess.Popen(
        "gcloud auth print-access-token", stdout=subprocess.PIPE, shell=True
    )
    access_token_bytes, _ = process.communicate()
    access_token = access_token_bytes.decode("utf-8").strip()

    url = (
        f"https://{VERTEX_REGION}-aiplatform.googleapis.com/v1/projects/"
        f"{VERTEX_PROJECT_ID}/locations/{VERTEX_REGION}/publishers/"
        f"mistralai/models/{VERTEX_MODEL_ID}:rawPredict"
    )

    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}

    payload = {
        "model": VERTEX_MODEL_ID,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        yield wrap(f"Vertex AI error: {response.status_code} {response.text}")
        return

    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        yield wrap(content)
    except Exception as e:
        yield wrap(f"Error parsing Vertex response: {str(e)}\nRaw: {response.text}")

    yield wrap("[DONE]")


def chat_completion(prompt: str, sse: bool = True):
    system_prompt = (
        "You are a concise assistant answering technical questions about the Firefox codebase. "
        "Do not show internal reasoning or say 'thinking'. Just answer clearly and directly."
    )

    if PROVIDER == "openai":
        return stream_openai(OPENAI_MODEL, system_prompt, prompt, sse)
    elif PROVIDER == "vertex":
        return stream_vertexai(prompt, sse)
    else:
        return stream_ollama(OLLAMA_MODEL, system_prompt, prompt, sse)
