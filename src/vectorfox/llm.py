import os
import requests
import json
import subprocess
from fastapi.responses import StreamingResponse
from typing import List, Tuple, Generator


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


def stream_ollama(model: str, system_prompt: str, user_prompt: str, sse: bool):
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


def get_access_token() -> str:
    process = subprocess.Popen(
        "gcloud auth print-access-token --scopes=https://www.googleapis.com/auth/cloud-platform",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
    )
    access_token_bytes, error_bytes = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"Failed to get access token: {error_bytes.decode()}")
    return access_token_bytes.decode("utf-8").strip()


def stream_vertexai(prompt: str, sse: bool) -> Generator[str, None, None]:
    def wrap(msg):
        return f"data: {msg}\n\n" if sse else msg

    try:
        access_token = get_access_token()
    except RuntimeError as e:
        yield wrap(f"Token error: {str(e)}")
        return

    # OpenAI-compatible streaming endpoint for Vertex AI
    url = (
        f"https://{VERTEX_REGION}-aiplatform.googleapis.com/v1/projects/"
        f"{VERTEX_PROJECT_ID}/locations/{VERTEX_REGION}/publishers/"
        f"{config['vertex'].get('publisher', 'mistralai')}/models/"
        f"{VERTEX_MODEL_ID}:streamChatCompletions"
    )

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "X-Goog-User-Project": VERTEX_PROJECT_ID,  # required for quota/billing
    }

    payload = {
        "model": VERTEX_MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "stream": True,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, stream=True)

        if response.status_code != 200:
            yield wrap(f"Vertex AI error: {response.status_code} {response.text}")
            return

        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                data = line.removeprefix("data: ").strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)
                    content = delta["choices"][0].get("delta", {}).get("content", "")
                    if content:
                        yield wrap(content)
                except Exception as e:
                    yield wrap(f"[stream error] {e}")

    except Exception as e:
        yield wrap(f"Request failed: {str(e)}")

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
