import requests
import json
from fastapi.responses import StreamingResponse

from typing import List, Tuple

with open("config.json") as f:
    config = json.load(f)

PROVIDER = config.get("provider", "ollama")

# Ollama settings
OLLAMA_MODEL = config["ollama"]["model"]
OLLAMA_CHAT_URL = config["ollama"]["url"]

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


def chat_completion(prompt: str, sse: bool = True):
    system_prompt = (
        "You are a concise assistant answering technical questions about the Firefox codebase. "
        "Do not show internal reasoning or say 'thinking'. Just answer clearly and directly."
    )

    if PROVIDER == "openai":
        return stream_openai(OPENAI_MODEL, system_prompt, prompt, sse)
    else:
        return stream_ollama(OLLAMA_MODEL, system_prompt, prompt, sse)
