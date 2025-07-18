import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from vectorfox.db import search_contexts
from vectorfox.llm import chat_completion, build_rag_prompt


os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI()

HERE = os.path.dirname(__file__)

templates = Jinja2Templates(directory=os.path.join(HERE, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/stream")
def stream(query: str):
    contexts = search_contexts(query)
    prompt = build_rag_prompt(contexts, query)

    return StreamingResponse(
        chat_completion(prompt),
        media_type="text/event-stream",
    )


@app.get("/sources")
def get_sources(query: str):
    contexts = search_contexts(query)
    return [url for (url, _) in contexts]
