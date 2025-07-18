# src/vectorfox/cli.py
import uvicorn


def serve():
    uvicorn.run("vectorfox.app:app", host="127.0.0.1", port=8000, reload=True)
