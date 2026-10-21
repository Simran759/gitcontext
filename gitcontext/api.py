"""FastAPI backend + browser UI.

Run:  uvicorn gitcontext.api:app --reload
  - http://127.0.0.1:8000/       chat UI (webui.py)
  - http://127.0.0.1:8000/docs   interactive Swagger UI
Endpoints: POST /index, POST /reindex, POST /query.
"""
from __future__ import annotations
from typing import Optional
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .pipeline import GitContext, build_index, reindex
from .webui import INDEX_HTML

app = FastAPI(title="GitContext", description="RAG developer copilot over a repo")
_state: dict = {"gc": None}


class IndexReq(BaseModel):
    repo_path: str


class QueryReq(BaseModel):
    question: str
    top_k: int = 8
    multi_query: bool = False
    filter: Optional[dict] = None          # e.g. {"source": "code"} or {"path_prefix": "auth/"}


@app.get("/", response_class=HTMLResponse)
def home():
    return INDEX_HTML


@app.post("/index")
def index(req: IndexReq):
    n = build_index(req.repo_path)
    _state["gc"] = None                 # force reload of the fresh index
    return {"indexed_chunks": n}


@app.post("/reindex")
def reindex_endpoint(req: IndexReq):
    """Incremental refresh - a GitHub push webhook would call this."""
    n = reindex(req.repo_path)
    _state["gc"] = None
    return {"new_chunks": n}


@app.post("/query")
def query(req: QueryReq):
    if _state["gc"] is None:
        _state["gc"] = GitContext()
    return _state["gc"].query(req.question, top_k=req.top_k,
                              multi_query=req.multi_query, flt=req.filter)
