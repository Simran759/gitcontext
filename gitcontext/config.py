"""Central configuration.

Every knob you might tweak lives here so the rest of the code reads cleanly.
Each value can be overridden with an environment variable.
"""
from __future__ import annotations
import os
from pathlib import Path

# Where the index (vectors + chunk metadata + bm25 corpus) is written on disk.
INDEX_DIR = Path(os.getenv("GITCONTEXT_INDEX_DIR", ".gitcontext_index"))

# --- Retrieval knobs (see README for what each one does) ---
TOP_K_DENSE = int(os.getenv("GITCONTEXT_TOP_K_DENSE", "25"))    # vector candidates
TOP_K_SPARSE = int(os.getenv("GITCONTEXT_TOP_K_SPARSE", "25"))  # bm25 candidates
RRF_K = int(os.getenv("GITCONTEXT_RRF_K", "60"))                # RRF fusion constant
TOP_K_FINAL = int(os.getenv("GITCONTEXT_TOP_K_FINAL", "8"))     # chunks sent to the LLM

# --- Multi-query retrieval (query translation / RAG-fusion) ---
MULTI_QUERY = os.getenv("GITCONTEXT_MULTI_QUERY", "0") == "1"   # off by default (adds LLM latency)
MULTI_QUERY_N = int(os.getenv("GITCONTEXT_MULTI_QUERY_N", "3")) # extra rephrasings to generate

# --- Commit / PR history ingestion ---
INGEST_COMMITS = os.getenv("GITCONTEXT_INGEST_COMMITS", "1") == "1"
COMMIT_LIMIT = int(os.getenv("GITCONTEXT_COMMIT_LIMIT", "200")) # most-recent commits to index

# File types we ingest.
CODE_EXTS = {".py"}
DOC_EXTS = {".md", ".markdown", ".rst", ".txt"}

# Fixed-size chunking fallback (counted in words for simplicity).
CHUNK_WORDS = 160
CHUNK_OVERLAP = 30
