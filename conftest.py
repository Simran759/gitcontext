"""Pytest session setup.

Force the fast, offline, deterministic backends so the suite runs with no model
downloads and no network: hashing embedder, no-op reranker, no LLM, no git
history ingestion by default. These env vars are read by get_embedder /
get_reranker / generate at call time, so setting them here (before any gitcontext
module is imported) is enough.
"""
import os

os.environ.setdefault("GITCONTEXT_EMBED", "hashing")
os.environ.setdefault("GITCONTEXT_RERANK", "noop")
os.environ.setdefault("GITCONTEXT_LLM", "none")
os.environ.setdefault("GITCONTEXT_INGEST_COMMITS", "0")
os.environ.setdefault("GITCONTEXT_MULTI_QUERY", "0")
