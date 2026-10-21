"""Step 8 - Augmentation + generation (LOCAL-first, no API key needed).

Exposes two things:

- complete(system, user)  -> (text, backend) : a generic local LLM call, reused
  by both answer generation AND multi-query expansion.
- generate(query, chunks) -> (answer, prompt): builds the grounded prompt and
  answers from it, with citations.

Backends tried in order (all local, no key):
1. Ollama       - if a server is running at localhost:11434.
2. transformers - a small local instruct model on CPU (default Qwen2.5-0.5B-Instruct).
3. Extractive   - for generate(): return the top chunk so nothing hard-fails.

Anthropic is opt-in only (GITCONTEXT_LLM=anthropic + ANTHROPIC_API_KEY).
"""
from __future__ import annotations
import json
import os
import urllib.request
from typing import List, Optional, Tuple

SYSTEM = (
    "You are a codebase assistant. Answer ONLY from the context below. "
    "Cite sources as [file_path]. If the answer isn't in the context, say so. "
    "Be concise."
)

_LOCAL_MODEL = os.getenv("GITCONTEXT_LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
_OLLAMA_URL = os.getenv("GITCONTEXT_OLLAMA_URL", "http://localhost:11434")
_OLLAMA_MODEL = os.getenv("GITCONTEXT_OLLAMA_MODEL", "llama3.2")

_cache: dict = {"tok": None, "model": None}


# --------------------------------------------------------------------------
# Prompt building
# --------------------------------------------------------------------------
def build_context(chunks: List[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        m = c["metadata"]
        cite = m.get("file_path", "?")
        if m.get("symbol"):
            cite += f":{m['symbol']}"
        blocks.append(f"[{i}] {cite}\n{c['text']}")
    return "\n\n".join(blocks)


def build_prompt(query: str, chunks: List[dict]) -> str:
    """The full flat prompt (used for display)."""
    return f"{SYSTEM}\n\nContext:\n{build_context(chunks)}\n\nQuestion: {query}"


# --------------------------------------------------------------------------
# Backend 1: Ollama
# --------------------------------------------------------------------------
def _ollama_available() -> bool:
    try:
        urllib.request.urlopen(_OLLAMA_URL + "/api/tags", timeout=2).read()
        return True
    except Exception:
        return False


def _ollama_complete(prompt: str, max_new_tokens: int) -> str:
    payload = json.dumps({
        "model": _OLLAMA_MODEL, "prompt": prompt, "stream": False,
        "options": {"num_predict": max_new_tokens},
    }).encode()
    req = urllib.request.Request(
        _OLLAMA_URL + "/api/generate", data=payload,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())["response"].strip()


# --------------------------------------------------------------------------
# Backend 2: transformers local instruct model (CPU)
# --------------------------------------------------------------------------
def _load_local():
    if _cache["model"] is None:
        from .quiet import quiet_ml
        quiet_ml()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _cache["tok"] = AutoTokenizer.from_pretrained(_LOCAL_MODEL)
        _cache["model"] = AutoModelForCausalLM.from_pretrained(_LOCAL_MODEL)
    return _cache["tok"], _cache["model"]


def _local_complete(system: str, user: str, max_new_tokens: int) -> str:
    tok, model = _load_local()
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    inputs = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    gen = out[0][inputs["input_ids"].shape[-1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()


# --------------------------------------------------------------------------
# Generic local completion (shared by answering + query expansion)
# --------------------------------------------------------------------------
def complete(system: str, user: str, max_new_tokens: int = 256) -> Tuple[Optional[str], Optional[str]]:
    """Return (text, backend_label). (None, None) if no local model is available."""
    backend = os.getenv("GITCONTEXT_LLM", "auto").lower()

    if backend == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model="claude-sonnet-5-5", max_tokens=max_new_tokens,
                system=system, messages=[{"role": "user", "content": user}])
            return msg.content[0].text.strip(), "anthropic"
        except Exception:
            return None, None

    if backend in ("auto", "ollama") and _ollama_available():
        try:
            return _ollama_complete(f"{system}\n\n{user}", max_new_tokens), f"ollama:{_OLLAMA_MODEL}"
        except Exception:
            pass
    if backend in ("auto", "local"):
        try:
            return _local_complete(system, user, max_new_tokens), f"local:{_LOCAL_MODEL}"
        except Exception:
            pass
    return None, None


# --------------------------------------------------------------------------
# Answer generation
# --------------------------------------------------------------------------
def _extractive(chunks: List[dict]) -> str:
    if not chunks:
        return "No relevant context was retrieved."
    m = chunks[0]["metadata"]
    cite = m.get("file_path", "?") + (f":{m['symbol']}" if m.get("symbol") else "")
    return f"(no local model loaded) Most relevant source [{cite}]:\n\n{chunks[0]['text']}"


def generate(query: str, chunks: List[dict]) -> Tuple[str, str]:
    prompt = build_prompt(query, chunks)
    user = f"Context:\n{build_context(chunks)}\n\nQuestion: {query}"
    text, backend = complete(SYSTEM, user, max_new_tokens=256)
    if text is None:
        return "[extractive] " + _extractive(chunks), prompt
    return f"[{backend}] " + text, prompt
