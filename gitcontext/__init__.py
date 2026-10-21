"""GitContext - a minimal, end-to-end RAG developer copilot over a code repo.

Read the modules in pipeline order to learn RAG:
    ingest  -> chunk -> embed -> store / bm25_index   (offline indexing)
    retrieve -> rerank -> generate                     (online querying)
    eval                                               (how you measure it)
"""
# --- Quiet the cosmetic ML logs BEFORE any heavy library is imported. ---
# These env vars must be set before huggingface_hub / transformers load, which
# is why they live at package import time (the ML libs are imported lazily).
import os as _os
import warnings as _warnings
import logging as _logging

for _k, _v in {
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}.items():
    _os.environ.setdefault(_k, _v)

_warnings.filterwarnings("ignore")
for _name in ("transformers", "huggingface_hub", "sentence_transformers", "torch"):
    _logging.getLogger(_name).setLevel(_logging.ERROR)

__version__ = "0.1.0"
