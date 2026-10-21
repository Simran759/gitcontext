"""Silence transformers/huggingface progress bars and load reports.

Called right before any heavy model is loaded. The env-var and stdlib-logging
setup in __init__.py handles the rest; this turns off the tqdm "Loading weights"
bars and the verbose weight-load reports that need the libraries imported first.
"""
from __future__ import annotations


def quiet_ml() -> None:
    try:
        from transformers.utils import logging as tlog
        tlog.set_verbosity_error()
        tlog.disable_progress_bar()
    except Exception:
        pass
    try:
        from huggingface_hub.utils import logging as hlog
        hlog.set_verbosity_error()
    except Exception:
        pass
    try:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    except Exception:
        pass
