"""LLM-as-judge: score answer quality.

Retrieval metrics (Recall@K ...) tell you if the right chunk was found, but not
whether the final ANSWER was good. Without hand-written gold answers, a common
technique is to ask an LLM to grade the answer. We score two axes 1-5:

- correctness : does the answer actually answer the question?
- faithfulness: is it supported by the retrieved context (no invented facts)?

Uses the same local model as generation (no API key). The score PARSER is a
pure function so it can be unit-tested without calling a model.
"""
from __future__ import annotations
import json
import re
from typing import Optional
from .generate import complete

_SYS = (
    "You are a strict evaluator. Given a QUESTION, an ANSWER, and the CONTEXT the "
    "answer was supposed to use, rate the answer from 1 to 5 on two axes:\n"
    "- correctness: does it correctly answer the question?\n"
    "- faithfulness: is every claim supported by the context (no made-up facts)?\n"
    'Respond with ONLY compact JSON: {"correctness": N, "faithfulness": N}.'
)


def _clip(n: int) -> int:
    return max(1, min(5, n))


def _parse_scores(text: str) -> Optional[dict]:
    """Pull {correctness, faithfulness} out of a model's reply, robustly."""
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            return {"correctness": _clip(int(d["correctness"])),
                    "faithfulness": _clip(int(d["faithfulness"]))}
        except Exception:
            pass
    # Fallback: first two digits in 1-5 (small models wander off JSON).
    nums = [int(n) for n in re.findall(r"[1-5]", text)]
    if len(nums) >= 2:
        return {"correctness": nums[0], "faithfulness": nums[1]}
    return None


def judge_answer(question: str, answer: str, context: str) -> Optional[dict]:
    user = f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nCONTEXT:\n{context}"
    text, _ = complete(_SYS, user, max_new_tokens=60)
    return _parse_scores(text) if text else None
