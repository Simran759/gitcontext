"""Step 10 - Evaluation harness.

Each eval item names the file(s) that SHOULD be retrieved for its question.
We report several retrieval metrics plus latency:

- Recall@K   : fraction of questions where a gold file is in the top-K.
- Precision@K: of the K retrieved chunks, fraction from a gold file (noise).
- MRR        : mean reciprocal rank of the first gold hit (reward high ranks).
- Latency    : mean retrieval time per question (ms).

Correctness, citation accuracy, and faithfulness need gold ANSWERS (an
LLM-as-judge); this harness covers the retrieval layer, which caps everything
downstream. See the README for how to extend it to answer-quality metrics.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from .config import TOP_K_FINAL
from .pipeline import GitContext
from .retrieve import hybrid_search
from .generate import build_context, generate
from .judge import judge_answer


def evaluate(questions_path, k: int = 5) -> dict:
    gc = GitContext()
    items = json.loads(Path(questions_path).read_text(encoding="utf-8"))

    recall_hits = 0
    precision_sum = 0.0
    rr_sum = 0.0
    latencies = []

    for item in items:
        gold = set(item["gold_files"])

        t0 = time.perf_counter()
        cands = hybrid_search(item["question"], gc.embedder, gc.store, gc.bm25)[:k]
        latencies.append((time.perf_counter() - t0) * 1000)

        files = [c[2]["metadata"].get("file_path") for c in cands]

        # Recall@K: did any gold file appear in the top-K?
        hit = any(f in gold for f in files)
        recall_hits += int(hit)

        # Precision@K: fraction of the K chunks that come from a gold file.
        if files:
            precision_sum += sum(f in gold for f in files) / len(files)

        # Reciprocal rank: 1/(position of first gold hit), else 0.
        rr = 0.0
        for rank, f in enumerate(files, 1):
            if f in gold:
                rr = 1.0 / rank
                break
        rr_sum += rr

        print(f"[{'HIT ' if hit else 'MISS'}] (rr={rr:.2f}) {item['question']}")

    n = len(items) or 1
    report = {
        f"Recall@{k}": recall_hits / n,
        f"Precision@{k}": precision_sum / n,
        "MRR": rr_sum / n,
        "mean_latency_ms": sum(latencies) / n,
    }
    print("\n=== Retrieval report ===")
    for key, val in report.items():
        print(f"{key:18} = {val:.3f}" + (" ms" if key.endswith('ms') else ""))
    return report


def evaluate_answers(questions_path) -> dict:
    """Answer-quality layer: generate an answer per question and LLM-judge it."""
    gc = GitContext()
    items = json.loads(Path(questions_path).read_text(encoding="utf-8"))

    correctness, faithfulness, judged = 0, 0, 0
    for item in items:
        cands = hybrid_search(item["question"], gc.embedder, gc.store, gc.bm25)
        chunks = [c[2] for c in gc.reranker.rerank(item["question"], cands, TOP_K_FINAL)]
        answer, _ = generate(item["question"], chunks)
        answer = answer.split("] ", 1)[-1]              # drop the "[backend] " tag
        verdict = judge_answer(item["question"], answer, build_context(chunks))

        if verdict:
            judged += 1
            correctness += verdict["correctness"]
            faithfulness += verdict["faithfulness"]
            print(f"[c={verdict['correctness']} f={verdict['faithfulness']}] {item['question']}")
        else:
            print(f"[judge unavailable] {item['question']}")

    report = {}
    if judged:
        report = {"mean_correctness": correctness / judged,
                  "mean_faithfulness": faithfulness / judged,
                  "judged": judged}
        print("\n=== Answer-quality report ===")
        print(f"mean_correctness  = {report['mean_correctness']:.2f} / 5")
        print(f"mean_faithfulness = {report['mean_faithfulness']:.2f} / 5  ({judged} judged)")
    else:
        print("\nNo local judge model available (set up Ollama or the local model).")
    return report


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--answers" in args:                             # answer-quality mode
        args.remove("--answers")
        evaluate_answers(args[0] if args else "eval/questions.json")
    else:                                               # retrieval mode (default)
        path = args[0] if args else "eval/questions.json"
        k = int(args[1]) if len(args) > 1 else 5
        evaluate(path, k)
