"""Shared inference path: fuse per-view runs, optionally rerank, evaluate.

Used by Stage 7 (main results) and Stage 8 (ablations) so both follow exactly
the same code path — an ablation differs only in which views are enabled and
whether the reranker is applied.
"""

import torch

from .config import (RERANK_DEPTH, RERANKER_INFER_BATCH_SIZE,
                     RERANKER_MAX_LEN, EVAL_K_VALUES)
from .fusion import rrf_fuse
from .metrics import MetricAccumulator
from .rerank import build_query_text
from .utils import log

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def rerank_candidates(model, tokenizer, query_text, candidates, doc_text,
                      batch_size=RERANKER_INFER_BATCH_SIZE):
    """Rescore candidates with the cross-encoder; returns them best-first."""
    usable = [c for c in candidates if doc_text.get(c)]
    if not usable:
        return candidates
    scores = []
    for s in range(0, len(usable), batch_size):
        chunk = usable[s:s + batch_size]
        enc = tokenizer([query_text] * len(chunk), [doc_text[c] for c in chunk],
                        max_length=RERANKER_MAX_LEN, padding="max_length",
                        truncation=True, return_tensors="pt")
        kwargs = {"input_ids": enc["input_ids"].to(DEVICE),
                  "attention_mask": enc["attention_mask"].to(DEVICE)}
        if "token_type_ids" in enc:
            kwargs["token_type_ids"] = enc["token_type_ids"].to(DEVICE)
        scores.extend(model(**kwargs).float().cpu().tolist())
    order = sorted(range(len(usable)), key=lambda i: -scores[i])
    reranked = [usable[i] for i in order]
    # Candidates the reranker could not score keep their fused order, appended.
    tail = [c for c in candidates if c not in set(usable)]
    return reranked + tail


def evaluate(per_query, ground_truth, weights=None, views=None,
             reranker=None, tokenizer=None, facet_map=None, doc_text=None,
             rerank_depth=RERANK_DEPTH, show_examples=0, title_map=None,
             progress_desc=None):
    """Fuse -> (optionally) rerank -> accumulate metrics.

    per_query    : {query_id: {view: [doc_id, ...]}}
    ground_truth : {query_id: set(gold_ids)}
    views        : subset of views to fuse (None = all present) — this is the
                   single knob every facet ablation turns.
    """
    acc = MetricAccumulator()
    max_k = max(EVAL_K_VALUES["recall"])
    queries = [q for q in per_query if q in ground_truth]

    iterator = queries
    try:
        from tqdm import tqdm
        iterator = tqdm(queries, desc=progress_desc or "  Evaluating")
    except ImportError:
        pass

    shown = 0
    for qid in iterator:
        view_runs = per_query[qid]
        if views is not None:
            view_runs = {v: r for v, r in view_runs.items() if v in views}
            if not view_runs:
                continue

        fused, _ = rrf_fuse(view_runs, weights=weights, depth=max_k)
        fused = [p for p in fused if p != qid]

        if reranker is not None:
            head = fused[:rerank_depth]
            tail = fused[rerank_depth:]
            q_text = build_query_text(facet_map.get(qid) if facet_map else None,
                                      (facet_map or {}).get(qid, {}).get("full", ""))
            fused = rerank_candidates(reranker, tokenizer, q_text, head, doc_text) + tail

        m = acc.add(ground_truth[qid], fused)

        if shown < show_examples:
            shown += 1
            tm = title_map or {}
            print(f"\n    --- Query {shown}: {tm.get(qid, qid)} ---")
            print(f"    Ground truth positives: {len(ground_truth[qid])}")
            print("    Top 5 retrieved:")
            for rank, pid in enumerate(fused[:5], start=1):
                mark = "OK " if pid in ground_truth[qid] else "-- "
                print(f"      {rank}. [{mark}] {tm.get(pid, pid)}")
            print(f"    Recall@100 for this query: {m['recall'][100]:.4f}")

    return acc.result()
