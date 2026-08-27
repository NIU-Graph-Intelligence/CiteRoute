"""Evaluation metrics.

Formulas are identical to the masterset-benchmark baseline utils so that
CiteCoute numbers drop straight into the same result tables.
"""

import numpy as np

from .config import EVAL_K_VALUES


def compute_map(relevant, retrieved):
    if not relevant:
        return 0.0
    score, num_hits = 0.0, 0
    for i, paper_id in enumerate(retrieved):
        if paper_id in relevant:
            num_hits += 1
            score += num_hits / (i + 1)
    return score / len(relevant)


def compute_mrr(relevant, retrieved):
    for i, paper_id in enumerate(retrieved):
        if paper_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def compute_recall_at_k(relevant, retrieved, k):
    if not relevant:
        return 0.0
    hits = sum(1 for pid in retrieved[:k] if pid in relevant)
    return hits / len(relevant)


def compute_precision_at_k(relevant, retrieved, k):
    """Fraction of the top-k that is relevant (P@k)."""
    if k <= 0:
        return 0.0
    hits = sum(1 for pid in retrieved[:k] if pid in relevant)
    return hits / float(k)


def compute_hit_rate_at_k(relevant, retrieved, k):
    top_k = set(retrieved[:k])
    return 1.0 if any(item in top_k for item in relevant) else 0.0


def compute_dcg_at_k(relevant, retrieved, k):
    dcg = 0.0
    for i, paper_id in enumerate(retrieved[:k]):
        if paper_id in relevant:
            dcg += 1.0 / np.log2(i + 2)
    return dcg


def compute_ndcg_at_k(relevant, retrieved, k):
    if not relevant:
        return 0.0
    dcg = compute_dcg_at_k(relevant, retrieved, k)
    ideal_retrieved = list(relevant) + [0] * k
    idcg = compute_dcg_at_k(relevant, ideal_retrieved, k)
    return dcg / idcg if idcg > 0 else 0.0


def compute_all_metrics(relevant, retrieved):
    return {
        "map": compute_map(relevant, retrieved),
        "mrr": compute_mrr(relevant, retrieved),
        "recall": {k: compute_recall_at_k(relevant, retrieved, k) for k in EVAL_K_VALUES["recall"]},
        "ndcg": {k: compute_ndcg_at_k(relevant, retrieved, k) for k in EVAL_K_VALUES["ndcg"]},
        "hr": {k: compute_hit_rate_at_k(relevant, retrieved, k) for k in EVAL_K_VALUES["hr"]},
        "precision": {k: compute_precision_at_k(relevant, retrieved, k)
                      for k in EVAL_K_VALUES["precision"]},
    }


class MetricAccumulator:
    """Accumulates per-query metrics and aggregates them like the baselines."""

    def __init__(self):
        self.map_scores = []
        self.mrr_scores = []
        self.recall_scores = {k: [] for k in EVAL_K_VALUES["recall"]}
        self.ndcg_scores = {k: [] for k in EVAL_K_VALUES["ndcg"]}
        self.hr_scores = {k: [] for k in EVAL_K_VALUES["hr"]}
        self.precision_scores = {k: [] for k in EVAL_K_VALUES["precision"]}

    def add(self, relevant, retrieved):
        m = compute_all_metrics(relevant, retrieved)
        self.map_scores.append(m["map"])
        self.mrr_scores.append(m["mrr"])
        for k in EVAL_K_VALUES["recall"]:
            self.recall_scores[k].append(m["recall"][k])
        for k in EVAL_K_VALUES["ndcg"]:
            self.ndcg_scores[k].append(m["ndcg"][k])
        for k in EVAL_K_VALUES["hr"]:
            self.hr_scores[k].append(m["hr"][k])
        for k in EVAL_K_VALUES["precision"]:
            self.precision_scores[k].append(m["precision"][k])
        return m

    def result(self):
        if not self.map_scores:
            return None
        return {
            "num_queries": len(self.map_scores),
            "mean_map": float(np.mean(self.map_scores)),
            "mean_mrr": float(np.mean(self.mrr_scores)),
            "mean_recall": {k: float(np.mean(v)) for k, v in self.recall_scores.items()},
            "mean_ndcg": {k: float(np.mean(v)) for k, v in self.ndcg_scores.items()},
            "mean_hr": {k: float(np.mean(v)) for k, v in self.hr_scores.items()},
            "mean_precision": {k: float(np.mean(v)) for k, v in self.precision_scores.items()},
            "map_std": float(np.std(self.map_scores)),
            "mrr_std": float(np.std(self.mrr_scores)),
            "map_min": float(np.min(self.map_scores)),
            "map_max": float(np.max(self.map_scores)),
            "mrr_min": float(np.min(self.mrr_scores)),
            "mrr_max": float(np.max(self.mrr_scores)),
        }


def print_result(name, description, res):
    print(f"\n  {'=' * 70}")
    print(f"    {name}  ({description})")
    print(f"  {'=' * 70}")
    print(f"    Queries evaluated: {res['num_queries']}")
    print(f"    MAP:  {res['mean_map']:.6f}")
    print(f"    MRR:  {res['mean_mrr']:.6f}")
    print("    nDCG: " + "  ".join(f"@{k}: {res['mean_ndcg'][k]:.6f}" for k in EVAL_K_VALUES["ndcg"]))
    print("    Recall: " + "  ".join(f"@{k}: {res['mean_recall'][k]:.6f}" for k in EVAL_K_VALUES["recall"]))
    print("    HR:   " + "  ".join(f"@{k}: {res['mean_hr'][k]:.6f}" for k in EVAL_K_VALUES["hr"]))
    print("    P:    " + "  ".join(f"@{k}: {res['mean_precision'][k]:.6f}"
                                   for k in EVAL_K_VALUES["precision"]))
