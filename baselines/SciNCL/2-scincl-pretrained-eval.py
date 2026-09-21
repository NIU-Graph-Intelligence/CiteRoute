"""
SciNCL (Pretrained) — Evaluation Script
==========================================
Evaluates the official pretrained SciNCL model (malteos/scincl)
on all 3 label types. FAISS cosine retrieval from the candidate pool.
"""

import os
import sys
import json
import torch
import numpy as np
import polars as pl
import faiss
import time
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# -------- Config --------
load_dotenv()
ROOT_DIR = Path(os.getenv("ROOT_DIR", "/home/ratul/mustcite/"))
OUTPUT_DIR = ROOT_DIR / "masterset-benchmark/output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_PARQUET = ROOT_DIR / "data/train_eval_set/v7.0/eval_v7.0.parquet"
EMBED_DIR = OUTPUT_DIR / "dense/SciNCL-pretrained/embeddings/"
RESULTS_DIR = OUTPUT_DIR / "dense/SciNCL-pretrained/evaluation_results/"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TYPE_2_THRESHOLD = 4.0
TYPE_3_THRESHOLD = 3.0

LABEL_TYPES = {
    "type_1": {"name": "Type 1 (Binary Relevance)", "field": "type_1_output",
               "threshold_fn": lambda v: float(v) == 1.0, "description": "binary, label == 1"},
    "type_2": {"name": "Type 2 (Usefulness)", "field": "type_2_output",
               "threshold_fn": lambda v: float(v) >= TYPE_2_THRESHOLD, "description": f"usefulness >= {TYPE_2_THRESHOLD}"},
    "type_3": {"name": "Type 3 (Relatedness)", "field": "type_3_output",
               "threshold_fn": lambda v: float(v) >= TYPE_3_THRESHOLD, "description": f"relatedness >= {TYPE_3_THRESHOLD}"},
}
EVAL_K_VALUES = {"recall": [10, 50, 100, 500], "ndcg": [10, 20, 30, 50], "hr": [10, 20]}

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def extract_relevant_sets(references):
    relevant = {key: set() for key in LABEL_TYPES}
    for ref in references:
        mid = ref.get("matched_paper_id")
        if not mid: continue
        for key, cfg in LABEL_TYPES.items():
            raw_val = ref.get(cfg["field"])
            if raw_val is None: continue
            try:
                if cfg["threshold_fn"](raw_val): relevant[key].add(mid)
            except: continue
    return relevant

def compute_map(rel, ret):
    if not rel: return 0.0
    s, n = 0.0, 0
    for i, p in enumerate(ret):
        if p in rel: n += 1; s += n / (i + 1)
    return s / len(rel)

def compute_mrr(rel, ret):
    for i, p in enumerate(ret):
        if p in rel: return 1.0 / (i + 1)
    return 0.0

def compute_recall(rel, ret, k):
    if not rel: return 0.0
    return sum(1 for p in ret[:k] if p in rel) / len(rel)

def compute_hr(rel, ret, k):
    return 1.0 if any(p in set(ret[:k]) for p in rel) else 0.0

def compute_ndcg(rel, ret, k):
    if not rel: return 0.0
    dcg = sum(1.0 / np.log2(i + 2) for i, p in enumerate(ret[:k]) if p in rel)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    log("=" * 80)
    log("SciNCL (Pretrained) — Evaluation")
    log("=" * 80)

    cand_data = torch.load(EMBED_DIR / "candidates_embeddings.pt", map_location="cpu", weights_only=False)
    cand_embs = cand_data["embeddings"].numpy()
    cand_pids = cand_data["paper_ids"]

    # Gold refs outside the candidate pool (e.g., 2026 papers cited by 2026
    # queries) are unretrievable by construction — exclude them from gold.
    pool_ids = set(cand_pids)
    gold_outside_pool = {key: 0 for key in LABEL_TYPES}

    eval_data = torch.load(EMBED_DIR / "eval_embeddings.pt", map_location="cpu", weights_only=False)
    eval_embs = eval_data["embeddings"].numpy()
    with open(EMBED_DIR / "eval_paper_id_to_index.json") as f:
        eval_id_to_idx = json.load(f)

    eval_df = pl.read_parquet(EVAL_PARQUET)

    ground_truth = {key: {} for key in LABEL_TYPES}
    for row in eval_df.iter_rows(named=True):
        pid = row["paper_id"]
        if pid not in eval_id_to_idx: continue
        refs = json.loads(row["references"]) if isinstance(row["references"], str) else row["references"]
        rel = extract_relevant_sets(refs)
        for key in LABEL_TYPES:
            gold_outside_pool[key] += len(rel[key] - pool_ids)
            rel[key] &= pool_ids
            if rel[key]: ground_truth[key][pid] = rel[key]

    for key in LABEL_TYPES:
        if gold_outside_pool[key] > 0:
            log(f"  [{key}] gold refs outside candidate pool (excluded): {gold_outside_pool[key]}")

    t_norms = np.linalg.norm(cand_embs, axis=1, keepdims=True); t_norms[t_norms == 0] = 1.0
    cand_norm = (cand_embs / t_norms).astype("float32")
    e_norms = np.linalg.norm(eval_embs, axis=1, keepdims=True); e_norms[e_norms == 0] = 1.0
    eval_norm = (eval_embs / e_norms).astype("float32")

    index = faiss.IndexFlatIP(cand_norm.shape[1])
    index.add(cand_norm)

    all_qids = sorted(set().union(*[gt.keys() for gt in ground_truth.values()]))
    q_embs = np.array([eval_norm[eval_id_to_idx[pid]] for pid in all_qids]).astype("float32")
    max_k = max(EVAL_K_VALUES["recall"])
    _, indices = index.search(q_embs, max_k)

    retrieved_map = {}
    for i, qid in enumerate(all_qids):
        ret = [cand_pids[idx] for idx in indices[i] if cand_pids[idx] != qid]
        retrieved_map[qid] = ret

    all_results = {}
    for key, cfg in LABEL_TYPES.items():
        gt = ground_truth[key]
        if not gt: continue
        acc = {"map": [], "mrr": [], "recall": {k: [] for k in EVAL_K_VALUES["recall"]},
               "ndcg": {k: [] for k in EVAL_K_VALUES["ndcg"]}, "hr": {k: [] for k in EVAL_K_VALUES["hr"]}}
        for qid in tqdm(gt, desc=f"  {key}"):
            if qid not in retrieved_map: continue
            ret = retrieved_map[qid]
            acc["map"].append(compute_map(gt[qid], ret))
            acc["mrr"].append(compute_mrr(gt[qid], ret))
            for k in EVAL_K_VALUES["recall"]: acc["recall"][k].append(compute_recall(gt[qid], ret, k))
            for k in EVAL_K_VALUES["ndcg"]: acc["ndcg"][k].append(compute_ndcg(gt[qid], ret, k))
            for k in EVAL_K_VALUES["hr"]: acc["hr"][k].append(compute_hr(gt[qid], ret, k))
        all_results[key] = {
            "num_queries": len(acc["map"]),
            "mean_map": float(np.mean(acc["map"])), "mean_mrr": float(np.mean(acc["mrr"])),
            "mean_recall": {k: float(np.mean(v)) for k, v in acc["recall"].items()},
            "mean_ndcg": {k: float(np.mean(v)) for k, v in acc["ndcg"].items()},
            "mean_hr": {k: float(np.mean(v)) for k, v in acc["hr"].items()},
        }
        log(f"  {cfg['name']}: MAP={all_results[key]['mean_map']:.6f}")

    with open(RESULTS_DIR / "scincl_pretrained_metrics.txt", "w") as f:
        f.write("SciNCL (Pretrained: malteos/scincl) — No finetuning on our data\n\n")
        for key, cfg in LABEL_TYPES.items():
            if key not in all_results: continue
            r = all_results[key]
            f.write(f"--- {cfg['name']} ---\nQueries: {r['num_queries']}\nMAP: {r['mean_map']:.6f}  MRR: {r['mean_mrr']:.6f}\n")
            for k in EVAL_K_VALUES["ndcg"]: f.write(f"  nDCG@{k}: {r['mean_ndcg'][k]:.6f}\n")
            for k in EVAL_K_VALUES["recall"]: f.write(f"  Recall@{k}: {r['mean_recall'][k]:.6f}\n")
            for k in EVAL_K_VALUES["hr"]: f.write(f"  HR@{k}: {r['mean_hr'][k]:.6f}\n")
            f.write("\n")

    with open(RESULTS_DIR / "scincl_pretrained_metrics.json", "w") as f:
        json.dump({"model": "malteos/scincl", "type": "pretrained (no finetuning)", **all_results}, f, indent=2)

    log(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
