"""
SciBERT + NT-Xent — Evaluation Script
=======================================
Evaluates each fine-tuned SciBERT model on its corresponding label type.

  - type_1 model → evaluated with type_1 ground truth only
  - type_2 model → evaluated with type_2 ground truth only
  - type_3 model → evaluated with type_3 ground truth only

Uses FAISS for fast cosine similarity search.
Retrieval is from the candidate pool (all papers <= 2025).

Resume capability:
  - Skips label types whose embeddings don't exist yet.
  - Evaluates whatever is available.

Input:
  - output/dense/SciBERT-NTXent/embeddings/<type_key>/candidates_embeddings.pt
  - output/dense/SciBERT-NTXent/embeddings/<type_key>/eval_embeddings.pt
  - data/train_eval_set/v7.0/eval_v7.0.parquet

Output:
  - output/dense/SciBERT-NTXent/evaluation_results/scibert_ntxent_evaluation_metrics.txt
  - output/dense/SciBERT-NTXent/evaluation_results/scibert_ntxent_evaluation_metrics.json
"""

import os
import sys
import json
import torch
import numpy as np
import polars as pl
import faiss
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

from scibert_ntx_utils import (
    log, load_paper_title_map,
    extract_relevant_sets, compute_all_metrics,
    LABEL_TYPES, EVAL_K_VALUES,
    TYPE_2_THRESHOLD, TYPE_3_THRESHOLD,
)

# -------- Config --------
load_dotenv()
ROOT_DIR = Path(os.getenv("ROOT_DIR", "/home/ratul/mustcite/"))
OUTPUT_DIR = ROOT_DIR / "masterset-benchmark/output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_PARQUET = ROOT_DIR / "data/train_eval_set/v7.0/candidate_pool_v7.0.parquet"
EVAL_PARQUET = ROOT_DIR / "data/train_eval_set/v7.0/eval_v7.0.parquet"

EMBEDDINGS_BASE_DIR = OUTPUT_DIR / "dense/SciBERT-NTXent/embeddings/"

RESULTS_DIR = OUTPUT_DIR / "dense/SciBERT-NTXent/evaluation_results/"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "scibert_ntxent_evaluation_metrics.txt"
RESULTS_JSON = RESULTS_DIR / "scibert_ntxent_evaluation_metrics.json"


# ======================================================================
# Evaluate a single label type
# ======================================================================

def evaluate_label_type(label_key, eval_df, paper_title_map):
    """
    Evaluate one label type's fine-tuned embeddings.
    Returns results dict or None if embeddings not available.
    """
    cfg = LABEL_TYPES[label_key]
    emb_dir = EMBEDDINGS_BASE_DIR / label_key

    cand_emb_path = emb_dir / "candidates_embeddings.pt"
    eval_emb_path = emb_dir / "eval_embeddings.pt"
    cand_map_path = emb_dir / "candidates_paper_id_to_index.json"
    eval_map_path = emb_dir / "eval_paper_id_to_index.json"

    # ---- Check if embeddings exist ----
    for p in [cand_emb_path, eval_emb_path, cand_map_path, eval_map_path]:
        if not p.exists():
            log(f"  Missing: {p} — skipping {cfg['name']}.")
            return None

    # ---- Load embeddings ----
    log(f"  Loading train embeddings...")
    cand_data = torch.load(cand_emb_path, map_location="cpu", weights_only=False)
    cand_embeddings = cand_data["embeddings"].numpy()
    cand_paper_ids = cand_data["paper_ids"]

    # Gold refs outside the candidate pool are unretrievable by construction
    pool_ids = set(cand_paper_ids)
    gold_outside_pool = 0

    log(f"  Loading eval embeddings...")
    eval_data = torch.load(eval_emb_path, map_location="cpu", weights_only=False)
    eval_embeddings = eval_data["embeddings"].numpy()
    eval_paper_ids = eval_data["paper_ids"]

    with open(eval_map_path, "r") as f:
        eval_id_to_idx = json.load(f)

    log(f"  Train: {len(cand_paper_ids)} papers, Eval: {len(eval_paper_ids)} papers")
    log(f"  Embedding dim: {cand_embeddings.shape[1]}")

    # ---- Build ground truth for THIS label type only ----
    ground_truth = {}
    for row in eval_df.iter_rows(named=True):
        paper_id = row["paper_id"]
        if paper_id not in eval_id_to_idx:
            continue

        references = json.loads(row["references"]) if isinstance(row["references"], str) else row["references"]
        relevant_sets = extract_relevant_sets(references)
        gold_outside_pool += len(relevant_sets[label_key] - pool_ids)
        relevant = relevant_sets[label_key] & pool_ids

        if relevant:
            ground_truth[paper_id] = relevant

    if gold_outside_pool > 0:
        log(f"  Gold refs outside candidate pool (excluded): {gold_outside_pool}")
    log(f"  Queries with relevant papers: {len(ground_truth)}")
    if len(ground_truth) == 0:
        log(f"  No queries with relevant papers — skipping.")
        return None

    avg_pos = np.mean([len(refs) for refs in ground_truth.values()])
    log(f"  Avg positives/query: {avg_pos:.2f}")

    # ---- Normalize embeddings for cosine similarity ----
    cand_norms = np.linalg.norm(cand_embeddings, axis=1, keepdims=True)
    cand_norms[cand_norms == 0] = 1.0
    cand_embeddings_norm = cand_embeddings / cand_norms

    eval_norms = np.linalg.norm(eval_embeddings, axis=1, keepdims=True)
    eval_norms[eval_norms == 0] = 1.0
    eval_embeddings_norm = eval_embeddings / eval_norms

    # ---- Build FAISS index from TRAIN embeddings only ----
    log(f"  Building FAISS index (candidate pool papers)...")
    embedding_dim = cand_embeddings_norm.shape[1]
    faiss_index = faiss.IndexFlatIP(embedding_dim)
    faiss_index.add(cand_embeddings_norm.astype("float32"))

    # ---- Prepare query embeddings ----
    query_embeddings = []
    query_ids = []
    for paper_id in ground_truth.keys():
        idx = eval_id_to_idx[paper_id]
        query_embeddings.append(eval_embeddings_norm[idx])
        query_ids.append(paper_id)

    query_embeddings = np.array(query_embeddings).astype("float32")
    log(f"  Prepared {len(query_embeddings)} query embeddings")

    # ---- Retrieve ----
    max_k = max(EVAL_K_VALUES["recall"])
    log(f"  Performing FAISS retrieval with k={max_k}...")
    distances, indices = faiss_index.search(query_embeddings, max_k)

    # ---- Compute metrics ----
    log(f"  Computing metrics...")
    acc = {
        "map_scores": [], "mrr_scores": [],
        "recall_scores": {k: [] for k in EVAL_K_VALUES["recall"]},
        "ndcg_scores": {k: [] for k in EVAL_K_VALUES["ndcg"]},
        "hr_scores": {k: [] for k in EVAL_K_VALUES["hr"]},
    }

    for i, query_id in enumerate(tqdm(query_ids, desc=f"  Evaluating {label_key}")):
        retrieved_paper_ids = [cand_paper_ids[idx] for idx in indices[i]]
        retrieved_paper_ids = [pid for pid in retrieved_paper_ids if pid != query_id]

        relevant = ground_truth[query_id]
        metrics = compute_all_metrics(relevant, retrieved_paper_ids)

        acc["map_scores"].append(metrics["map"])
        acc["mrr_scores"].append(metrics["mrr"])
        for k in EVAL_K_VALUES["recall"]:
            acc["recall_scores"][k].append(metrics["recall"][k])
        for k in EVAL_K_VALUES["ndcg"]:
            acc["ndcg_scores"][k].append(metrics["ndcg"][k])
        for k in EVAL_K_VALUES["hr"]:
            acc["hr_scores"][k].append(metrics["hr"][k])

        # Show examples for first 3 queries
        if i < 3:
            query_title = paper_title_map.get(query_id, f"ID: {query_id}")
            print(f"\n    --- Query {i+1}: {query_title} ---")
            print(f"    Ground truth positives: {len(relevant)}")
            print(f"    Top 5 retrieved:")
            for rank, pid in enumerate(retrieved_paper_ids[:5]):
                title = paper_title_map.get(pid, f"ID: {pid}")
                is_relevant = "✓" if pid in relevant else "✗"
                sim = distances[i][rank] if rank < len(distances[i]) else 0.0
                print(f"      {rank+1}. [{is_relevant}] (sim: {sim:.3f}) {title}")

    # ---- Aggregate ----
    n_q = len(acc["map_scores"])
    result = {
        "num_queries": n_q,
        "mean_map": float(np.mean(acc["map_scores"])),
        "mean_mrr": float(np.mean(acc["mrr_scores"])),
        "mean_recall": {k: float(np.mean(v)) for k, v in acc["recall_scores"].items()},
        "mean_ndcg": {k: float(np.mean(v)) for k, v in acc["ndcg_scores"].items()},
        "mean_hr": {k: float(np.mean(v)) for k, v in acc["hr_scores"].items()},
        "map_std": float(np.std(acc["map_scores"])),
        "mrr_std": float(np.std(acc["mrr_scores"])),
        "map_min": float(np.min(acc["map_scores"])),
        "map_max": float(np.max(acc["map_scores"])),
        "mrr_min": float(np.min(acc["mrr_scores"])),
        "mrr_max": float(np.max(acc["mrr_scores"])),
    }

    # Print
    print(f"\n  {'=' * 70}")
    print(f"    {cfg['name']}  ({cfg['description']})")
    print(f"  {'=' * 70}")
    print(f"    Queries evaluated: {n_q}")
    print(f"    MAP:  {result['mean_map']:.6f}")
    print(f"    MRR:  {result['mean_mrr']:.6f}")
    print(f"    nDCG: ", end="")
    print("  ".join(f"@{k}: {result['mean_ndcg'][k]:.6f}" for k in EVAL_K_VALUES["ndcg"]))
    print(f"    Recall: ", end="")
    print("  ".join(f"@{k}: {result['mean_recall'][k]:.6f}" for k in EVAL_K_VALUES["recall"]))
    print(f"    HR:   ", end="")
    print("  ".join(f"@{k}: {result['mean_hr'][k]:.6f}" for k in EVAL_K_VALUES["hr"]))

    return result


# ======================================================================
# Main
# ======================================================================

def main():
    log("=" * 80)
    log("SciBERT + NT-Xent — Evaluation")
    log("=" * 80)
    log(f"Python: {sys.version.split()[0]}")

    # Load eval ground truth
    log("\nLoading evaluation dataset...")
    eval_df = pl.read_parquet(EVAL_PARQUET)
    log(f"  Total eval papers: {eval_df.height}")

    # Load title map for debug printing
    paper_title_map = load_paper_title_map(CANDIDATE_PARQUET)

    # ---- Evaluate each label type ----
    all_results = {}

    for label_key, label_cfg in LABEL_TYPES.items():
        log(f"\n{'=' * 80}")
        log(f"Evaluating: {label_cfg['name']}  ({label_cfg['description']})")
        log(f"{'=' * 80}")

        result = evaluate_label_type(label_key, eval_df, paper_title_map)
        if result is not None:
            all_results[label_key] = result

    if not all_results:
        log("\nNo label types were evaluated. Generate embeddings first.")
        return

    # ---- Save results to text file ----
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("SCIBERT + NT-XENT EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Method: SciBERT fine-tuned with NT-Xent contrastive loss\n")
        f.write(f"Base model: allenai/scibert_scivocab_uncased\n")
        f.write(f"Projection dim: 128\n")
        f.write(f"Text: title + abstract\n")
        f.write(f"Note: Each label type has its own fine-tuned model.\n")
        f.write(f"      Retrieval is from train set only.\n\n")

        for key in LABEL_TYPES:
            if key not in all_results:
                f.write(f"-" * 80 + "\n")
                f.write(f"  {LABEL_TYPES[key]['name']} — NOT EVALUATED (embeddings not available)\n\n")
                continue

            cfg = LABEL_TYPES[key]
            res = all_results[key]
            f.write("-" * 80 + "\n")
            f.write(f"  {cfg['name']}  ({cfg['description']})\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Queries evaluated: {res['num_queries']}\n\n")

            f.write(f"  Mean Average Precision (MAP): {res['mean_map']:.6f}\n")
            f.write(f"  Mean Reciprocal Rank (MRR):   {res['mean_mrr']:.6f}\n\n")

            f.write("  Normalized Discounted Cumulative Gain:\n")
            for k in EVAL_K_VALUES["ndcg"]:
                f.write(f"    nDCG@{k:3d}: {res['mean_ndcg'][k]:.6f}\n")
            f.write("\n")

            f.write("  Recall Metrics:\n")
            for k in EVAL_K_VALUES["recall"]:
                f.write(f"    Recall@{k:3d}: {res['mean_recall'][k]:.6f}\n")
            f.write("\n")

            f.write("  Hit Rate Metrics:\n")
            for k in EVAL_K_VALUES["hr"]:
                f.write(f"    HR@{k:3d}: {res['mean_hr'][k]:.6f}\n")
            f.write("\n")

            f.write("  Detailed Statistics:\n")
            f.write(f"    MAP - Min: {res['map_min']:.6f}, Max: {res['map_max']:.6f}, Std: {res['map_std']:.6f}\n")
            f.write(f"    MRR - Min: {res['mrr_min']:.6f}, Max: {res['mrr_max']:.6f}, Std: {res['mrr_std']:.6f}\n")
            f.write("\n")

    # ---- Save results to JSON ----
    results_dict = {
        "model_info": {
            "method": "SciBERT + NT-Xent Contrastive Fine-Tuning",
            "base_model": "allenai/scibert_scivocab_uncased",
            "projection_dim": 128,
            "text_source": "title + abstract",
            "retrieval_pool": "train set only",
            "note": "Each label type has a separately fine-tuned model",
        },
        "label_thresholds": {
            "type_1": "binary, label == 1",
            "type_2": f"usefulness >= {TYPE_2_THRESHOLD}",
            "type_3": f"relatedness >= {TYPE_3_THRESHOLD}",
        },
    }

    for key, cfg in LABEL_TYPES.items():
        if key not in all_results:
            results_dict[key] = {"status": "not_evaluated"}
            continue

        res = all_results[key]
        results_dict[key] = {
            "name": cfg["name"],
            "description": cfg["description"],
            "num_queries": res["num_queries"],
            "metrics": {
                "map": res["mean_map"],
                "mrr": res["mean_mrr"],
                "ndcg": {f"ndcg@{k}": res["mean_ndcg"][k] for k in EVAL_K_VALUES["ndcg"]},
                "recall": {f"recall@{k}": res["mean_recall"][k] for k in EVAL_K_VALUES["recall"]},
                "hit_rate": {f"hr@{k}": res["mean_hr"][k] for k in EVAL_K_VALUES["hr"]},
            },
            "statistics": {
                "map_std": res["map_std"],
                "mrr_std": res["mrr_std"],
            },
        }

    with open(RESULTS_JSON, "w") as f:
        json.dump(results_dict, f, indent=2)

    log(f"\nResults saved to:")
    log(f"  Text: {RESULTS_FILE}")
    log(f"  JSON: {RESULTS_JSON}")


if __name__ == "__main__":
    main()