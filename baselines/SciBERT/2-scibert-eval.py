"""
SciBERT Baseline — Evaluation Script
======================================
Evaluates SciBERT (pure CLS embedding) retrieval on the eval set.

Uses FAISS for fast cosine similarity search. Retrieves top-K candidates
from the train set for each eval query, then computes metrics for all
3 label types independently.

Label Types:
  - Type 1 (binary):      label == 1 → relevant
  - Type 2 (usefulness):  label >= 4 → relevant
  - Type 3 (relatedness): label >= 3 → relevant

Metrics (per label type):
  - MAP (Mean Average Precision)
  - MRR (Mean Reciprocal Rank)
  - Recall@{10, 50, 100, 500}
  - nDCG@{10, 20, 30, 50}
  - HR@{10, 20} (Hit Rate)

Input:
  - output/dense/SciBERT/embeddings/candidates_embeddings.pt
  - output/dense/SciBERT/embeddings/eval_embeddings.pt
  - output/dense/SciBERT/embeddings/*_paper_id_to_index.json
  - data/train_eval_set/v7.0/eval_v7.0.parquet

Output:
  - output/dense/SciBERT/evaluation_results/scibert_evaluation_metrics.txt
  - output/dense/SciBERT/evaluation_results/scibert_evaluation_metrics.json
"""

import os
import torch
import polars as pl
from pathlib import Path
from tqdm import tqdm
import numpy as np
import faiss
import json
from dotenv import load_dotenv

from scibert_utils import (
    load_paper_title_map,
    average_precision,
    recall_at_k,
    hit_rate_at_k,
    reciprocal_rank,
    ndcg_at_k,
    SEP,
)

# -------- Config --------
load_dotenv()
ROOT_DIR = Path(os.getenv("ROOT_DIR", "/home/ratul/mustcite/"))
OUTPUT_DIR = ROOT_DIR / "masterset-benchmark/output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_PARQUET = ROOT_DIR / "data/train_eval_set/v7.0/candidate_pool_v7.0.parquet"
EVAL_PARQUET = ROOT_DIR / "data/train_eval_set/v7.0/eval_v7.0.parquet"

EMBEDDINGS_DIR = OUTPUT_DIR / "dense/SciBERT/embeddings/"

TRAIN_EMBEDDINGS_PATH = EMBEDDINGS_DIR / "candidates_embeddings.pt"
EVAL_EMBEDDINGS_PATH = EMBEDDINGS_DIR / "eval_embeddings.pt"
TRAIN_PAPER_ID_INDEX_MAP_PATH = EMBEDDINGS_DIR / "candidates_paper_id_to_index.json"
EVAL_PAPER_ID_INDEX_MAP_PATH = EMBEDDINGS_DIR / "eval_paper_id_to_index.json"
METADATA_FILE_PATH = EMBEDDINGS_DIR / "metadata.json"

RESULTS_DIR = OUTPUT_DIR / "dense/SciBERT/evaluation_results/"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "scibert_evaluation_metrics.txt"
RESULTS_JSON = RESULTS_DIR / "scibert_evaluation_metrics.json"

# Metrics to compute (matching BM25 evaluation convention)
EVAL_K_VALUES = {
    "recall": [10, 50, 100, 500],
    "ndcg": [10, 20, 30, 50],
    "hr": [10, 20],
}

# ---- Label thresholds ----
TYPE_2_THRESHOLD = 4.0
TYPE_3_THRESHOLD = 3.0

LABEL_TYPES = {
    "type_1": {
        "name": "Type 1 (Binary Relevance)",
        "field": "type_1_output",
        "threshold_fn": lambda v: float(v) == 1.0,
        "description": "binary, label == 1",
    },
    "type_2": {
        "name": "Type 2 (Usefulness)",
        "field": "type_2_output",
        "threshold_fn": lambda v: float(v) >= TYPE_2_THRESHOLD,
        "description": f"usefulness >= {TYPE_2_THRESHOLD}",
    },
    "type_3": {
        "name": "Type 3 (Relatedness)",
        "field": "type_3_output",
        "threshold_fn": lambda v: float(v) >= TYPE_3_THRESHOLD,
        "description": f"relatedness >= {TYPE_3_THRESHOLD}",
    },
}


# ======================================================================
# Ground truth extraction
# ======================================================================

def extract_relevant_sets(references):
    """
    Given a list of reference dicts, extract the relevant paper_id sets
    for each label type.

    Returns dict: label_type_key -> set of relevant matched_paper_ids
    """
    relevant = {key: set() for key in LABEL_TYPES}

    for ref in references:
        matched_paper_id = ref.get("matched_paper_id")
        if not matched_paper_id:
            continue

        for key, cfg in LABEL_TYPES.items():
            field = cfg["field"]
            raw_val = ref.get(field)
            if raw_val is None:
                continue
            try:
                if cfg["threshold_fn"](raw_val):
                    relevant[key].add(matched_paper_id)
            except (ValueError, TypeError):
                continue

    return relevant


# ======================================================================
# Per-query metric computation
# ======================================================================

def compute_all_metrics(retrieved_ids, ground_truth_ids):
    """Compute all metrics for a single query given retrieved list and ground truth set."""
    return {
        "map": average_precision(retrieved_ids, ground_truth_ids),
        "mrr": reciprocal_rank(retrieved_ids, ground_truth_ids),
        "recall": {k: recall_at_k(retrieved_ids, ground_truth_ids, k) for k in EVAL_K_VALUES["recall"]},
        "ndcg": {k: ndcg_at_k(retrieved_ids, ground_truth_ids, k) for k in EVAL_K_VALUES["ndcg"]},
        "hr": {k: hit_rate_at_k(retrieved_ids, ground_truth_ids, k) for k in EVAL_K_VALUES["hr"]},
    }


# ======================================================================
# Main Evaluation
# ======================================================================

def main():
    """Evaluate SciBERT embeddings on the evaluation set with 3 label types."""
    print("\n" + "=" * 80)
    print("SciBERT Pure Evaluation (3 Label Types)")
    print("=" * 80 + "\n")

    # Load paper title mapping (for debug printing)
    print("Loading paper titles...")
    paper_title_map = load_paper_title_map(CANDIDATE_PARQUET)

    # ---- Load train embeddings ----
    print("Loading train embeddings...")
    if not TRAIN_EMBEDDINGS_PATH.exists():
        print(f"Error: Candidate embeddings file not found at {TRAIN_EMBEDDINGS_PATH}")
        print("Please run the embedding generation script first.")
        return

    if not TRAIN_PAPER_ID_INDEX_MAP_PATH.exists():
        print(f"Error: Candidate ID mapping file not found at {TRAIN_PAPER_ID_INDEX_MAP_PATH}")
        print("Please run the embedding generation script first.")
        return

    cand_data = torch.load(TRAIN_EMBEDDINGS_PATH, weights_only=False)
    cand_embeddings = cand_data["embeddings"].numpy()
    cand_paper_ids = cand_data["paper_ids"]

    with open(TRAIN_PAPER_ID_INDEX_MAP_PATH, "r") as f:
        cand_id_to_idx = json.load(f)
        # paper_ids are UUID strings, keep as-is (no int conversion)

    # Gold refs outside the candidate pool are unretrievable by construction —
    # exclude them from the gold sets below.
    pool_ids = set(cand_paper_ids)
    gold_outside_pool = {key: 0 for key in LABEL_TYPES}

    print(f"  Loaded candidate embeddings for {len(cand_paper_ids)} papers")
    print(f"  Embedding dimension: {cand_embeddings.shape[1]}")

    # ---- Load eval embeddings ----
    print("\nLoading eval embeddings...")
    if not EVAL_EMBEDDINGS_PATH.exists():
        print(f"Error: Eval embeddings file not found at {EVAL_EMBEDDINGS_PATH}")
        return

    eval_data = torch.load(EVAL_EMBEDDINGS_PATH, weights_only=False)
    eval_embeddings = eval_data["embeddings"].numpy()
    eval_paper_ids = eval_data["paper_ids"]

    with open(EVAL_PAPER_ID_INDEX_MAP_PATH, "r") as f:
        eval_id_to_idx = json.load(f)
        # paper_ids are UUID strings, keep as-is

    print(f"  Loaded eval embeddings for {len(eval_paper_ids)} papers")

    # ---- Load evaluation ground truth ----
    print("\nLoading evaluation dataset...")
    eval_df = pl.read_parquet(EVAL_PARQUET)

    # Build ground truth mappings for all 3 label types
    # ground_truth[label_type_key] = {paper_id: set(relevant_matched_paper_ids)}
    ground_truth = {key: {} for key in LABEL_TYPES}
    total_eval_papers = 0

    for row in eval_df.iter_rows(named=True):
        total_eval_papers += 1
        paper_id = row["paper_id"]

        # Skip if we don't have embeddings for this paper
        if paper_id not in eval_id_to_idx:
            continue

        references = json.loads(row["references"]) if isinstance(row["references"], str) else row["references"]
        relevant_sets = extract_relevant_sets(references)

        for key in LABEL_TYPES:
            gold_outside_pool[key] += len(relevant_sets[key] - pool_ids)
            relevant_sets[key] &= pool_ids
            if relevant_sets[key]:  # only add if there are relevant papers
                ground_truth[key][paper_id] = relevant_sets[key]

    for key in LABEL_TYPES:
        if gold_outside_pool[key] > 0:
            print(f"  [{key}] gold refs outside candidate pool (excluded): {gold_outside_pool[key]}")

    for key, cfg in LABEL_TYPES.items():
        n_queries = len(ground_truth[key])
        avg_pos = np.mean([len(refs) for refs in ground_truth[key].values()]) if n_queries > 0 else 0
        print(f"  {cfg['name']}: {n_queries} queries with relevant papers (avg {avg_pos:.2f} positives/query)")

    # ---- Normalize embeddings for cosine similarity ----
    print("\nNormalizing embeddings for cosine similarity...")
    cand_norms = np.linalg.norm(cand_embeddings, axis=1, keepdims=True)
    cand_norms[cand_norms == 0] = 1.0  # avoid division by zero
    cand_embeddings_norm = cand_embeddings / cand_norms

    eval_norms = np.linalg.norm(eval_embeddings, axis=1, keepdims=True)
    eval_norms[eval_norms == 0] = 1.0
    eval_embeddings_norm = eval_embeddings / eval_norms

    # ---- Build FAISS index ----
    print("Building FAISS index from train embeddings...")
    embedding_dim = cand_embeddings_norm.shape[1]
    faiss_index = faiss.IndexFlatIP(embedding_dim)  # Inner product = cosine sim on normalized vectors
    faiss_index.add(cand_embeddings_norm.astype("float32"))

    # ---- Collect all unique query paper_ids across all label types ----
    all_query_ids = set()
    for key in LABEL_TYPES:
        all_query_ids.update(ground_truth[key].keys())
    all_query_ids = sorted(all_query_ids)

    # Prepare query embeddings
    query_embeddings = []
    query_ids = []
    for paper_id in all_query_ids:
        idx = eval_id_to_idx[paper_id]
        query_embeddings.append(eval_embeddings_norm[idx])
        query_ids.append(paper_id)

    query_embeddings = np.array(query_embeddings).astype("float32")
    print(f"Prepared {len(query_embeddings)} unique query embeddings")

    # ---- Perform retrieval (ONCE for all label types) ----
    max_k = max(EVAL_K_VALUES["recall"])
    print(f"\nPerforming FAISS retrieval with k={max_k}...")
    distances, indices = faiss_index.search(query_embeddings, max_k)

    # ---- Initialize per-label-type metric accumulators ----
    accumulators = {}
    for key in LABEL_TYPES:
        accumulators[key] = {
            "map_scores": [],
            "mrr_scores": [],
            "recall_scores": {k: [] for k in EVAL_K_VALUES["recall"]},
            "ndcg_scores": {k: [] for k in EVAL_K_VALUES["ndcg"]},
            "hr_scores": {k: [] for k in EVAL_K_VALUES["hr"]},
            "num_queries": 0,
        }

    # ---- Evaluate ----
    print("Calculating metrics...")
    for i, query_id in enumerate(tqdm(query_ids, desc="Evaluating queries")):
        # Get retrieved paper IDs from train set
        retrieved_indices = indices[i]
        retrieved_paper_ids = [cand_paper_ids[idx] for idx in retrieved_indices]

        # Remove self from results
        retrieved_paper_ids = [pid for pid in retrieved_paper_ids if pid != query_id]

        # Show example for first 3 queries (using type_1 ground truth for display)
        if i < 3:
            query_title = paper_title_map.get(query_id, f"ID: {query_id}")
            gt_type1 = ground_truth["type_1"].get(query_id, set())
            print(f"\n--- Query {i+1}: {query_title} ---")
            print(f"  Type 1 ground truth positives: {len(gt_type1)}")
            print("  Top 5 retrieved papers:")
            for rank, pid in enumerate(retrieved_paper_ids[:5]):
                title = paper_title_map.get(pid, f"ID: {pid}")
                is_relevant = "✓" if pid in gt_type1 else "✗"
                similarity = distances[i][rank] if rank < len(distances[i]) else 0.0
                print(f"    {rank+1}. [{is_relevant}] (sim: {similarity:.3f}) {title}")

        # Compute metrics for each label type independently
        for key in LABEL_TYPES:
            gt_set = ground_truth[key].get(query_id)
            if gt_set is None:
                continue  # This query has no relevant papers for this label type

            acc = accumulators[key]
            acc["num_queries"] += 1

            metrics = compute_all_metrics(retrieved_paper_ids, gt_set)
            acc["map_scores"].append(metrics["map"])
            acc["mrr_scores"].append(metrics["mrr"])
            for k in EVAL_K_VALUES["recall"]:
                acc["recall_scores"][k].append(metrics["recall"][k])
            for k in EVAL_K_VALUES["ndcg"]:
                acc["ndcg_scores"][k].append(metrics["ndcg"][k])
            for k in EVAL_K_VALUES["hr"]:
                acc["hr_scores"][k].append(metrics["hr"][k])

    # ---- Compute means and print results ----
    all_results = {}

    for key, cfg in LABEL_TYPES.items():
        acc = accumulators[key]
        n_q = acc["num_queries"]

        mean_map = np.mean(acc["map_scores"]) if acc["map_scores"] else 0.0
        mean_mrr = np.mean(acc["mrr_scores"]) if acc["mrr_scores"] else 0.0
        mean_recall = {k: np.mean(v) if v else 0.0 for k, v in acc["recall_scores"].items()}
        mean_ndcg = {k: np.mean(v) if v else 0.0 for k, v in acc["ndcg_scores"].items()}
        mean_hr = {k: np.mean(v) if v else 0.0 for k, v in acc["hr_scores"].items()}

        all_results[key] = {
            "num_queries": n_q,
            "mean_map": float(mean_map),
            "mean_mrr": float(mean_mrr),
            "mean_recall": {k: float(v) for k, v in mean_recall.items()},
            "mean_ndcg": {k: float(v) for k, v in mean_ndcg.items()},
            "mean_hr": {k: float(v) for k, v in mean_hr.items()},
            "map_std": float(np.std(acc["map_scores"])) if acc["map_scores"] else 0.0,
            "mrr_std": float(np.std(acc["mrr_scores"])) if acc["mrr_scores"] else 0.0,
            "map_min": float(np.min(acc["map_scores"])) if acc["map_scores"] else 0.0,
            "map_max": float(np.max(acc["map_scores"])) if acc["map_scores"] else 0.0,
            "mrr_min": float(np.min(acc["mrr_scores"])) if acc["mrr_scores"] else 0.0,
            "mrr_max": float(np.max(acc["mrr_scores"])) if acc["mrr_scores"] else 0.0,
        }

        # Print
        print(f"\n{'=' * 80}")
        print(f"  {cfg['name']}  ({cfg['description']})")
        print(f"{'=' * 80}")
        print(f"  Queries evaluated: {n_q}")
        print(f"\n  Mean Average Precision (MAP): {mean_map:.6f}")
        print(f"  Mean Reciprocal Rank (MRR):   {mean_mrr:.6f}")

        print(f"\n  Normalized Discounted Cumulative Gain:")
        for k in EVAL_K_VALUES["ndcg"]:
            print(f"    nDCG@{k:3d}: {mean_ndcg[k]:.6f}")

        print(f"\n  Recall Metrics:")
        for k in EVAL_K_VALUES["recall"]:
            print(f"    Recall@{k:3d}: {mean_recall[k]:.6f}")

        print(f"\n  Hit Rate Metrics:")
        for k in EVAL_K_VALUES["hr"]:
            print(f"    HR@{k:3d}: {mean_hr[k]:.6f}")

    print(f"\n{'=' * 80}")

    # ---- Save results to text file ----
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("SCIBERT PURE EVALUATION RESULTS (3 Label Types)\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Model: {cand_data['model_name']}\n")
        f.write(f"Embedding dimension: {cand_data['embedding_dim']}\n")
        f.write(f"Max sequence length: {cand_data['max_length']}\n")
        f.write(f"Text: title + abstract\n")
        f.write(f"Candidates: {len(cand_paper_ids):,} train papers\n\n")

        for key, cfg in LABEL_TYPES.items():
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
            "method": "SciBERT (Pure CLS Embedding)",
            "model_name": cand_data["model_name"],
            "embedding_dim": cand_data["embedding_dim"],
            "max_length": cand_data["max_length"],
            "text_source": "title + abstract",
            "similarity": "cosine (FAISS IndexFlatIP on normalized vectors)",
            "num_candidates": len(cand_paper_ids),
        },
        "label_thresholds": {
            "type_1": "binary, label == 1",
            "type_2": f"usefulness >= {TYPE_2_THRESHOLD}",
            "type_3": f"relatedness >= {TYPE_3_THRESHOLD}",
        },
    }

    for key, cfg in LABEL_TYPES.items():
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

    print(f"\nResults saved to:")
    print(f"  Text: {RESULTS_FILE}")
    print(f"  JSON: {RESULTS_JSON}")


if __name__ == "__main__":
    main()