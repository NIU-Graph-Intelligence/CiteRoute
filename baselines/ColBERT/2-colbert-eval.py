"""
ColBERT Baseline — Evaluation Script
======================================
Evaluates ColBERTv2 retrieval on the eval set using the prebuilt index.

For each eval paper, uses its title + abstract as query text,
retrieves top-K candidates from the ColBERT index, maps passage PIDs
back to paper IDs, and computes metrics for all 3 label types independently.

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
  - output/dense/ColBERT/index/<INDEX_NAME>/            (ColBERT index)
  - output/dense/ColBERT/collection/candidates_pid_to_paperid.npy
  - data/train_eval_set/v7.0/eval_v7.0.parquet

Output:
  - output/dense/ColBERT/evaluation_results/colbert_evaluation_metrics.txt
  - output/dense/ColBERT/evaluation_results/colbert_evaluation_metrics.json
"""

import os
import sys
import time
import json
import numpy as np
import polars as pl
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

from colbert.infra import Run, ColBERTConfig
from colbert import Searcher

from colbert_utils import (
    log,
    load_pid_to_paperid_map,
    retrieve_suggestions,
    extract_relevant_sets,
    compute_all_metrics,
    LABEL_TYPES,
    EVAL_K_VALUES,
    CHECKPOINT,
    INDEX_NAME,
    TYPE_2_THRESHOLD,
    TYPE_3_THRESHOLD,
)

# -------- Config --------
load_dotenv()
ROOT_DIR = Path(os.getenv("ROOT_DIR", "/home/ratul/mustcite/"))
OUTPUT_DIR = ROOT_DIR / "masterset-benchmark/output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_PARQUET = ROOT_DIR / "data/train_eval_set/v7.0/eval_v7.0.parquet"

COLBERT_DIR = OUTPUT_DIR / "dense/ColBERT/"
COLLECTION_DIR = COLBERT_DIR / "collection/"
INDEX_ROOT = COLBERT_DIR / "index/"

PID2PAPER_NPY = COLLECTION_DIR / "candidates_pid_to_paperid.npy"

RESULTS_DIR = COLBERT_DIR / "evaluation_results/"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "colbert_evaluation_metrics.txt"
RESULTS_JSON = RESULTS_DIR / "colbert_evaluation_metrics.json"


# ======================================================================
# Main Evaluation
# ======================================================================

def main():
    log("=" * 80)
    log("ColBERT Baseline — Evaluation (3 Label Types)")
    log("=" * 80)
    log(f"Python: {sys.version.split()[0]}")

    # ---- Load pid -> paper_id mapping ----
    log("\nLoading ColBERT pid → paper_id mapping...")
    pid_to_paperid = load_pid_to_paperid_map(PID2PAPER_NPY)
    log(f"  Total passages: {len(pid_to_paperid)}")

    # Gold refs outside the candidate pool are unretrievable by construction
    # load_pid_to_paperid_map returns a numpy array indexed by passage id
    # (NOT a dict), so build the paper-id set from its values.
    pool_ids = {str(x) for x in np.asarray(pid_to_paperid).ravel().tolist()}
    gold_outside_pool = {key: 0 for key in LABEL_TYPES}

    # ---- Load eval data ----
    log("\nLoading evaluation dataset...")
    eval_df = pl.read_parquet(EVAL_PARQUET)
    num_eval = eval_df.height
    log(f"  Total evaluation papers: {num_eval}")

    # ---- Initialize ColBERT searcher ----
    log("\nInitializing ColBERT searcher...")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    config = ColBERTConfig(
        root=str(COLBERT_DIR),
        index_root=str(INDEX_ROOT),
    )

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

    max_k = max(EVAL_K_VALUES["recall"])

    with Run().context(config):
        searcher = Searcher(index=INDEX_NAME, collection=None, checkpoint=CHECKPOINT)
        log(f"  Searcher initialized. Index: {INDEX_NAME}")

        # ---- Evaluate ----
        log("\nEvaluating...")
        skipped_no_text = 0

        for eval_idx in tqdm(range(num_eval), desc="Processing eval papers"):
            try:
                row = eval_df.row(eval_idx, named=True)
            except IndexError:
                continue

            qid = row["paper_id"]
            qtitle = (row.get("title", "") or "").strip()
            qabs = (row.get("abstract", "") or "").strip()
            references = json.loads(row["references"]) if isinstance(row["references"], str) else row["references"]

            # Extract relevant sets for all 3 label types
            relevant_sets = extract_relevant_sets(references)

            # Keep only gold refs that exist in the candidate pool
            for key in LABEL_TYPES:
                gold_outside_pool[key] += len(relevant_sets[key] - pool_ids)
                relevant_sets[key] &= pool_ids

            # Skip if no label type has any relevant papers
            if all(len(v) == 0 for v in relevant_sets.values()):
                continue

            # Prepare query text: title + abstract
            query_text = f"{qtitle}. {qabs}".strip()
            if not query_text or query_text == ".":
                skipped_no_text += 1
                continue

            # Retrieve using ColBERT (done ONCE, shared across all label types)
            retrieved = retrieve_suggestions(
                query_text, searcher, pid_to_paperid, k=max_k, query_id=qid
            )

            # Compute metrics for each label type independently
            for key in LABEL_TYPES:
                relevant = relevant_sets[key]
                if not relevant:
                    continue

                acc = accumulators[key]
                acc["num_queries"] += 1

                if not retrieved:
                    acc["map_scores"].append(0.0)
                    acc["mrr_scores"].append(0.0)
                    for k in EVAL_K_VALUES["recall"]:
                        acc["recall_scores"][k].append(0.0)
                    for k in EVAL_K_VALUES["ndcg"]:
                        acc["ndcg_scores"][k].append(0.0)
                    for k in EVAL_K_VALUES["hr"]:
                        acc["hr_scores"][k].append(0.0)
                    continue

                metrics = compute_all_metrics(relevant, retrieved)
                acc["map_scores"].append(metrics["map"])
                acc["mrr_scores"].append(metrics["mrr"])
                for k in EVAL_K_VALUES["recall"]:
                    acc["recall_scores"][k].append(metrics["recall"][k])
                for k in EVAL_K_VALUES["ndcg"]:
                    acc["ndcg_scores"][k].append(metrics["ndcg"][k])
                for k in EVAL_K_VALUES["hr"]:
                    acc["hr_scores"][k].append(metrics["hr"][k])

    for key in LABEL_TYPES:
        if gold_outside_pool[key] > 0:
            log(f"  [{key}] gold refs outside candidate pool (excluded): {gold_outside_pool[key]}")
    if skipped_no_text > 0:
        log(f"  Skipped {skipped_no_text} papers with no text")

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
        f.write("COLBERT BASELINE EVALUATION RESULTS (3 Label Types)\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Method: ColBERTv2 (Late Interaction)\n")
        f.write(f"Checkpoint: {CHECKPOINT}\n")
        f.write(f"Text: title + abstract\n")
        f.write(f"Total passages indexed: {len(pid_to_paperid):,}\n\n")

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
            "method": "ColBERTv2 (Late Interaction)",
            "checkpoint": CHECKPOINT,
            "text_source": "title + abstract",
            "total_passages": len(pid_to_paperid),
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