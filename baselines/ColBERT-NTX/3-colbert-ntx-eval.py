"""
ColBERT + NT-Xent — Evaluation Script
=======================================
Evaluates each fine-tuned ColBERT model on its corresponding label type.
Each model's index is searched independently.

Resume: skips label types whose indexes don't exist yet.

Output:
  - output/dense/ColBERT-NTX/evaluation_results/colbert_ntx_evaluation_metrics.txt
  - output/dense/ColBERT-NTX/evaluation_results/colbert_ntx_evaluation_metrics.json
"""

import os
import sys
import json
import numpy as np
import polars as pl
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

from colbert.infra import Run, ColBERTConfig
from colbert import Searcher

from colbert_ntx_utils import (
    log, load_pid_to_paperid_map, retrieve_suggestions,
    extract_relevant_sets, compute_all_metrics,
    LABEL_TYPES, EVAL_K_VALUES, DOC_MAXLEN,
    TYPE_2_THRESHOLD, TYPE_3_THRESHOLD,
)

# -------- Config --------
load_dotenv()
ROOT_DIR = Path(os.getenv("ROOT_DIR", "/home/ratul/mustcite/"))
OUTPUT_DIR = ROOT_DIR / "masterset-benchmark/output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_PARQUET = ROOT_DIR / "data/train_eval_set/v7.0/eval_v7.0.parquet"

COLBERT_NTX_DIR = OUTPUT_DIR / "dense/ColBERT-NTX/"
MODELS_DIR = COLBERT_NTX_DIR / "fine_tuned_models/"
COLLECTION_DIR = COLBERT_NTX_DIR / "collection/"
INDEXES_DIR = COLBERT_NTX_DIR / "indexes/"
PID2PAPER_NPY = COLLECTION_DIR / "candidates_pid_to_paperid.npy"

RESULTS_DIR = COLBERT_NTX_DIR / "evaluation_results/"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "colbert_ntx_evaluation_metrics.txt"
RESULTS_JSON = RESULTS_DIR / "colbert_ntx_evaluation_metrics.json"

SEP = "[SEP]"


def evaluate_label_type(label_key, eval_df, pid_to_paperid):
    # Gold refs outside the candidate pool are unretrievable by construction
    # load_pid_to_paperid_map returns a numpy array indexed by passage id
    # (NOT a dict), so build the paper-id set from its values.
    pool_ids = {str(x) for x in np.asarray(pid_to_paperid).ravel().tolist()}
    gold_outside_pool = 0
    cfg = LABEL_TYPES[label_key]
    model_dir = MODELS_DIR / label_key / "final_model"
    index_name = f"colbert_ntx_{label_key}"
    index_root = INDEXES_DIR / label_key

    index_dir = index_root / index_name
    if not index_dir.exists():
        log(f"  Index not found for {cfg['name']} — skipping.")
        return None

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    config = ColBERTConfig(root=str(COLBERT_NTX_DIR), index_root=str(index_root))

    acc = {
        "map_scores": [], "mrr_scores": [],
        "recall_scores": {k: [] for k in EVAL_K_VALUES["recall"]},
        "ndcg_scores": {k: [] for k in EVAL_K_VALUES["ndcg"]},
        "hr_scores": {k: [] for k in EVAL_K_VALUES["hr"]},
        "num_queries": 0,
    }
    max_k = max(EVAL_K_VALUES["recall"])

    with Run().context(config):
        searcher = Searcher(index=index_name, collection=None, checkpoint=str(model_dir))
        log(f"  Searcher initialized for {cfg['name']}")

        for eval_idx in tqdm(range(eval_df.height), desc=f"  Eval {label_key}"):
            row = eval_df.row(eval_idx, named=True)
            qid = row["paper_id"]
            qtitle = (row.get("title", "") or "").strip()
            qabs = (row.get("abstract", "") or "").strip()
            references = json.loads(row["references"]) if isinstance(row["references"], str) else row["references"]

            relevant_sets = extract_relevant_sets(references)
            gold_outside_pool += len(relevant_sets[label_key] - pool_ids)
            relevant = relevant_sets[label_key] & pool_ids
            if not relevant:
                continue

            query_text = f"{qtitle} {SEP} {qabs}".strip()
            if not query_text or query_text == SEP:
                continue

            acc["num_queries"] += 1
            retrieved = retrieve_suggestions(query_text, searcher, pid_to_paperid, k=max_k, query_id=qid)

            if not retrieved:
                acc["map_scores"].append(0.0)
                acc["mrr_scores"].append(0.0)
                for k in EVAL_K_VALUES["recall"]: acc["recall_scores"][k].append(0.0)
                for k in EVAL_K_VALUES["ndcg"]: acc["ndcg_scores"][k].append(0.0)
                for k in EVAL_K_VALUES["hr"]: acc["hr_scores"][k].append(0.0)
                continue

            metrics = compute_all_metrics(relevant, retrieved)
            acc["map_scores"].append(metrics["map"])
            acc["mrr_scores"].append(metrics["mrr"])
            for k in EVAL_K_VALUES["recall"]: acc["recall_scores"][k].append(metrics["recall"][k])
            for k in EVAL_K_VALUES["ndcg"]: acc["ndcg_scores"][k].append(metrics["ndcg"][k])
            for k in EVAL_K_VALUES["hr"]: acc["hr_scores"][k].append(metrics["hr"][k])

    if gold_outside_pool > 0:
        log(f"  [{label_key}] gold refs outside candidate pool (excluded): {gold_outside_pool}")

    n_q = acc["num_queries"]
    if n_q == 0:
        return None

    return {
        "num_queries": n_q,
        "mean_map": float(np.mean(acc["map_scores"])),
        "mean_mrr": float(np.mean(acc["mrr_scores"])),
        "mean_recall": {k: float(np.mean(v)) for k, v in acc["recall_scores"].items()},
        "mean_ndcg": {k: float(np.mean(v)) for k, v in acc["ndcg_scores"].items()},
        "mean_hr": {k: float(np.mean(v)) for k, v in acc["hr_scores"].items()},
        "map_std": float(np.std(acc["map_scores"])),
        "mrr_std": float(np.std(acc["mrr_scores"])),
    }


def main():
    log("=" * 80)
    log("ColBERT + NT-Xent — Evaluation (3 Label Types)")
    log("=" * 80)

    eval_df = pl.read_parquet(EVAL_PARQUET)
    pid_to_paperid = load_pid_to_paperid_map(PID2PAPER_NPY)
    log(f"  Eval papers: {eval_df.height}, Passages: {len(pid_to_paperid)}")

    all_results = {}
    for label_key, label_cfg in LABEL_TYPES.items():
        log(f"\n{'=' * 80}")
        log(f"Evaluating: {label_cfg['name']}")
        log(f"{'=' * 80}")
        result = evaluate_label_type(label_key, eval_df, pid_to_paperid)
        if result:
            all_results[label_key] = result
            log(f"  MAP: {result['mean_map']:.6f}  MRR: {result['mean_mrr']:.6f}")

    if not all_results:
        log("No label types evaluated.")
        return

    # Save txt
    with open(RESULTS_FILE, "w") as f:
        f.write("=" * 80 + "\nCOLBERT + NT-XENT EVALUATION RESULTS\n" + "=" * 80 + "\n\n")
        for key, cfg in LABEL_TYPES.items():
            if key not in all_results:
                continue
            res = all_results[key]
            f.write(f"--- {cfg['name']} ({cfg['description']}) ---\n")
            f.write(f"Queries: {res['num_queries']}\n")
            f.write(f"MAP: {res['mean_map']:.6f}  MRR: {res['mean_mrr']:.6f}\n")
            for k in EVAL_K_VALUES["ndcg"]: f.write(f"  nDCG@{k}: {res['mean_ndcg'][k]:.6f}\n")
            for k in EVAL_K_VALUES["recall"]: f.write(f"  Recall@{k}: {res['mean_recall'][k]:.6f}\n")
            for k in EVAL_K_VALUES["hr"]: f.write(f"  HR@{k}: {res['mean_hr'][k]:.6f}\n")
            f.write("\n")

    # Save json
    results_dict = {
        "model_info": {"method": "ColBERTv2 + NT-Xent Fine-Tuning", "text_source": "title + abstract"},
        "label_thresholds": {"type_1": "label == 1", "type_2": f">= {TYPE_2_THRESHOLD}", "type_3": f">= {TYPE_3_THRESHOLD}"},
    }
    for key in LABEL_TYPES:
        if key in all_results:
            res = all_results[key]
            results_dict[key] = {
                "name": LABEL_TYPES[key]["name"], "num_queries": res["num_queries"],
                "metrics": {
                    "map": res["mean_map"], "mrr": res["mean_mrr"],
                    "ndcg": {f"ndcg@{k}": res["mean_ndcg"][k] for k in EVAL_K_VALUES["ndcg"]},
                    "recall": {f"recall@{k}": res["mean_recall"][k] for k in EVAL_K_VALUES["recall"]},
                    "hit_rate": {f"hr@{k}": res["mean_hr"][k] for k in EVAL_K_VALUES["hr"]},
                },
                "statistics": {"map_std": res["map_std"], "mrr_std": res["mrr_std"]},
            }

    with open(RESULTS_JSON, "w") as f:
        json.dump(results_dict, f, indent=2)

    log(f"\nResults saved to {RESULTS_FILE} and {RESULTS_JSON}")


if __name__ == "__main__":
    main()