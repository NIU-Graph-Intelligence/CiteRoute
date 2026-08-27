"""
Stage 7 — Fuse, rerank, evaluate (main results)
================================================
Runs the full CiteCoute inference path on the eval queries and writes results
in the same .txt/.json format as the baseline suite, so the numbers drop
straight into the paper's Type I/II/III tables.

Input : Stage 4 eval runs, Stage 5 weights (optional), Stage 6 reranker (optional)
Output: OUTPUT_DIR/evaluation_results/mustcite_evaluation_metrics.{txt,json}

Flags:
  --no-reranker   fuse only (ablation (a): reranker removed)
  --uniform-rrf   ignore learned weights (ablation (a): learned fusion removed)
"""

import argparse
import json

import torch
from transformers import AutoTokenizer

from citeroute.config import (eval_parquet_for,  # noqa: F401
                             ALL_PAPERS_PARQUET, CANDIDATE_PARQUET,
                             EVAL_K_VALUES, EVAL_PARQUET, FACETS_PARQUET,
                             FUSION_DIR, LABEL_TYPES, RERANK_DEPTH,
                             RERANKER_DIR, RERANKER_MODEL_NAME, RESULTS_DIR,
                             TYPE_2_THRESHOLD, TYPE_3_THRESHOLD, VIEWS,
                             check_inputs, describe_paths, ensure_dirs)
from citeroute.data import (build_ground_truth, load_facet_map,
                           load_paper_text_map, load_paper_title_map)
from citeroute.metrics import print_result
from citeroute.pipeline import evaluate
from citeroute.rerank import CrossEncoderReranker
from citeroute.runs import load_view_runs
from citeroute.utils import banner, log

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_results(all_results, meta, txt_path, json_path):
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("MUSTCITE EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        for k, v in meta.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")
        for key in LABEL_TYPES:
            cfg = LABEL_TYPES[key]
            if key not in all_results:
                f.write("-" * 80 + "\n")
                f.write(f"  {cfg['name']} — NOT EVALUATED\n\n")
                continue
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
            f.write("\n  Recall Metrics:\n")
            for k in EVAL_K_VALUES["recall"]:
                f.write(f"    Recall@{k:3d}: {res['mean_recall'][k]:.6f}\n")
            f.write("\n  Hit Rate Metrics:\n")
            for k in EVAL_K_VALUES["hr"]:
                f.write(f"    HR@{k:3d}: {res['mean_hr'][k]:.6f}\n")
            f.write("\n  Precision Metrics:\n")
            for k in EVAL_K_VALUES["precision"]:
                f.write(f"    P@{k:3d}: {res['mean_precision'][k]:.6f}\n")
            f.write("\n  Detailed Statistics:\n")
            f.write(f"    MAP - Min: {res['map_min']:.6f}, Max: {res['map_max']:.6f}, "
                    f"Std: {res['map_std']:.6f}\n")
            f.write(f"    MRR - Min: {res['mrr_min']:.6f}, Max: {res['mrr_max']:.6f}, "
                    f"Std: {res['mrr_std']:.6f}\n\n")

    out = {"model_info": meta,
           "label_thresholds": {k: cfg["description"] for k, cfg in LABEL_TYPES.items()}}
    for key, cfg in LABEL_TYPES.items():
        if key not in all_results:
            out[key] = {"status": "not_evaluated"}
            continue
        res = all_results[key]
        out[key] = {
            "name": cfg["name"], "description": cfg["description"],
            "num_queries": res["num_queries"],
            "metrics": {
                "map": res["mean_map"], "mrr": res["mean_mrr"],
                "ndcg": {f"ndcg@{k}": res["mean_ndcg"][k] for k in EVAL_K_VALUES["ndcg"]},
                "recall": {f"recall@{k}": res["mean_recall"][k] for k in EVAL_K_VALUES["recall"]},
                "hit_rate": {f"hr@{k}": res["mean_hr"][k] for k in EVAL_K_VALUES["hr"]},
                "precision": {f"p@{k}": res["mean_precision"][k]
                              for k in EVAL_K_VALUES["precision"]},
            },
            "statistics": {"map_std": res["map_std"], "mrr_std": res["mrr_std"]},
        }
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", nargs="*", default=list(LABEL_TYPES))
    ap.add_argument("--no-reranker", action="store_true")
    ap.add_argument("--uniform-rrf", action="store_true")
    ap.add_argument("--tag", default="citeroute", help="output filename prefix")
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()

    ensure_dirs()
    banner("Stage 7 — CiteCoute evaluation")
    for line in describe_paths():
        log(f"  {line}")
    check_inputs([CANDIDATE_PARQUET, EVAL_PARQUET, FACETS_PARQUET],
                 hint="Run Stages 0-4 first, and check DATA_DIR in .env.")

    doc_text = load_paper_text_map(CANDIDATE_PARQUET)
    if ALL_PAPERS_PARQUET.exists():
        doc_text = {**load_paper_text_map(ALL_PAPERS_PARQUET), **doc_text}
    eval_text = load_paper_text_map(EVAL_PARQUET)
    facet_map = load_facet_map(FACETS_PARQUET, text_map=eval_text)
    title_map = load_paper_title_map(CANDIDATE_PARQUET)
    pool_ids = set(title_map.keys())

    tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME) \
        if not args.no_reranker else None

    all_results = {}
    for label_key in args.types:
        banner(f"{LABEL_TYPES[label_key]['name']}", char="-")
        per_query, views_found = load_view_runs(label_key, "eval")
        if not per_query:
            log("  No eval runs — run Stage 4 first. Skipping.")
            continue

        gt, outside = build_ground_truth(eval_parquet_for(label_key), label_key, pool_ids=pool_ids,
                                         restrict_to=set(per_query))
        if outside:
            log(f"  Gold refs outside candidate pool (excluded): {outside}")
        log(f"  Queries with relevant papers: {len(gt)}")
        if not gt:
            continue

        weights = None
        if not args.uniform_rrf:
            wp = FUSION_DIR / f"{label_key}_weights.json"
            if wp.exists():
                weights = json.load(open(wp))["weights"]
                log("  Using learned fusion weights.")
            else:
                log("  No learned weights found — uniform RRF.")

        reranker = None
        if not args.no_reranker:
            mp = RERANKER_DIR / label_key / "final_model.pt"
            if mp.exists():
                reranker = CrossEncoderReranker(RERANKER_MODEL_NAME).to(DEVICE)
                state = torch.load(mp, map_location=DEVICE, weights_only=False)
                reranker.load_state_dict(state.get("model_state_dict", state))
                reranker.eval()
                log(f"  Reranking top-{RERANK_DEPTH} with {mp}")
            else:
                log("  No reranker found — fusion-only ranking.")

        res = evaluate(per_query, gt, weights=weights, views=views_found,
                       reranker=reranker, tokenizer=tokenizer,
                       facet_map=facet_map, doc_text=doc_text,
                       show_examples=args.examples, title_map=title_map,
                       progress_desc=f"  Evaluating {label_key}")
        if res:
            all_results[label_key] = res
            print_result(LABEL_TYPES[label_key]["name"],
                         LABEL_TYPES[label_key]["description"], res)

        if reranker is not None:
            del reranker
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not all_results:
        log("Nothing evaluated.")
        return

    meta = {
        "method": "CiteCoute (semantic factorization + facet-routed retrieval)",
        "backbone": "contrastive encoder (NT-Xent), per label type",
        "views": ", ".join(VIEWS),
        "fusion": "uniform RRF" if args.uniform_rrf else "RRF with learned per-tier weights",
        "reranker": "disabled" if args.no_reranker else f"cross-encoder, top-{RERANK_DEPTH}",
        "retrieval_pool": "candidate pool (all papers <= 2025)",
    }
    txt = RESULTS_DIR / f"{args.tag}_evaluation_metrics.txt"
    js = RESULTS_DIR / f"{args.tag}_evaluation_metrics.json"
    write_results(all_results, meta, txt, js)
    log(f"\nResults saved to:\n  Text: {txt}\n  JSON: {js}")


if __name__ == "__main__":
    main()
