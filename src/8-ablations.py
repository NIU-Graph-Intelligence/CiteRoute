"""
Stage 8 — Ablations (Sec. 4.5)
===============================
Produces the ablation tables directly from the Stage-4 runs — no retraining,
because every ablation is a subset of views and/or a switch on the fusion and
reranking stages.

  (a) component ablation : full / -reranker / -learned-fusion / -factorization
                           (the last keeps only the undecomposed views)
  (b) facet-tier alignment: leave-one-facet-out, per tier
  (d) leakage control     : drop the two inferential facets
                           (builds_on, compares_against)

Ablations (c) factorizer swap and (e) query-style robustness need new facet
JSONs; regenerate them with semantic_factorizer into a separate FACETS_DIR and
re-run stages 0/2/4/7 with that directory.

Output: OUTPUT_DIR/evaluation_results/ablations.{json,txt}
"""

import argparse
import json

import torch
from transformers import AutoTokenizer

from citeroute.config import (eval_parquet_for,  # noqa: F401
                             ALL_PAPERS_PARQUET, CANDIDATE_PARQUET,
                             DENSE_FACETS, EVAL_PARQUET, FACETS_PARQUET,
                             FULL_VIEW, FUSION_DIR, LABEL_TYPES,
                             RERANKER_DIR, RERANKER_MODEL_NAME, RESULTS_DIR,
                             SPARSE_FACETS, SPARSE_FULL_RUN, ensure_dirs)
from citeroute.data import (build_ground_truth, load_facet_map,
                           load_paper_text_map, load_paper_title_map)
from citeroute.pipeline import evaluate
from citeroute.rerank import CrossEncoderReranker
from citeroute.runs import load_view_runs
from citeroute.utils import banner, log, save_json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RAW_VIEWS = [FULL_VIEW, SPARSE_FULL_RUN]          # undecomposed dense + sparse
ALL_FACETS = DENSE_FACETS + SPARSE_FACETS
INFERENTIAL = ["builds_on", "compares_against"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", nargs="*", default=list(LABEL_TYPES))
    ap.add_argument("--skip-rerank", action="store_true",
                    help="Run every ablation fusion-only (much faster; the "
                         "facet comparison is unaffected since the reranker "
                         "is constant across facet settings)")
    args = ap.parse_args()

    ensure_dirs()
    banner("Stage 8 — Ablations")

    doc_text = load_paper_text_map(CANDIDATE_PARQUET)
    if ALL_PAPERS_PARQUET.exists():
        doc_text = {**load_paper_text_map(ALL_PAPERS_PARQUET), **doc_text}
    facet_map = load_facet_map(FACETS_PARQUET, text_map=load_paper_text_map(EVAL_PARQUET))
    pool_ids = set(load_paper_title_map(CANDIDATE_PARQUET).keys())
    # Loaded lazily: --skip-rerank must not require transformers at all.
    tokenizer = None if args.skip_rerank else AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)

    results = {}
    for label_key in args.types:
        banner(f"{LABEL_TYPES[label_key]['name']}", char="-")
        per_query, views_found = load_view_runs(label_key, "eval")
        if not per_query:
            log("  No eval runs — skipping.")
            continue
        gt, _ = build_ground_truth(eval_parquet_for(label_key), label_key, pool_ids=pool_ids,
                                   restrict_to=set(per_query))
        if not gt:
            continue

        wp = FUSION_DIR / f"{label_key}_weights.json"
        weights = json.load(open(wp))["weights"] if wp.exists() else None

        reranker = None
        mp = RERANKER_DIR / label_key / "final_model.pt"
        if not args.skip_rerank and mp.exists():
            reranker = CrossEncoderReranker(RERANKER_MODEL_NAME).to(DEVICE)
            state = torch.load(mp, map_location=DEVICE, weights_only=False)
            reranker.load_state_dict(state.get("model_state_dict", state))
            reranker.eval()

        def run(views, use_weights=True, use_rerank=True, desc=""):
            v = [x for x in views if x in views_found]
            if not v:
                return None
            return evaluate(per_query, gt,
                            weights=weights if use_weights else None,
                            views=v,
                            reranker=reranker if use_rerank else None,
                            tokenizer=tokenizer, facet_map=facet_map,
                            doc_text=doc_text, progress_desc=f"  {desc}")

        settings = {}

        # ---- (a) component ablation ----
        settings["full"] = run(views_found, True, True, "full")
        settings["minus_reranker"] = run(views_found, True, False, "-reranker")
        settings["minus_learned_fusion"] = run(views_found, False, True, "-learned fusion")
        settings["minus_factorization"] = run(RAW_VIEWS, True, True, "-factorization")

        # ---- (b) leave-one-facet-out ----
        for facet in ALL_FACETS:
            if facet not in views_found:
                continue
            settings[f"minus_{facet}"] = run(
                [v for v in views_found if v != facet], True, True, f"-{facet}")

        # ---- (d) leakage control ----
        settings["minus_inferential"] = run(
            [v for v in views_found if v not in INFERENTIAL], True, True,
            "-inferential facets")

        results[label_key] = {k: v for k, v in settings.items() if v}

        base = results[label_key].get("full", {}).get("mean_recall", {}).get(100)
        log(f"\n  Recall@100 summary — {label_key}")
        for name, res in results[label_key].items():
            r = res["mean_recall"][100]
            delta = f"  ({r - base:+.4f})" if base is not None and name != "full" else ""
            log(f"    {name:28s} {r:.4f}{delta}")

        if reranker is not None:
            del reranker
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not results:
        log("Nothing to report.")
        return

    save_json(results, RESULTS_DIR / "ablations.json")
    with open(RESULTS_DIR / "ablations.txt", "w") as f:
        f.write("MUSTCITE ABLATIONS — Recall@100 / Recall@50 / MAP\n")
        f.write("=" * 80 + "\n")
        for label_key, settings in results.items():
            f.write(f"\n{LABEL_TYPES[label_key]['name']}\n" + "-" * 80 + "\n")
            base = settings.get("full", {}).get("mean_recall", {}).get(100)
            for name, res in settings.items():
                d = ""
                if base is not None and name != "full":
                    d = f"   delta R@100 {res['mean_recall'][100] - base:+.4f}"
                f.write(f"  {name:28s} R@100 {res['mean_recall'][100]:.4f}  "
                        f"R@50 {res['mean_recall'][50]:.4f}  "
                        f"MAP {res['mean_map']:.4f}{d}\n")
    log(f"\nAblations saved to {RESULTS_DIR}/ablations.{{json,txt}}")


if __name__ == "__main__":
    main()
