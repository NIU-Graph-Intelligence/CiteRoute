"""
Stage 5 — Fit per-tier fusion weights (Eq. 5)
==============================================
Learns lambda_i, one weight per retrieval view, by logistic regression on
reciprocal-rank features over a sample of TRAIN queries. One weight vector
per ground-truth tier — this is the interpretable readout of which facet
serves which tier (Sec. 3.6).

Input : OUTPUT_DIR/runs/<type_key>/fusion_<view>.npz  (Stage 4, split=fusion)
Output: OUTPUT_DIR/fusion/<type_key>_weights.json

Weights are never required: Stage 7 falls back to uniform (classical) RRF if
this stage has not been run.
"""

import argparse

from citeroute.config import (CANDIDATE_PARQUET, FUSION_DIR, LABEL_TYPES,
                             TRAIN_PARQUET, VIEWS, ensure_dirs)
from citeroute.data import build_ground_truth, paper_ids
from citeroute.fusion import fit_weights
from citeroute.runs import load_view_runs
from citeroute.utils import banner, log, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", nargs="*", default=list(LABEL_TYPES))
    args = ap.parse_args()

    ensure_dirs()
    banner("Stage 5 — Fitting per-tier fusion weights")

    pool_ids = set(paper_ids(CANDIDATE_PARQUET))

    for label_key in args.types:
        banner(f"{LABEL_TYPES[label_key]['name']}", char="-")
        per_query, views_found = load_view_runs(label_key, "fusion")
        if not per_query:
            log("  No fusion runs found — run Stage 4 with --splits fusion. Skipping.")
            continue

        gt, _ = build_ground_truth(TRAIN_PARQUET, label_key, pool_ids=pool_ids,
                                   restrict_to=set(per_query))
        samples = [(per_query[q], gt[q]) for q in per_query if q in gt]
        log(f"  Training samples: {len(samples)} queries with gold labels")
        if not samples:
            log("  No labelled fusion queries — skipping.")
            continue

        weights = fit_weights(samples, views=VIEWS)
        out = FUSION_DIR / f"{label_key}_weights.json"
        save_json({"label_type": label_key,
                   "views": VIEWS,
                   "views_present": views_found,
                   "num_samples": len(samples),
                   "weights": weights}, out)
        log(f"  Saved -> {out}")

    banner(f"Fusion weights in {FUSION_DIR}")


if __name__ == "__main__":
    main()
