"""
Stage 0 — Consolidate semantic factorization JSONs
===================================================
Walks the semantic_factorizer output tree (one JSON per paper) and writes a
single parquet keyed by paper_id, so later stages never touch 168K files.

Input:
  - $FACETS_DIR (default: DATA_DIR/semantic_factors)/<venue>/<year>/<paper_id>.json

Output:
  - OUTPUT_DIR/facets/facets.parquet
  - OUTPUT_DIR/facets/coverage.json   (how many pool/eval papers have facets)

Resume: skips entirely if facets.parquet already exists (use --force to rebuild).
"""

import sys

import polars as pl

from citeroute.config import (ALL_PAPERS_PARQUET, CANDIDATE_PARQUET,
                             DENSE_FACETS, EVAL_PARQUET, FACETS_JSON_DIR,
                             FACETS_PARQUET, SPARSE_FACETS, TRAIN_PARQUET,
                             check_inputs, describe_paths, ensure_dirs)
from citeroute.data import (consolidate_facet_jsons, paper_index_from,
                           paper_ids)
from citeroute.utils import banner, log, save_json


def main():
    force = "--force" in sys.argv
    ensure_dirs()
    banner("Stage 0 — Consolidate semantic factorization JSONs")
    for line in describe_paths():
        log(f"  {line}")
    check_inputs([FACETS_JSON_DIR, CANDIDATE_PARQUET, EVAL_PARQUET],
                 hint="Set DATA_DIR / FACETS_DIR in .env for this server.")

    if FACETS_PARQUET.exists() and not force:
        df = pl.read_parquet(FACETS_PARQUET)
        log(f"  {FACETS_PARQUET} already exists ({df.height} rows) — skipping. "
            f"Use --force to rebuild.")
        return

    log(f"Source: {FACETS_JSON_DIR}/<venue>/<year>/<paper_id>.json")
    # Address the JSONs directly from the dataframes' venue/year/paper_id.
    index = paper_index_from([CANDIDATE_PARQUET, EVAL_PARQUET,
                              TRAIN_PARQUET, ALL_PAPERS_PARQUET])
    log(f"  Papers to resolve: {len(index)}")
    df = consolidate_facet_jsons(FACETS_JSON_DIR, FACETS_PARQUET, index)

    # ---- Coverage report -------------------------------------------------
    # Two distinct kinds of gap are reported, because they are handled
    # differently downstream:
    #   (1) paper has NO factorization JSON at all (extraction failed/missing)
    #   (2) paper has a JSON, but a given facet is null/empty for it
    have = set(df["paper_id"].to_list())

    # Which facets does the schema actually on disk define? v1 has no
    # builds_on; anything empty for EVERY paper is treated as schema-absent.
    schema_label = (df["schema"].mode().to_list() or ["unknown"])[0]
    absent_facets = set()
    for col in DENSE_FACETS + SPARSE_FACETS:
        if col in df.columns:
            filled = df.filter((pl.col(col).is_not_null())
                               & (pl.col(col).str.strip_chars() != "")).height
            if filled == 0:
                absent_facets.add(col)
    if absent_facets:
        log(f"  Facets not present in the {schema_label} schema: "
            f"{sorted(absent_facets)} — their views are omitted from fusion.")

    coverage = {}
    for name, pq in [("candidate_pool", CANDIDATE_PARQUET), ("eval", EVAL_PARQUET)]:
        try:
            ids = set(paper_ids(pq))
        except Exception as e:  # noqa: BLE001
            log(f"  Could not read {pq}: {e}")
            continue
        n_have = len(ids & have)
        entry = {"total": len(ids), "with_facets": n_have,
                 "missing_json": len(ids) - n_have,
                 "pct": round(100.0 * n_have / max(len(ids), 1), 2)}
        log(f"  {name}: {n_have}/{len(ids)} papers have a factorization "
            f"({entry['pct']}%)")
        if n_have < len(ids):
            log(f"    {len(ids) - n_have} papers have NO factorization JSON — they "
                f"still participate via the undecomposed 'full' view.")

        # Per-facet fill rate among the papers that DO have a JSON.
        # Facets that the detected schema does not define at all (e.g.
        # builds_on under v1) are reported as n/a rather than 0% — they are
        # absent by design, not sparse, and flagging them as "low" would bury
        # the facets that really are underpopulated.
        sub = df.filter(pl.col("paper_id").is_in(list(ids)))
        per_facet = {}
        if sub.height:
            for col in DENSE_FACETS + SPARSE_FACETS:
                if col not in sub.columns:
                    continue
                n_filled = sub.filter(
                    (pl.col(col).is_not_null()) & (pl.col(col).str.strip_chars() != "")
                ).height
                per_facet[col] = {"filled": n_filled, "of": sub.height,
                                  "pct": round(100.0 * n_filled / sub.height, 2),
                                  "in_schema": col not in absent_facets}
            log(f"    per-facet fill rate:")
            for col, st in per_facet.items():
                if not st["in_schema"]:
                    log(f"      {col:22s} {'n/a':>15s}  (not defined in the "
                        f"{schema_label} schema — view omitted)")
                    continue
                flag = "  <-- low" if st["pct"] < 80 else ""
                log(f"      {col:22s} {st['filled']:7d}/{st['of']:<7d} "
                    f"{st['pct']:6.2f}%{flag}")
        entry["per_facet"] = per_facet
        coverage[name] = entry

    save_json(coverage, FACETS_PARQUET.parent / "coverage.json")
    log(f"  Coverage report -> {FACETS_PARQUET.parent / 'coverage.json'}")
    log("Done.")


if __name__ == "__main__":
    main()
