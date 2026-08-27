"""
Stage 3 — Build the BM25 sparse index over the candidate pool (Eq. 3)
=====================================================================
One index, shared by all label types and all sparse facet views. Documents
are indexed as title + abstract (matching the BM25 baseline exactly, so the
'full' sparse view reproduces that baseline).

Output:
  - OUTPUT_DIR/sparse/bm25_index.pkl
  - OUTPUT_DIR/sparse/index_stats.json

Resume: skips if the index exists (use --force to rebuild).
"""

import sys

from citeroute.config import (CANDIDATE_PARQUET, SPARSE_DIR, check_inputs,
                             describe_paths, ensure_dirs)
from citeroute.data import load_paper_text_map, paper_ids
from citeroute.sparse import BM25Index
from citeroute.utils import Timer, banner, log, save_json

INDEX_PATH = SPARSE_DIR / "bm25_index.pkl"


def main():
    force = "--force" in sys.argv
    ensure_dirs()
    banner("Stage 3 — BM25 sparse index over the candidate pool")
    for line in describe_paths():
        log(f"  {line}")
    check_inputs([CANDIDATE_PARQUET], hint="Set DATA_DIR in .env for this server.")

    if INDEX_PATH.exists() and not force:
        log(f"  {INDEX_PATH} exists — skipping. Use --force to rebuild.")
        return

    ids = paper_ids(CANDIDATE_PARQUET)
    text_map = load_paper_text_map(CANDIDATE_PARQUET)
    texts = [text_map.get(pid, "") for pid in ids]
    log(f"  Documents: {len(ids)}")

    with Timer("Index build"):
        index = BM25Index.build(ids, texts)
    with Timer("Index save"):
        index.save(INDEX_PATH)

    save_json({"num_docs": len(ids),
               "vocab_size": len(index.vocab),
               "postings": int(index.weights.nnz)},
              SPARSE_DIR / "index_stats.json")
    log(f"  Saved -> {INDEX_PATH}")


if __name__ == "__main__":
    main()
