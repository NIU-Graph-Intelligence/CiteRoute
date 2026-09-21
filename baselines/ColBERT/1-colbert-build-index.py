"""
ColBERT Baseline — Index Building Script
==========================================
Builds a ColBERT index over all training papers using title + abstract.

ColBERTv2 uses late-interaction: each token gets its own embedding, and
document-query scoring is done via MaxSim (max cosine similarity between
each query token and all document tokens). This preserves fine-grained
token-level matching that single-vector models like SciBERT lose.

Pipeline:
  1. Build a TSV collection by chunking train papers into passages
  2. Build a ColBERTv2 index over the collection

Input:
  - data/train_eval_set/v7.0/candidate_pool_v7.0.parquet   (candidate library)

Output:
  - output/dense/ColBERT/collection/train_collection.tsv
  - output/dense/ColBERT/collection/candidates_pid_to_paperid.npy
  - output/dense/ColBERT/collection/train_passages.parquet
  - output/dense/ColBERT/index/<INDEX_NAME>/        (ColBERT index)
"""

import sys
import os
import time
import json
import torch
import numpy as np
import polars as pl
from pathlib import Path
from transformers import AutoTokenizer
from transformers import logging as hf_logging
from dotenv import load_dotenv

hf_logging.set_verbosity_error()

from colbert.infra import Run, ColBERTConfig
from colbert import Indexer

from colbert_utils import (
    log, load_paper_text_map, chunk_by_wordpieces,
    CHECKPOINT, DOC_MAXLEN, NBITS, INDEX_NAME,
)

# -------- Config --------
load_dotenv()
ROOT_DIR = Path(os.getenv("ROOT_DIR", "/home/ratul/mustcite/"))
OUTPUT_DIR = ROOT_DIR / "masterset-benchmark/output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_PARQUET = ROOT_DIR / "data/train_eval_set/v7.0/candidate_pool_v7.0.parquet"

COLBERT_DIR = OUTPUT_DIR / "dense/ColBERT/"
COLBERT_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION_DIR = COLBERT_DIR / "collection/"
COLLECTION_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION_TSV = COLLECTION_DIR / "train_collection.tsv"
PID2PAPER_NPY = COLLECTION_DIR / "candidates_pid_to_paperid.npy"
PASSAGES_PARQUET = COLLECTION_DIR / "candidates_passages.parquet"

INDEX_ROOT = COLBERT_DIR / "index/"
INDEX_ROOT.mkdir(parents=True, exist_ok=True)


# ======================================================================
# Step 1: Build Train Collection (chunked passages)
# ======================================================================

def build_train_collection(train_df: pl.DataFrame, papers_map: dict):
    log("Step 1/2: Building TRAIN collection (title + abstract, chunked)...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    train_ids = train_df["paper_id"].to_list()

    total = len(train_ids)
    pids, paper_ids, texts = [], [], []
    pid = 0

    with open(COLLECTION_TSV, "w", encoding="utf-8") as out:
        for i, paper_id in enumerate(train_ids, 1):
            if i % 5000 == 0 or i == total:
                log(f"  processed {i}/{total} train papers...")

            text = papers_map.get(paper_id, "")
            if not text.strip():
                continue

            wrote_any = False
            for passage in chunk_by_wordpieces(text, tokenizer, DOC_MAXLEN):
                clean = passage.replace("\n", " ").replace("\t", " ")
                out.write(f"{pid}\t{clean}\n")
                pids.append(pid)
                paper_ids.append(paper_id)
                texts.append(passage)
                pid += 1
                wrote_any = True

            if not wrote_any:
                tiny = (text.split(".")[0] or "NA").replace("\n", " ").replace("\t", " ")
                out.write(f"{pid}\t{tiny}\n")
                pids.append(pid)
                paper_ids.append(paper_id)
                texts.append(tiny)
                pid += 1

    # Save pid -> paper_id mapping (as string UUIDs)
    np.save(PID2PAPER_NPY, np.array(paper_ids, dtype=object))

    # Save passages parquet for debugging/inspection
    pl.DataFrame({"pid": pids, "paper_id": paper_ids, "text": texts}).write_parquet(
        PASSAGES_PARQUET
    )

    log(f"  Wrote TRAIN passages → {COLLECTION_TSV}  (#passages={len(pids)})")
    log(f"  Saved pid→paper_id → {PID2PAPER_NPY}")
    log("Step 1/2: Done.")


# ======================================================================
# Step 2: Build ColBERT Index
# ======================================================================

def build_index():
    log("Step 2/2: Building ColBERT index (train-only)...")
    INDEX_ROOT.mkdir(parents=True, exist_ok=True)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    config = ColBERTConfig(
        root=str(COLBERT_DIR),
        index_root=str(INDEX_ROOT),
    )
    config.nbits = NBITS
    config.doc_maxlen = DOC_MAXLEN

    log(f"  torch.cuda.is_available() = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"  CUDA device 0 = {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True

    config.bsize = 128
    config.index_bsize = 128
    config.amp = True

    log("  Effective ColBERTConfig:")
    for k in ["root", "index_root", "nbits", "doc_maxlen", "bsize", "index_bsize", "amp"]:
        log(f"    {k} = {getattr(config, k)}")

    with Run().context(config):
        indexer = Indexer(checkpoint=CHECKPOINT, config=config)
        log(f"  Indexing with checkpoint: {CHECKPOINT}")
        log(f"  Collection: {COLLECTION_TSV}")
        indexer.index(name=INDEX_NAME, collection=str(COLLECTION_TSV))

    log(f"Step 2/2: Index built at {INDEX_ROOT}/{INDEX_NAME}")


# ======================================================================
# Main
# ======================================================================

def main():
    log("=" * 80)
    log("ColBERT Baseline — Index Building")
    log("=" * 80)
    log(f"Python: {sys.version.split()[0]}")

    try:
        import colbert
        log(f"colbert-ai: {getattr(colbert, '__version__', 'unknown')}  torch: {torch.__version__}")
    except Exception:
        pass

    # Load train IDs
    log("\nLoading TRAIN paper IDs...")
    train_df = pl.read_parquet(CANDIDATE_PARQUET, columns=["paper_id"])
    log(f"  TRAIN rows: {train_df.height}")

    # Load paper text map
    log("\nLoading paper text map...")
    papers_map = load_paper_text_map(CANDIDATE_PARQUET)
    log(f"  Total papers in map: {len(papers_map)}")

    # Step 1: Build collection (skip if already exists)
    if not COLLECTION_TSV.exists() or not PID2PAPER_NPY.exists():
        build_train_collection(train_df, papers_map)
    else:
        log("Step 1/2: Collection already exists — reusing.")

    # Step 2: Build index (skip if already exists)
    index_dir = INDEX_ROOT / INDEX_NAME
    if index_dir.exists() and any(index_dir.iterdir()):
        log(f"Step 2/2: Index already exists — {index_dir} — reusing.")
    else:
        build_index()

    log("\n" + "=" * 80)
    log("ColBERT index built successfully!")
    log("=" * 80)


if __name__ == "__main__":
    main()