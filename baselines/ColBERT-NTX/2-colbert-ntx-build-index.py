"""
ColBERT + NT-Xent — Index Building Script
===========================================
For each fine-tuned ColBERT model (one per label type), builds a ColBERT
index over the train set. Reuses the same collection TSV across all models.

Resume: skips label types without fine-tuned models, skips already-built indexes.

Output (per label type):
  - output/dense/ColBERT-NTX/collection/train_collection.tsv  (shared)
  - output/dense/ColBERT-NTX/indexes/<type_key>/<index_name>/
"""

import os
import sys
import time
import json
import torch
import numpy as np
import polars as pl
from pathlib import Path
from transformers import AutoTokenizer, logging as hf_logging
from dotenv import load_dotenv

hf_logging.set_verbosity_error()

from colbert.infra import Run, ColBERTConfig
from colbert import Indexer

from colbert_ntx_utils import (
    log, load_paper_text_map, chunk_by_wordpieces,
    LABEL_TYPES, DOC_MAXLEN, NBITS, CHECKPOINT
)

# -------- Config --------
load_dotenv()
ROOT_DIR = Path(os.getenv("ROOT_DIR", "/home/ratul/mustcite/"))
OUTPUT_DIR = ROOT_DIR / "masterset-benchmark/output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_PARQUET = ROOT_DIR / "data/train_eval_set/v7.0/candidate_pool_v7.0.parquet"

COLBERT_NTX_DIR = OUTPUT_DIR / "dense/ColBERT-NTX/"
MODELS_DIR = COLBERT_NTX_DIR / "fine_tuned_models/"

COLLECTION_DIR = COLBERT_NTX_DIR / "collection/"
COLLECTION_DIR.mkdir(parents=True, exist_ok=True)
COLLECTION_TSV = COLLECTION_DIR / "train_collection.tsv"
PID2PAPER_NPY = COLLECTION_DIR / "candidates_pid_to_paperid.npy"
PASSAGES_PARQUET = COLLECTION_DIR / "candidates_passages.parquet"

INDEXES_DIR = COLBERT_NTX_DIR / "indexes/"
INDEXES_DIR.mkdir(parents=True, exist_ok=True)

TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)


def build_collection(train_df, papers_map):
    log("Building TRAIN collection (shared across all label types)...")
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
            for passage in chunk_by_wordpieces(text, TOKENIZER, DOC_MAXLEN):
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

    np.save(PID2PAPER_NPY, np.array(paper_ids, dtype=object))
    pl.DataFrame({"pid": pids, "paper_id": paper_ids, "text": texts}).write_parquet(PASSAGES_PARQUET)
    log(f"  Wrote {len(pids)} passages")


def build_index_for_label(label_key):
    cfg = LABEL_TYPES[label_key]
    model_dir = MODELS_DIR / label_key / "final_model"

    if not model_dir.exists():
        log(f"  Fine-tuned model not found for {cfg['name']} — skipping.")
        return

    # Ensure ColBERT config JSON exists (may be missing if model.save() failed
    # during training and fell back to state_dict-only save)
    config_path = model_dir / "artifact.metadata"
    if not config_path.exists() or config_path.stat().st_size == 0:
        import ujson
        colbert_meta = {
            "query_token_id": "[unused0]", "doc_token_id": "[unused1]",
            "query_token": "[Q]", "doc_token": "[D]",
            "similarity": "cosine", "dim": 128, "doc_maxlen": DOC_MAXLEN,
            "query_maxlen": DOC_MAXLEN, "mask_punctuation": True,
            "checkpoint": CHECKPOINT,
        }
        with open(config_path, "w") as f:
            ujson.dump(colbert_meta, f)
        log(f"  Created missing ColBERT config at {config_path}")

    index_name = f"colbert_ntx_{label_key}"
    index_root = INDEXES_DIR / label_key
    index_root.mkdir(parents=True, exist_ok=True)

    index_dir = index_root / index_name
    if index_dir.exists() and any(index_dir.iterdir()):
        log(f"  Index already exists for {cfg['name']} — skipping.")
        return

    log(f"  Building index with checkpoint: {model_dir}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    config = ColBERTConfig(root=str(COLBERT_NTX_DIR), index_root=str(index_root))
    config.nbits = NBITS
    config.doc_maxlen = DOC_MAXLEN
    config.bsize = 128
    config.index_bsize = 128
    config.amp = True

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    with Run().context(config):
        indexer = Indexer(checkpoint=str(model_dir), config=config)
        indexer.index(name=index_name, collection=str(COLLECTION_TSV))

    log(f"  Index built at {index_root / index_name}")


def main():
    log("=" * 80)
    log("ColBERT + NT-Xent — Index Building (3 Label Types)")
    log("=" * 80)

    train_df = pl.read_parquet(CANDIDATE_PARQUET, columns=["paper_id"])
    papers_map = load_paper_text_map(CANDIDATE_PARQUET)

    # Build shared collection
    if not COLLECTION_TSV.exists() or not PID2PAPER_NPY.exists():
        build_collection(train_df, papers_map)
    else:
        log("Collection already exists — reusing.")

    # Build index per label type
    for label_key in LABEL_TYPES:
        log(f"\n{'=' * 80}")
        log(f"Processing: {LABEL_TYPES[label_key]['name']}")
        log(f"{'=' * 80}")
        build_index_for_label(label_key)

    log("\nAll indexing complete!")


if __name__ == "__main__":
    main()