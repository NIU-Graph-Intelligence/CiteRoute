"""
SPECTER (Pretrained) — Embedding Generation (Inference Only)
==============================================================
Uses the official pretrained SPECTER model from Allen AI:
  allenai/specter

This is the released checkpoint — no finetuning on our data.
Input: title + [SEP] + abstract → [CLS] token embedding (768D).

Generates separate candidate-pool and eval embedding files.
"""

import os
import sys
import json
import torch
import numpy as np
import polars as pl
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv

# -------- Config --------
load_dotenv()
ROOT_DIR = Path(os.getenv("ROOT_DIR", "/home/ratul/mustcite/"))
OUTPUT_DIR = ROOT_DIR / "masterset-benchmark/output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_PARQUET = ROOT_DIR / "data/train_eval_set/v7.0/candidate_pool_v7.0.parquet"
EVAL_PARQUET = ROOT_DIR / "data/train_eval_set/v7.0/eval_v7.0.parquet"

MODEL_NAME = "allenai/specter"
EMBED_DIR = OUTPUT_DIR / "dense/SPECTER-pretrained/embeddings/"
EMBED_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 128
MAX_LEN = 512
SEP = "[SEP]"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import time
def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_paper_text_map(parquet_file):
    df = pl.read_parquet(parquet_file)
    paper_map = {}
    for row in df.iter_rows(named=True):
        pid = row["paper_id"]
        title = (row.get("title", "") or "").strip()
        abstract = (row.get("abstract", "") or "").strip()
        paper_map[pid] = f"{title} {SEP} {abstract}"
    return paper_map


def generate_embeddings(parquet_path, dataset_name, model, tokenizer, output_dir):
    emb_path = output_dir / f"{dataset_name}_embeddings.pt"
    map_path = output_dir / f"{dataset_name}_paper_id_to_index.json"

    if emb_path.exists() and map_path.exists():
        log(f"  {dataset_name} embeddings already exist — skipping.")
        return

    df = pl.read_parquet(parquet_path)
    paper_ids = df["paper_id"].to_list()
    paper_text_map = load_paper_text_map(parquet_path)

    all_embeddings = []
    all_paper_ids = []

    model.eval()
    for i in tqdm(range(0, len(paper_ids), BATCH_SIZE), desc=f"  Embedding {dataset_name}"):
        batch_ids = paper_ids[i:i + BATCH_SIZE]
        batch_texts = [paper_text_map.get(pid, "") for pid in batch_ids]

        inputs = tokenizer(batch_texts, padding=True, truncation=True,
                           max_length=MAX_LEN, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        all_embeddings.append(cls_emb.cpu())
        all_paper_ids.extend(batch_ids)

    final_emb = torch.cat(all_embeddings, dim=0)
    log(f"  {dataset_name} shape: {final_emb.shape}")

    torch.save({
        "embeddings": final_emb, "paper_ids": all_paper_ids,
        "model_name": MODEL_NAME, "embedding_dim": final_emb.shape[1],
    }, emb_path)

    with open(map_path, "w") as f:
        json.dump({pid: idx for idx, pid in enumerate(all_paper_ids)}, f)


def main():
    log("=" * 80)
    log("SPECTER (Pretrained) — Embedding Generation")
    log(f"Model: {MODEL_NAME}")
    log("=" * 80)

    log(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    log(f"  Hidden size: {model.config.hidden_size}")

    generate_embeddings(CANDIDATE_PARQUET, "candidates", model, tokenizer, EMBED_DIR)
    generate_embeddings(EVAL_PARQUET, "eval", model, tokenizer, EMBED_DIR)

    log("\nDone!")


if __name__ == "__main__":
    main()
