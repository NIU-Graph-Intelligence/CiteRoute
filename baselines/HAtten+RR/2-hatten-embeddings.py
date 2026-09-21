"""
HAtten — Embedding Generation Script
=======================================
Generates per-paper embeddings using each fine-tuned HAtten model.
Embeds train and eval sets by encoding title + abstract as paragraphs.

Resume: skips models that don't exist, skips already-generated embeddings.

Output (per label type):
  - output/dense/HAtten-RR/embeddings/<type_key>/candidates_embeddings.npy
  - output/dense/HAtten-RR/embeddings/<type_key>/candidates_paper_ids.npy
  - output/dense/HAtten-RR/embeddings/<type_key>/eval_embeddings.npy
  - output/dense/HAtten-RR/embeddings/<type_key>/eval_paper_ids.npy
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import polars as pl
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

from hatten_utils import (
    log, load_paper_text_map,
    HAttenModel, LABEL_TYPES, BATCH_SIZE_EMB, SEP,
    GLOVE_DIM, HIDDEN_DIM, NUM_HEADS,
)

# -------- Config --------
load_dotenv()
ROOT_DIR = Path(os.getenv("ROOT_DIR", "/home/ratul/mustcite/"))
OUTPUT_DIR = ROOT_DIR / "masterset-benchmark/output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_PARQUET = ROOT_DIR / "data/train_eval_set/v7.0/candidate_pool_v7.0.parquet"
EVAL_PARQUET = ROOT_DIR / "data/train_eval_set/v7.0/eval_v7.0.parquet"

MODELS_DIR = OUTPUT_DIR / "dense/HAtten-RR/fine_tuned_models/"
EMBEDDINGS_BASE_DIR = OUTPUT_DIR / "dense/HAtten-RR/embeddings/"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_embeddings_for_dataset(parquet_path, dataset_name, model, paper_text_map, output_dir):
    emb_path = output_dir / f"{dataset_name}_embeddings.npy"
    ids_path = output_dir / f"{dataset_name}_paper_ids.npy"

    if emb_path.exists() and ids_path.exists():
        log(f"    {dataset_name} embeddings already exist — skipping.")
        return True

    df = pl.read_parquet(parquet_path)
    paper_ids = df["paper_id"].to_list()
    log(f"    Processing {len(paper_ids)} papers...")

    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(paper_ids), BATCH_SIZE_EMB), desc=f"    {dataset_name}"):
            batch_ids = paper_ids[i:i + BATCH_SIZE_EMB]
            batch_paragraphs = []
            batch_types = []

            for pid in batch_ids:
                text = paper_text_map.get(pid, "")
                parts = text.split(f" {SEP} ", 1)
                title = parts[0] if len(parts) > 0 else ""
                abstract = parts[1] if len(parts) > 1 else ""
                batch_paragraphs.append([title, abstract])
                batch_types.append([0, 1])

            batch_types_tensor = torch.tensor(batch_types, dtype=torch.long, device=DEVICE)

            try:
                emb = model(batch_paragraphs, batch_types_tensor)
                emb = F.normalize(emb, p=2, dim=-1)
                all_embeddings.append(emb.cpu().numpy())
            except Exception as e:
                log(f"    Error in batch {i}: {e}")
                continue

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    log(f"    {dataset_name} shape: {all_embeddings.shape}")

    np.save(emb_path, all_embeddings)
    np.save(ids_path, np.array(paper_ids, dtype=object))
    return True


def main():
    log("=" * 80)
    log("HAtten — Embedding Generation")
    log("=" * 80)

    generated_count = 0

    for label_key, label_cfg in LABEL_TYPES.items():
        log(f"\n{'=' * 80}")
        log(f"Processing: {label_cfg['name']}")
        log(f"{'=' * 80}")

        model_path = MODELS_DIR / label_key / "final_model.pt"
        if not model_path.exists():
            log(f"  Model not found — skipping.")
            continue

        log(f"  Loading model...")
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

        model = HAttenModel(
            vocab_size=checkpoint["vocab_size"], embedding_dim=checkpoint["embedding_dim"],
            hidden_dim=checkpoint["hidden_dim"], num_heads=checkpoint["num_heads"],
            pretrained_embeddings=None, freeze_embeddings=True,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.set_word2idx(checkpoint["word2idx"])
        model = model.to(DEVICE)
        model.eval()

        # Load text maps
        cand_map = load_paper_text_map(CANDIDATE_PARQUET)
        eval_map = load_paper_text_map(EVAL_PARQUET)
        paper_text_map = {**cand_map, **eval_map}

        emb_dir = EMBEDDINGS_BASE_DIR / label_key
        emb_dir.mkdir(parents=True, exist_ok=True)

        generate_embeddings_for_dataset(CANDIDATE_PARQUET, "candidates", model, paper_text_map, emb_dir)
        generate_embeddings_for_dataset(EVAL_PARQUET, "eval", model, paper_text_map, emb_dir)

        generated_count += 1
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    log(f"\nDone! Processed {generated_count}/{len(LABEL_TYPES)} label types.")


if __name__ == "__main__":
    main()
