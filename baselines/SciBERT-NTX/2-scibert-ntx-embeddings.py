"""
SciBERT + NT-Xent — Embedding Generation Script
=================================================
Generates embeddings using each fine-tuned model (one per label type).
Embeds every paper in the candidate pool and eval set individually through the
fine-tuned ContrastiveSciBERT, producing per-paper CLS+projection vectors.

Resume capability:
  - Skips label types whose fine-tuned model doesn't exist yet.
  - Skips candidate/eval embeddings that are already generated.

Input:
  - output/dense/SciBERT-NTXent/fine_tuned_models/<type_key>/final_model.pt
  - data/train_eval_set/v7.0/candidate_pool_v7.0.parquet
  - data/train_eval_set/v7.0/eval_v7.0.parquet

Output (per label type):
  - output/dense/SciBERT-NTXent/embeddings/<type_key>/candidates_embeddings.pt
  - output/dense/SciBERT-NTXent/embeddings/<type_key>/eval_embeddings.pt
  - output/dense/SciBERT-NTXent/embeddings/<type_key>/candidates_paper_id_to_index.json
  - output/dense/SciBERT-NTXent/embeddings/<type_key>/eval_paper_id_to_index.json
"""

import os
import sys
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

from scibert_ntx_utils import (
    log, load_paper_text_map,
    ContrastiveSciBERT, PaperTextDataset,
    LABEL_TYPES, MODEL_NAME, EMBED_DIM, MAX_LEN, BATCH_SIZE_EMB,
)

# -------- Config --------
load_dotenv()
ROOT_DIR = Path(os.getenv("ROOT_DIR", "/home/ratul/mustcite/"))
OUTPUT_DIR = ROOT_DIR / "masterset-benchmark/output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_PARQUET = ROOT_DIR / "data/train_eval_set/v7.0/candidate_pool_v7.0.parquet"
EVAL_PARQUET = ROOT_DIR / "data/train_eval_set/v7.0/eval_v7.0.parquet"

MODELS_DIR = OUTPUT_DIR / "dense/SciBERT-NTXent/fine_tuned_models/"
EMBEDDINGS_BASE_DIR = OUTPUT_DIR / "dense/SciBERT-NTXent/embeddings/"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================
# Generate embeddings for a single dataset using a loaded model
# ======================================================================

def generate_embeddings_for_dataset(parquet_path, dataset_name, model, tokenizer, output_dir):
    """
    Generate per-paper embeddings and save to disk.
    Returns True if embeddings were generated or already exist, False on failure.
    """
    embeddings_path = output_dir / f"{dataset_name}_embeddings.pt"
    mapping_path = output_dir / f"{dataset_name}_paper_id_to_index.json"

    # ---- Check if already exists ----
    if embeddings_path.exists() and mapping_path.exists():
        log(f"    {dataset_name} embeddings already exist — skipping.")
        return True

    import polars as pl
    df = pl.read_parquet(parquet_path)
    paper_ids = df["paper_id"].to_list()

    paper_text_map = load_paper_text_map(parquet_path)
    dataset = PaperTextDataset(paper_ids, paper_text_map, tokenizer, max_length=MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE_EMB, shuffle=False, num_workers=4, pin_memory=True)

    all_embeddings = []
    all_paper_ids = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"    Embedding {dataset_name}"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            batch_paper_ids = batch["paper_id"]

            emb = model(input_ids, attention_mask)
            all_embeddings.append(emb.cpu())
            all_paper_ids.extend(list(batch_paper_ids))

    final_embeddings = torch.cat(all_embeddings, dim=0)
    log(f"    {dataset_name} embeddings shape: {final_embeddings.shape}")

    # Save
    paper_id_to_idx = {pid: idx for idx, pid in enumerate(all_paper_ids)}

    output_data = {
        "embeddings": final_embeddings,
        "paper_ids": all_paper_ids,
        "model_name": MODEL_NAME,
        "embedding_dim": final_embeddings.shape[1],
        "max_length": MAX_LEN,
    }

    torch.save(output_data, embeddings_path)
    with open(mapping_path, "w") as f:
        json.dump(paper_id_to_idx, f)

    log(f"    Saved to {embeddings_path}")
    return True


# ======================================================================
# Main
# ======================================================================

def main():
    log("=" * 80)
    log("SciBERT + NT-Xent — Embedding Generation")
    log("=" * 80)
    log(f"Python: {sys.version.split()[0]}")
    log(f"Device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    generated_count = 0

    for label_key, label_cfg in LABEL_TYPES.items():
        log(f"\n{'=' * 80}")
        log(f"Processing: {label_cfg['name']}")
        log(f"{'=' * 80}")

        # ---- Check if fine-tuned model exists ----
        model_path = MODELS_DIR / label_key / "final_model.pt"
        if not model_path.exists():
            log(f"  Fine-tuned model not found at {model_path} — skipping.")
            log(f"  (Run the fine-tuning script first for this label type.)")
            continue

        # ---- Load fine-tuned model ----
        log(f"  Loading fine-tuned model from {model_path}...")
        model = ContrastiveSciBERT(MODEL_NAME, embed_dim=EMBED_DIM).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))
        model.eval()

        # ---- Output directory for this label type ----
        emb_dir = EMBEDDINGS_BASE_DIR / label_key
        emb_dir.mkdir(parents=True, exist_ok=True)

        # ---- Generate candidate pool embeddings ----
        log(f"  Generating candidate pool embeddings...")
        generate_embeddings_for_dataset(CANDIDATE_PARQUET, "candidates", model, tokenizer, emb_dir)

        # ---- Generate eval embeddings ----
        log(f"  Generating eval embeddings...")
        generate_embeddings_for_dataset(EVAL_PARQUET, "eval", model, tokenizer, emb_dir)

        generated_count += 1

        # Free GPU memory before loading next model
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    log(f"\n{'=' * 80}")
    log(f"Embedding generation complete! Processed {generated_count}/{len(LABEL_TYPES)} label types.")
    log(f"{'=' * 80}")


if __name__ == "__main__":
    main()