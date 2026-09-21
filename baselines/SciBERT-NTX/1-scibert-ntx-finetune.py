"""
SciBERT + NT-Xent Fine-Tuning Script
======================================
Fine-tunes SciBERT with NT-Xent contrastive loss, separately for each
of the 3 label types. Each label type produces its own fine-tuned model.

Resume capability:
  - If a final model for a label type exists, skip it entirely.
  - If training was interrupted mid-epoch, resume from the last checkpoint.
  - Checkpoints are saved after every epoch.

Input:
  - data/train_eval_set/v7.0/train_v7.0.parquet
  - data/train_eval_set/v7.0/all_papers_with_refs_and_labels.parquet  (text lookup for cited papers)

Output (per label type):
  - output/dense/SciBERT-NTXent/fine_tuned_models/<type_key>/final_model.pt
  - output/dense/SciBERT-NTXent/fine_tuned_models/<type_key>/epoch_<N>.pt
  - output/dense/SciBERT-NTXent/fine_tuned_models/<type_key>/training_log.json
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
    ContrastiveSciBERT, NTXentLoss, PaperPositivePairsDataset,
    LABEL_TYPES, MODEL_NAME, EMBED_DIM, MAX_LEN,
    BATCH_SIZE_FT, EPOCHS, LR,
)

# -------- Config --------
load_dotenv()
ROOT_DIR = Path(os.getenv("ROOT_DIR", "/home/ratul/mustcite/"))
OUTPUT_DIR = ROOT_DIR / "masterset-benchmark/output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PARQUET = ROOT_DIR / "data/train_eval_set/v7.0/train_v7.0.parquet"
ALL_PAPERS_PARQUET = ROOT_DIR / "data/train_eval_set/v7.0/all_papers_with_refs_and_labels.parquet"

MODELS_DIR = OUTPUT_DIR / "dense/SciBERT-NTXent/fine_tuned_models/"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================
# Fine-tune for a single label type
# ======================================================================

def fine_tune_for_label(label_type_key, paper_text_map, tokenizer):
    """Fine-tune a ContrastiveSciBERT model for one label type with resume support."""
    cfg = LABEL_TYPES[label_type_key]
    model_dir = MODELS_DIR / label_type_key
    model_dir.mkdir(parents=True, exist_ok=True)

    final_model_path = model_dir / "final_model.pt"
    log_file = model_dir / "training_log.json"

    # ---- Check if already completed ----
    if final_model_path.exists():
        log(f"  Final model already exists at {final_model_path} — skipping.")
        return

    # ---- Find last completed epoch (for resume) ----
    completed_epochs = 0
    if log_file.exists():
        with open(log_file, "r") as f:
            training_log = json.load(f)
        completed_epochs = len(training_log.get("epoch_losses", []))
        log(f"  Found existing log with {completed_epochs} completed epoch(s).")
    else:
        training_log = {"label_type": label_type_key, "config": cfg["description"], "epoch_losses": []}

    if completed_epochs >= EPOCHS:
        # All epochs done but final model wasn't saved — save it now
        last_ckpt = model_dir / f"epoch_{completed_epochs}.pt"
        if last_ckpt.exists():
            log(f"  All {EPOCHS} epochs completed. Copying last checkpoint to final model.")
            import shutil
            shutil.copy(last_ckpt, final_model_path)
        return

    # ---- Build dataset ----
    log(f"  Building positive-pairs dataset...")
    dataset = PaperPositivePairsDataset(TRAIN_PARQUET, paper_text_map, label_type_key)

    if len(dataset) == 0:
        log(f"  WARNING: No positive pairs found for {cfg['name']}. Skipping.")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE_FT, shuffle=True, num_workers=2, pin_memory=True)

    # ---- Initialize model ----
    model = ContrastiveSciBERT(MODEL_NAME, embed_dim=EMBED_DIM).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = NTXentLoss()

    # ---- Resume from checkpoint if available ----
    if completed_epochs > 0:
        resume_ckpt = model_dir / f"epoch_{completed_epochs}.pt"
        if resume_ckpt.exists():
            log(f"  Resuming from checkpoint: {resume_ckpt}")
            checkpoint = torch.load(resume_ckpt, map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            log(f"  WARNING: Checkpoint {resume_ckpt} not found. Starting from scratch.")
            completed_epochs = 0
            training_log["epoch_losses"] = []

    # ---- Training loop ----
    for epoch in range(completed_epochs + 1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for paper_texts, ref_texts in tqdm(dataloader, desc=f"  Epoch {epoch}/{EPOCHS}"):
            enc1 = tokenizer(
                list(paper_texts), padding=True, truncation=True,
                max_length=MAX_LEN, return_tensors="pt",
            ).to(DEVICE)
            enc2 = tokenizer(
                list(ref_texts), padding=True, truncation=True,
                max_length=MAX_LEN, return_tensors="pt",
            ).to(DEVICE)

            emb1 = model(enc1.input_ids, enc1.attention_mask)
            emb2 = model(enc2.input_ids, enc2.attention_mask)

            loss = criterion(emb1, emb2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        log(f"  Epoch {epoch} — Avg Loss: {avg_loss:.4f}")

        # Save checkpoint (model + optimizer for exact resume)
        ckpt_path = model_dir / f"epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        }, ckpt_path)

        training_log["epoch_losses"].append(avg_loss)
        with open(log_file, "w") as f:
            json.dump(training_log, f, indent=2)

    # ---- Save final model (just the model weights for embedding generation) ----
    torch.save(model.state_dict(), final_model_path)
    log(f"  Final model saved to {final_model_path}")


# ======================================================================
# Main
# ======================================================================

def main():
    log("=" * 80)
    log("SciBERT + NT-Xent Fine-Tuning (3 Label Types)")
    log("=" * 80)
    log(f"Python: {sys.version.split()[0]}")
    log(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    log(f"\nModel: {MODEL_NAME}")
    log(f"Projection dim: {EMBED_DIM}")
    log(f"Epochs: {EPOCHS}, LR: {LR}, Batch size: {BATCH_SIZE_FT}")

    # ---- Load paper texts ----
    # Train papers are the primary source; all_papers is for lookup of cited papers
    # that may not be in the train set
    log("\nLoading paper text maps...")
    train_map = load_paper_text_map(TRAIN_PARQUET)
    log(f"  Train papers: {len(train_map)}")

    # all_papers for lookup of references that point outside the train set
    if Path(ALL_PAPERS_PARQUET).exists():
        all_map = load_paper_text_map(ALL_PAPERS_PARQUET)
        log(f"  All papers (lookup): {len(all_map)}")
        paper_text_map = {**all_map, **train_map}  # train overrides all_papers
    else:
        log(f"  All papers parquet not found, using train only.")
        paper_text_map = train_map

    log(f"  Combined text map: {len(paper_text_map)} papers")

    # ---- Load tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ---- Fine-tune for each label type ----
    for label_key, label_cfg in LABEL_TYPES.items():
        log(f"\n{'=' * 80}")
        log(f"Fine-tuning for: {label_cfg['name']}  ({label_cfg['description']})")
        log(f"{'=' * 80}")
        fine_tune_for_label(label_key, paper_text_map, tokenizer)

    log(f"\n{'=' * 80}")
    log("All fine-tuning complete!")
    log(f"Models saved to: {MODELS_DIR}")
    log(f"{'=' * 80}")


if __name__ == "__main__":
    main()