"""
HAtten (Hierarchical Attention) — Fine-Tuning Script
======================================================
Fine-tunes a HAtten model (GloVe + Transformer + Multi-Head Pooling)
with triplet loss, separately for each of the 3 label types.

Uses in-batch negatives (shuffled positives) for efficiency.

Resume: skips completed models, resumes from last epoch checkpoint.

Output (per label type):
  - output/dense/HAtten-RR/fine_tuned_models/<type_key>/final_model.pt
  - output/dense/HAtten-RR/fine_tuned_models/<type_key>/epoch_<N>.pt
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from pathlib import Path
from tqdm import tqdm
from functools import partial
from dotenv import load_dotenv

from hatten_utils import (
    log, load_paper_text_map, load_glove_embeddings, simple_tokenize,
    HAttenModel, HAttenTripletDataset, hatten_collate_fn, triplet_loss,
    LABEL_TYPES, GLOVE_DIM, HIDDEN_DIM, NUM_HEADS,
    BATCH_SIZE_FT, EPOCHS, LR, WEIGHT_DECAY, MARGIN, NITER_CHECKPOINT,
)

# -------- Config --------
load_dotenv()
ROOT_DIR = Path(os.getenv("ROOT_DIR", "/home/ratul/mustcite/"))
OUTPUT_DIR = ROOT_DIR / "masterset-benchmark/output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PARQUET = ROOT_DIR / "data/train_eval_set/v7.0/train_v7.0.parquet"
ALL_PAPERS_PARQUET = ROOT_DIR / "data/train_eval_set/v7.0/all_papers_with_refs_and_labels.parquet"

MODELS_DIR = OUTPUT_DIR / "dense/HAtten-RR/fine_tuned_models/"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fine_tune_for_label(label_type_key, paper_text_map, embeddings, word2idx):
    cfg = LABEL_TYPES[label_type_key]
    model_dir = MODELS_DIR / label_type_key
    model_dir.mkdir(parents=True, exist_ok=True)

    final_model_path = model_dir / "final_model.pt"
    log_file = model_dir / "training_log.json"

    if final_model_path.exists():
        log(f"  Final model already exists — skipping.")
        return

    # Resume
    completed_epochs = 0
    if log_file.exists():
        with open(log_file, "r") as f:
            training_log = json.load(f)
        completed_epochs = len(training_log.get("epoch_losses", []))
        log(f"  Found {completed_epochs} completed epoch(s).")
    else:
        training_log = {"label_type": label_type_key, "epoch_losses": []}

    if completed_epochs >= EPOCHS:
        last_ckpt = model_dir / f"epoch_{completed_epochs}.pt"
        if last_ckpt.exists():
            ckpt = torch.load(last_ckpt, map_location="cpu", weights_only=False)
            torch.save({
                "model_state_dict": ckpt["model_state_dict"],
                "vocab_size": len(word2idx), "embedding_dim": GLOVE_DIM,
                "hidden_dim": HIDDEN_DIM, "num_heads": NUM_HEADS, "word2idx": word2idx,
            }, final_model_path)
        return

    # Dataset
    log(f"  Building triplet dataset...")
    dataset = HAttenTripletDataset(TRAIN_PARQUET, paper_text_map, label_type_key)
    if len(dataset) == 0:
        log(f"  WARNING: No triplets. Skipping.")
        return

    collate = partial(hatten_collate_fn, paper_text_map=paper_text_map)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE_FT, shuffle=True, collate_fn=collate, num_workers=0)

    # Model
    model = HAttenModel(
        vocab_size=len(word2idx), embedding_dim=GLOVE_DIM, hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS, pretrained_embeddings=embeddings, freeze_embeddings=True,
    )
    model.set_word2idx(word2idx)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler("cuda")

    # Resume
    if completed_epochs > 0:
        resume_ckpt = model_dir / f"epoch_{completed_epochs}.pt"
        if resume_ckpt.exists():
            log(f"  Resuming from {resume_ckpt}")
            ckpt = torch.load(resume_ckpt, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        else:
            completed_epochs = 0
            training_log["epoch_losses"] = []

    model.train()
    global_step = 0

    for epoch in range(completed_epochs + 1, EPOCHS + 1):
        epoch_loss = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc=f"  Epoch {epoch}/{EPOCHS}"):
            try:
                query_emb = model(batch["query_paragraphs"], batch["query_types"])
                pos_emb = model(batch["pos_paragraphs"], batch["pos_types"])

                # In-batch negatives: shuffle positives
                batch_size = query_emb.size(0)
                if batch_size > 1:
                    neg_indices = torch.randperm(batch_size)
                    neg_emb = pos_emb[neg_indices]
                else:
                    neg_emb = pos_emb

                loss = triplet_loss(query_emb, pos_emb, neg_emb, margin=MARGIN)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                if loss.grad_fn is None:
                    continue
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1

                if global_step % NITER_CHECKPOINT == 0:
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "global_step": global_step,
                    }, model_dir / f"iter_{global_step}.pt")

            except Exception as e:
                log(f"  Error: {e}")
                # Reset scaler bookkeeping so a failed step cannot poison
                # every subsequent batch.
                optimizer.zero_grad(set_to_none=True)
                scaler = GradScaler("cuda", init_scale=scaler.get_scale())
                continue

        avg_loss = epoch_loss / max(num_batches, 1)
        log(f"  Epoch {epoch} — Avg Loss: {avg_loss:.4f}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        }, model_dir / f"epoch_{epoch}.pt")

        training_log["epoch_losses"].append(avg_loss)
        with open(log_file, "w") as f:
            json.dump(training_log, f, indent=2)

    # Save final model (includes word2idx for embedding generation)
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": len(word2idx), "embedding_dim": GLOVE_DIM,
        "hidden_dim": HIDDEN_DIM, "num_heads": NUM_HEADS, "word2idx": word2idx,
    }, final_model_path)
    log(f"  Final model saved to {final_model_path}")


def main():
    log("=" * 80)
    log("HAtten — Fine-Tuning (3 Label Types)")
    log("=" * 80)
    log(f"Device: {DEVICE}")

    log("\nLoading paper text maps...")
    train_map = load_paper_text_map(TRAIN_PARQUET)
    if Path(ALL_PAPERS_PARQUET).exists():
        all_map = load_paper_text_map(ALL_PAPERS_PARQUET)
        paper_text_map = {**all_map, **train_map}
    else:
        paper_text_map = train_map
    log(f"  Combined: {len(paper_text_map)} papers")

    # Build vocab from text
    log("Building vocabulary...")
    vocab = set()
    for text in list(paper_text_map.values())[:10000]:
        vocab.update(simple_tokenize(text))

    glove_path = os.getenv("GLOVE_PATH")
    embeddings, word2idx = load_glove_embeddings(glove_path, vocab, GLOVE_DIM)
    log(f"  Vocab size: {len(word2idx)}")

    for label_key, label_cfg in LABEL_TYPES.items():
        log(f"\n{'=' * 80}")
        log(f"Fine-tuning for: {label_cfg['name']}")
        log(f"{'=' * 80}")
        fine_tune_for_label(label_key, paper_text_map, embeddings, word2idx)

    log("\nAll fine-tuning complete!")


if __name__ == "__main__":
    main()
