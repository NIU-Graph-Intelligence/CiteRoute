"""
ColBERT + NT-Xent Fine-Tuning Script
======================================
Fine-tunes ColBERTv2 with NT-Xent contrastive loss, separately for each
of the 3 label types. Uses mean-pooled token embeddings for the contrastive
objective while preserving ColBERT's late-interaction architecture.

Resume: skips completed models, resumes from last epoch checkpoint.

Output (per label type):
  - output/dense/ColBERT-NTX/fine_tuned_models/<type_key>/final_model/
  - output/dense/ColBERT-NTX/fine_tuned_models/<type_key>/epoch_<N>/
"""

import os
import sys
import time
import json
import shutil
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, logging as hf_logging
from dotenv import load_dotenv
from unittest.mock import MagicMock

hf_logging.set_verbosity_error()

try:
    import git
    try: git.Repo(search_parent_directories=True)
    except: sys.modules['git'] = MagicMock()
except ImportError:
    sys.modules['git'] = MagicMock()

from colbert.modeling.colbert import ColBERT
from colbert.infra import ColBERTConfig

from colbert_ntx_utils import (
    log, load_paper_text_map, ColBERTNTXentLoss, ColBERTPositivePairsDataset,
    LABEL_TYPES, CHECKPOINT, DOC_MAXLEN, BATCH_SIZE_FT, EPOCHS, LR, TEMPERATURE,
)

# -------- Config --------
load_dotenv()
ROOT_DIR = Path(os.getenv("ROOT_DIR", "/home/ratul/mustcite/"))
OUTPUT_DIR = ROOT_DIR / "masterset-benchmark/output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PARQUET = ROOT_DIR / "data/train_eval_set/v7.0/train_v7.0.parquet"
ALL_PAPERS_PARQUET = ROOT_DIR / "data/train_eval_set/v7.0/all_papers_with_refs_and_labels.parquet"

MODELS_DIR = OUTPUT_DIR / "dense/ColBERT-NTX/fine_tuned_models/"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(os.getenv("TORCH_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)


def encode_docs(model, inputs):
    """
    Differentiable re-implementation of ColBERT.doc().

    Uses the underlying HF encoder + projection directly so gradients are
    guaranteed to flow (some colbert-ai builds detach inside doc(), which
    silently turns every training batch into a no-op). Padding tokens are
    zeroed via the attention mask so the NT-Xent aggregation ignores them.
    """
    D = model.bert(inputs["input_ids"], attention_mask=inputs["attention_mask"])[0]
    D = model.linear(D)
    D = D * inputs["attention_mask"].unsqueeze(2).to(D.dtype)
    D = torch.nn.functional.normalize(D, p=2, dim=2)
    return D


def fine_tune_for_label(label_type_key, paper_text_map):
    cfg = LABEL_TYPES[label_type_key]
    model_dir = MODELS_DIR / label_type_key
    model_dir.mkdir(parents=True, exist_ok=True)

    final_model_dir = model_dir / "final_model"
    log_file = model_dir / "training_log.json"

    if final_model_dir.exists() and any(final_model_dir.iterdir()):
        log(f"  Final model already exists — skipping.")
        return

    # Find last completed epoch
    completed_epochs = 0
    if log_file.exists():
        with open(log_file, "r") as f:
            training_log = json.load(f)
        completed_epochs = len(training_log.get("epoch_losses", []))
        log(f"  Found {completed_epochs} completed epoch(s).")
    else:
        training_log = {"label_type": label_type_key, "epoch_losses": []}

    if completed_epochs >= EPOCHS:
        last_ckpt = model_dir / f"epoch_{completed_epochs}"
        if last_ckpt.exists():
            shutil.copytree(last_ckpt, final_model_dir, dirs_exist_ok=True)
        return

    # Build dataset
    log(f"  Building positive-pairs dataset...")
    dataset = ColBERTPositivePairsDataset(TRAIN_PARQUET, paper_text_map, label_type_key)
    if len(dataset) == 0:
        log(f"  WARNING: No positive pairs. Skipping.")
        return
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE_FT, shuffle=True)

    # Initialize ColBERT model
    colbert_config = ColBERTConfig()
    colbert_config.doc_maxlen = DOC_MAXLEN
    colbert_config.query_maxlen = DOC_MAXLEN
    colbert_config.checkpoint = CHECKPOINT

    model = ColBERT(name=CHECKPOINT, colbert_config=colbert_config).to(DEVICE)
    # Make absolutely sure the encoder is trainable: the installed colbert
    # package may freeze parameters or detach outputs inside doc().
    for p in model.parameters():
        p.requires_grad_(True)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  Trainable parameters: {n_trainable:,}")
    VOCAB_SIZE = model.bert.config.vocab_size
    criterion = ColBERTNTXentLoss(temperature=TEMPERATURE).to(DEVICE)
    # Include the criterion's learnable temperature in the optimizer;
    # without this it is declared learnable but never updated.
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()), lr=LR)
    scaler = GradScaler("cuda")

    # Resume from checkpoint
    if completed_epochs > 0:
        resume_dir = model_dir / f"epoch_{completed_epochs}"
        if resume_dir.exists():
            log(f"  Resuming from {resume_dir}")
            # Reload model FIRST, then rebuild the optimizer over the new
            # instance's parameters (otherwise the optimizer keeps updating
            # the discarded model and training silently does nothing).
            try:
                model = ColBERT(name=str(resume_dir), colbert_config=colbert_config).to(DEVICE)
            except Exception:
                log(f"  Could not reload from checkpoint, loading state_dict fallback")
                sd_path = resume_dir / "model.pt"
                if sd_path.exists():
                    model.load_state_dict(torch.load(sd_path, map_location=DEVICE, weights_only=False))
            optimizer = torch.optim.AdamW(
                list(model.parameters()) + list(criterion.parameters()), lr=LR)
            opt_path = resume_dir / "optimizer.pt"
            if opt_path.exists():
                optimizer.load_state_dict(torch.load(opt_path, map_location=DEVICE, weights_only=False))
        else:
            completed_epochs = 0
            training_log["epoch_losses"] = []

    model.train()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    for epoch in range(completed_epochs + 1, EPOCHS + 1):
        epoch_loss = 0.0
        num_batches = 0
        skipped_degenerate = 0
        skipped_detached = 0
        skipped_bad_ids = 0
        skipped_no_grad = 0

        for paper_texts, ref_texts in tqdm(dataloader, desc=f"  Epoch {epoch}/{EPOCHS}"):
            try:
                paper_inputs = TOKENIZER(list(paper_texts), padding="max_length", truncation=True,
                                         max_length=DOC_MAXLEN, return_tensors="pt")
                ref_inputs = TOKENIZER(list(ref_texts), padding="max_length", truncation=True,
                                       max_length=DOC_MAXLEN, return_tensors="pt")
                # Guard against out-of-vocab token ids: on the GPU these cause
                # an unrecoverable device-side assert (gather index OOB).
                max_id = max(int(paper_inputs["input_ids"].max()), int(ref_inputs["input_ids"].max()))
                if max_id >= VOCAB_SIZE:
                    if skipped_bad_ids == 0:
                        log(f"  [diag] token id {max_id} >= vocab size {VOCAB_SIZE} — skipping batch")
                    skipped_bad_ids += 1
                    continue
                paper_inputs = {k: v.to(DEVICE) for k, v in paper_inputs.items() if k != "token_type_ids"}
                ref_inputs = {k: v.to(DEVICE) for k, v in ref_inputs.items() if k != "token_type_ids"}

                with autocast("cuda"):
                    # Bypass model.doc(): in some colbert-ai versions it
                    # detaches outputs (no-grad path), which made EVERY batch
                    # a zero-gradient batch. bert+linear is the same encoder,
                    # guaranteed differentiable.
                    paper_emb = encode_docs(model, paper_inputs)
                    ref_emb = encode_docs(model, ref_inputs)

                loss = criterion(paper_emb, ref_emb)
                if loss is None:
                    if skipped_degenerate == 0:
                        log("  [diag] degenerate batch: aggregated embeddings "
                            "collapsed to zero norm")
                    skipped_degenerate += 1
                    continue
                if loss.grad_fn is None:
                    if skipped_detached == 0:
                        log(f"  [diag] detached batch: paper_emb.requires_grad="
                            f"{paper_emb.requires_grad} — encoder output has no "
                            f"grad path")
                    skipped_detached += 1
                    continue
                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if not any(p.grad is not None for g in optimizer.param_groups for p in g["params"]):
                    # backward reached none of the optimizer's params; stepping
                    # would raise "No inf checks were recorded" and poison the
                    # scaler state for every later batch.
                    if skipped_no_grad == 0:
                        n_req = sum(p.requires_grad for p in model.parameters())
                        log(f"  [diag] no-grad batch: loss.grad_fn={loss.grad_fn is not None}, "
                            f"model params requiring grad: {n_req}")
                    skipped_no_grad += 1
                    optimizer.zero_grad(set_to_none=True)
                    scaler = GradScaler("cuda", init_scale=scaler.get_scale())
                    continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                num_batches += 1
            except Exception as e:
                msg = str(e)
                if "device-side assert" in msg or "CUDA error" in msg:
                    # The CUDA context is corrupted beyond recovery; every
                    # later batch would fail too. Abort loudly instead of
                    # burning hours on a dead context.
                    log(f"  FATAL CUDA error — aborting run: {msg.splitlines()[0]}")
                    raise
                log(f"  Error in batch: {e}")
                # Reset optimizer grads and scaler bookkeeping: if the error
                # hit between unscale_() and update(), the scaler is stuck
                # mid-cycle and every following batch would fail with
                # "unscale_() has already been called ...".
                optimizer.zero_grad(set_to_none=True)
                scaler = GradScaler("cuda", init_scale=scaler.get_scale())
                continue

        avg_loss = epoch_loss / max(num_batches, 1)
        log(f"  Epoch {epoch} — Avg Loss: {avg_loss:.4f}  "
            f"(batches: {num_batches}, degenerate: {skipped_degenerate}, "
            f"detached: {skipped_detached}, bad-ids: {skipped_bad_ids}, "
            f"no-grad: {skipped_no_grad})")
        if num_batches == 0:
            raise RuntimeError(
                f"Epoch {epoch}: zero successful batches — the model has not "
                f"been trained at all. Refusing to save a checkpoint that "
                f"would masquerade as fine-tuned. Check the [diag] lines above.")

        # Save checkpoint (ColBERT-style dir + optimizer)
        ckpt_dir = model_dir / f"epoch_{epoch}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        try:
            model.save(str(ckpt_dir))
        except Exception:
            torch.save(model.state_dict(), ckpt_dir / "model.pt")
        torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")

        training_log["epoch_losses"].append(avg_loss)
        with open(log_file, "w") as f:
            json.dump(training_log, f, indent=2)

    # Save final model
    final_model_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.save(str(final_model_dir))
    except Exception:
        torch.save(model.state_dict(), final_model_dir / "model.pt")

    # Always ensure a valid ColBERT config JSON exists for the Indexer
    config_path = final_model_dir / "artifact.metadata"
    if not config_path.exists() or config_path.stat().st_size == 0:
        import ujson
        colbert_meta = {
            "query_token_id": "[unused0]", "doc_token_id": "[unused1]",
            "query_token": "[Q]", "doc_token": "[D]",
            "similarity": "cosine", "dim": 128, "doc_maxlen": DOC_MAXLEN,
            "query_maxlen": DOC_MAXLEN, "mask_punctuation": True,
            "checkpoint": CHECKPOINT,  # original base model name
        }
        with open(config_path, "w") as f:
            ujson.dump(colbert_meta, f)
        log(f"  Wrote ColBERT config to {config_path}")

    log(f"  Final model saved to {final_model_dir}")


def main():
    log("=" * 80)
    log("ColBERT + NT-Xent Fine-Tuning (3 Label Types)")
    log("=" * 80)
    log(f"Device: {DEVICE}")

    log("\nLoading paper text maps...")
    train_map = load_paper_text_map(TRAIN_PARQUET)
    if Path(ALL_PAPERS_PARQUET).exists():
        all_map = load_paper_text_map(ALL_PAPERS_PARQUET)
        paper_text_map = {**all_map, **train_map}
    else:
        paper_text_map = train_map
    log(f"  Combined text map: {len(paper_text_map)} papers")

    for label_key, label_cfg in LABEL_TYPES.items():
        log(f"\n{'=' * 80}")
        log(f"Fine-tuning for: {label_cfg['name']}")
        log(f"{'=' * 80}")
        fine_tune_for_label(label_key, paper_text_map)

    log("\nAll fine-tuning complete!")


if __name__ == "__main__":
    main()