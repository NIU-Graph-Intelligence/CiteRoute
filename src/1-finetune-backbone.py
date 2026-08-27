"""
Stage 1 — Fine-tune the contrastive dense backbone (Sec. 3.2, Eq. 1)
=====================================================================
Trains one ContrastiveEncoder per label type with NT-Xent loss on the train
core set. This is the backbone every dense facet view is embedded with; the
paper's argument is that factorization sits ON TOP of the strongest baseline
signal, not instead of it.

Input:
  - data/train_eval_set/<ver>/train_<ver>.parquet
  - data/train_eval_set/<ver>/all_papers_with_refs_and_labels.parquet (text lookup)

Output (per label type):
  - OUTPUT_DIR/backbone/<type_key>/final_model.pt
  - OUTPUT_DIR/backbone/<type_key>/epoch_<N>.pt
  - OUTPUT_DIR/backbone/<type_key>/training_log.json

Resume: finished types are skipped; interrupted runs resume from the last epoch
checkpoint (model + optimizer state).

Two ways to avoid training from scratch:

  --reuse-baseline   Adopt an ALREADY fine-tuned SciBERT-NTX baseline checkpoint
                     as the backbone (no training at all). The architectures are
                     identical (encoder + 2-layer projection head), and the
                     baseline was trained with the same NT-Xent objective on the
                     same train set, so its weights ARE the backbone this stage
                     would otherwise reproduce. Set BASELINE_BACKBONE_DIR in .env
                     or pass --baseline-dir. The checkpoint is copied into
                     OUTPUT_DIR (not referenced in place) so this repo's output
                     stays self-contained and reproducible.

  --init-from <path> Warm-start training from that checkpoint (still trains).

CUDA_VISIBLE_DEVICES=1 nohup python -u src/1-finetune-backbone.py --types type_3 > logs/step_1_ft_backbone_type_3.log 2>&1 &
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from citeroute.config import (ALL_PAPERS_PARQUET, BACKBONE_DIR, POOLING,
                             BACKBONE_MODEL_NAME, BASELINE_BACKBONE_DIR,
                             BATCH_SIZE_FT, EMBED_DIM, EPOCHS, LABEL_TYPES, LR,
                             MAX_LEN, TRAIN_PARQUET, check_inputs,
                             describe_paths, ensure_dirs)
from citeroute.data import load_paper_text_map
from citeroute.model import ContrastiveEncoder, NTXentLoss, PositivePairsDataset
from citeroute.utils import banner, log

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def baseline_checkpoint(baseline_dir, label_key):
    """Locate a baseline checkpoint for one label type.

    Tolerant of how the path is given, so all of these work:
      .../SciBERT-NTXent
      .../SciBERT-NTXent/fine_tuned_models
      .../SciBERT-NTXent/fine_tuned_models/type_1/final_model.pt
    If final_model.pt is absent (e.g. training still running), the highest
    epoch_N.pt is used instead and the choice is logged.
    """
    p = Path(baseline_dir)
    if p.is_file():
        return p

    roots = [p, p / "fine_tuned_models"]
    for root in roots:
        for cand in [root / label_key / "final_model.pt", root / f"{label_key}.pt"]:
            if cand.exists():
                return cand
    # Fall back to the newest epoch checkpoint.
    for root in roots:
        d = root / label_key
        if d.is_dir():
            epochs = sorted(d.glob("epoch_*.pt"),
                            key=lambda x: int(x.stem.split("_")[1]))
            if epochs:
                log(f"    (no final_model.pt for {label_key}; using {epochs[-1].name})")
                return epochs[-1]
    return None


def adopt_baseline(label_key, baseline_dir):
    """Copy an existing fine-tuned checkpoint in as this repo's backbone.

    Verified before copying: the state dict must actually load into
    ContrastiveEncoder, so a shape/architecture mismatch fails here rather
    than silently producing garbage embeddings in Stage 2.
    """
    model_dir = BACKBONE_DIR / label_key
    model_dir.mkdir(parents=True, exist_ok=True)
    final_path = model_dir / "final_model.pt"
    if final_path.exists():
        log(f"  Backbone already present at {final_path} — skipping.")
        return True

    src = baseline_checkpoint(baseline_dir, label_key)
    if src is None:
        log(f"  No baseline checkpoint for {label_key} under {baseline_dir} — skipping.")
        return False

    log(f"  Adopting baseline checkpoint: {src}")
    state = torch.load(src, map_location="cpu", weights_only=False)
    state = state.get("model_state_dict", state)

    model = ContrastiveEncoder(BACKBONE_MODEL_NAME, embed_dim=EMBED_DIM)
    missing, unexpected = model.load_state_dict(state, strict=False)
    n_loaded = len(model.state_dict()) - len(missing)
    log(f"    Loaded {n_loaded}/{len(model.state_dict())} tensors "
        f"(missing={len(missing)}, unexpected={len(unexpected)})")
    if missing:
        log(f"    First missing keys: {missing[:5]}")
        raise SystemExit(
            "Baseline checkpoint is not architecture-compatible with "
            "ContrastiveEncoder. Check BACKBONE_MODEL / EMBED_DIM in .env "
            f"(currently {BACKBONE_MODEL_NAME}, dim {EMBED_DIM}), or drop "
            "--reuse-baseline and train the backbone with Stage 1.")

    torch.save(model.state_dict(), final_path)
    with open(model_dir / "training_log.json", "w") as f:
        json.dump({"label_type": label_key, "source": "adopted_baseline",
                   "baseline_checkpoint": str(src),
                   "note": "no training performed; weights copied from the "
                           "SciBERT-NTX baseline (identical architecture and "
                           "NT-Xent objective)"}, f, indent=2)
    log(f"  Backbone written to {final_path}")
    return True


def fine_tune(label_key, text_map, tokenizer, init_from=None):
    cfg = LABEL_TYPES[label_key]
    model_dir = BACKBONE_DIR / label_key
    model_dir.mkdir(parents=True, exist_ok=True)
    final_path = model_dir / "final_model.pt"
    log_file = model_dir / "training_log.json"

    if final_path.exists():
        log(f"  Final model exists at {final_path} — skipping.")
        return

    completed = 0
    if log_file.exists():
        with open(log_file) as f:
            training_log = json.load(f)
        completed = len(training_log.get("epoch_losses", []))
        log(f"  Found existing log with {completed} completed epoch(s).")
    else:
        training_log = {"label_type": label_key, "config": cfg["description"],
                        "epoch_losses": []}

    if completed >= EPOCHS:
        last = model_dir / f"epoch_{completed}.pt"
        if last.exists():
            log(f"  All {EPOCHS} epochs done — promoting last checkpoint to final.")
            ckpt = torch.load(last, map_location="cpu", weights_only=False)
            torch.save(ckpt["model_state_dict"], final_path)
        return

    log("  Building positive-pairs dataset...")
    dataset = PositivePairsDataset(TRAIN_PARQUET, text_map, label_key)
    log(f"    {cfg['name']}: {len(dataset)} positive pairs")
    if len(dataset) == 0:
        log(f"  WARNING: no positive pairs for {cfg['name']} — skipping.")
        return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE_FT, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)

    model = ContrastiveEncoder(BACKBONE_MODEL_NAME, embed_dim=EMBED_DIM).to(DEVICE)
    if init_from and completed == 0:
        p = Path(init_from)
        if p.is_dir():
            p = p / label_key / "final_model.pt"
        if p.exists():
            state = torch.load(p, map_location=DEVICE, weights_only=False)
            state = state.get("model_state_dict", state)
            missing, unexpected = model.load_state_dict(state, strict=False)
            log(f"  Warm-started from {p} (missing={len(missing)}, unexpected={len(unexpected)})")
        else:
            log(f"  WARNING: --init-from path not found: {p} — training from pretrained.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = NTXentLoss()

    if completed > 0:
        ckpt_path = model_dir / f"epoch_{completed}.pt"
        if ckpt_path.exists():
            log(f"  Resuming from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        else:
            log(f"  WARNING: {ckpt_path} missing — restarting this label type.")
            completed = 0
            training_log["epoch_losses"] = []

    for epoch in range(completed + 1, EPOCHS + 1):
        model.train()
        total, nb = 0.0, 0
        for q_texts, r_texts in tqdm(loader, desc=f"  Epoch {epoch}/{EPOCHS}"):
            enc1 = tokenizer(list(q_texts), padding=True, truncation=True,
                             max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
            enc2 = tokenizer(list(r_texts), padding=True, truncation=True,
                             max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
            loss = criterion(model(enc1.input_ids, enc1.attention_mask),
                             model(enc2.input_ids, enc2.attention_mask))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            nb += 1

        avg = total / max(nb, 1)
        log(f"  Epoch {epoch} — Avg Loss: {avg:.4f}  (batches: {nb})")
        if nb == 0:
            raise RuntimeError(f"Epoch {epoch} had zero batches — refusing to save an untrained model.")

        torch.save({"epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg}, model_dir / f"epoch_{epoch}.pt")
        training_log["epoch_losses"].append(avg)
        with open(log_file, "w") as f:
            json.dump(training_log, f, indent=2)

    torch.save(model.state_dict(), final_path)
    log(f"  Final model saved to {final_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-baseline", action="store_true",
                    help="Adopt an already fine-tuned SciBERT-NTX checkpoint as "
                         "the backbone instead of training (no GPU time).")
    ap.add_argument("--baseline-dir", default=BASELINE_BACKBONE_DIR,
                    help="Directory holding <type_key>/final_model.pt "
                         "(default: BASELINE_BACKBONE_DIR from .env)")
    ap.add_argument("--init-from", default=None,
                    help="Warm-start training from an existing checkpoint "
                         "(file, or a directory containing <type_key>/final_model.pt)")
    ap.add_argument("--types", nargs="*", default=list(LABEL_TYPES),
                    help="Subset of label types to train")
    args = ap.parse_args()

    ensure_dirs()
    banner("Stage 1 — Contrastive backbone (NT-Xent)")
    for line in describe_paths():
        log(f"  {line}")
    log(f"Python: {sys.version.split()[0]}  |  Device: {DEVICE}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"Model: {BACKBONE_MODEL_NAME} | dim {EMBED_DIM} | epochs {EPOCHS} "
        f"| lr {LR} | batch {BATCH_SIZE_FT}")

    # ---- reuse path: no training, no data loading ----
    if args.reuse_baseline:
        if not args.baseline_dir:
            raise SystemExit(
                "--reuse-baseline needs a checkpoint location. Set "
                "BASELINE_BACKBONE_DIR in .env or pass --baseline-dir "
                "<path to SciBERT-NTX fine_tuned_models>.")
        banner(f"Reusing baseline backbone from {args.baseline_dir}", char="-")
        log(f"  Encoder architecture: {BACKBONE_MODEL_NAME} "
            f"({POOLING} pooling, dim {EMBED_DIM})")
        log(f"  The checkpoint must match this architecture — set BACKBONE_MODEL "
            f"accordingly (e.g. allenai/scibert_scivocab_uncased for SciBERT-NTX).")
        done, missing = [], []
        for key in args.types:
            log(f"  {LABEL_TYPES[key]['name']}")
            (done if adopt_baseline(key, args.baseline_dir) else missing).append(key)
        if not done:
            raise SystemExit(
                f"No checkpoints found under {args.baseline_dir}. Expected "
                f"<type_key>/final_model.pt, optionally under fine_tuned_models/.")
        log("")
        log(f"  Adopted : {', '.join(done)}")
        if missing:
            log(f"  MISSING : {', '.join(missing)} — these types have no backbone, so "
                f"later stages will skip them. Re-run this stage for them once "
                f"their checkpoints exist.")
        banner(f"Adopted {len(done)} backbone(s) — {BACKBONE_DIR}")
        return

    check_inputs([TRAIN_PARQUET],
                 hint="Set DATA_DIR (and DATA_VERSION) in .env for this server.")

    log("\nLoading paper texts...")
    text_map = load_paper_text_map(TRAIN_PARQUET)
    log(f"  Train papers: {len(text_map)}")
    if Path(ALL_PAPERS_PARQUET).exists():
        all_map = load_paper_text_map(ALL_PAPERS_PARQUET)
        log(f"  All papers (lookup): {len(all_map)}")
        text_map = {**all_map, **text_map}
    log(f"  Combined text map: {len(text_map)} papers")

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_NAME)

    for key in args.types:
        banner(f"Fine-tuning: {LABEL_TYPES[key]['name']} ({LABEL_TYPES[key]['description']})",
               char="-")
        fine_tune(key, text_map, tokenizer, init_from=args.init_from)

    banner(f"All backbones complete — {BACKBONE_DIR}")


if __name__ == "__main__":
    main()
