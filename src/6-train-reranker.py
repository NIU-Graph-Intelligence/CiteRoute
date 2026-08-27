"""
Stage 6 — Train the cross-encoder reranker (Eqs. 6-7)
======================================================
Positives are the gold must-cite papers of TRAIN queries; negatives are HARD
negatives mined from the fused top-C of the same queries (retrieved but not
cited) — the discriminations the reranker will actually have to make at
inference time.

Input : OUTPUT_DIR/runs/<type_key>/reranker_<view>.npz (Stage 4, split=reranker)
        OUTPUT_DIR/fusion/<type_key>_weights.json      (optional, Stage 5)
Output: OUTPUT_DIR/reranker/<type_key>/final_model.pt
        OUTPUT_DIR/reranker/<type_key>/training_log.json

Resume: types with a final model are skipped; epoch checkpoints allow resume.
"""

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from citeroute.config import (ALL_PAPERS_PARQUET, CANDIDATE_PARQUET,
                             FACETS_PARQUET, FUSION_DIR, LABEL_TYPES,
                             NEGATIVES_PER_POSITIVE, RERANK_DEPTH,
                             RERANKER_BATCH_SIZE, RERANKER_DIR,
                             RERANKER_EPOCHS, RERANKER_LR, RERANKER_MAX_LEN,
                             RERANKER_MODEL_NAME, SEED, TRAIN_PARQUET,
                             ensure_dirs)
from citeroute.data import (build_ground_truth, load_facet_map,
                           load_paper_text_map, paper_ids)
from citeroute.fusion import rrf_fuse
from citeroute.rerank import CrossEncoderReranker, PairDataset, build_query_text
from citeroute.runs import load_view_runs
from citeroute.utils import banner, log

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_weights(label_key):
    p = FUSION_DIR / f"{label_key}_weights.json"
    if p.exists():
        return json.load(open(p))["weights"]
    log("    (no learned fusion weights — using uniform RRF for negative mining)")
    return None


def collect_superficial(pool_ids, query_ids):
    """{query_id: {paper_id}} for Type-4 superficial citations (label 1.0)."""
    import polars as pl
    from citeroute.data import parse_references
    out = {}
    df = pl.read_parquet(TRAIN_PARQUET)
    for row in df.iter_rows(named=True):
        qid = row["paper_id"]
        if qid not in query_ids:
            continue
        s = set()
        for ref in parse_references(row):
            mid = ref.get("matched_paper_id")
            raw = ref.get("type_4_output")
            if not mid or raw in (None, ""):
                continue
            try:
                if float(raw) == 1.0 and mid in pool_ids:
                    s.add(mid)
            except (TypeError, ValueError):
                continue
        if s:
            out[qid] = s
    return out


def build_pairs(label_key, facet_map, doc_text, pool_ids):
    per_query, _ = load_view_runs(label_key, "reranker")
    if not per_query:
        return []
    superficial_negs = (collect_superficial(pool_ids, set(per_query))
                        if label_key == "type_4" else {})
    gt, _ = build_ground_truth(TRAIN_PARQUET, label_key, pool_ids=pool_ids,
                               restrict_to=set(per_query))
    weights = load_weights(label_key)
    rng = np.random.default_rng(SEED)

    pairs = []
    for qid, view_runs in per_query.items():
        gold = gt.get(qid)
        if not gold:
            continue
        q_text = build_query_text(facet_map.get(qid), facet_map.get(qid, {}).get("full", ""))
        fused, _ = rrf_fuse(view_runs, weights=weights, depth=RERANK_DEPTH)

        n_pos = 0
        for pid in gold:
            d = doc_text.get(pid)
            if d:
                pairs.append((q_text, d, 1))
                n_pos += 1
        if n_pos == 0:
            continue

        # Hard negatives: retrieved-but-not-gold. For Type 4 the labels give
        # us something better — SUPERFICIAL citations (cited by this query but
        # not core) are the exact discrimination the reranker must learn, so
        # they are placed first in the negative pool.
        hard = [p for p in fused if p not in gold and p != qid]
        if label_key == "type_4":
            superficial = superficial_negs.get(qid, set())
            hard = ([p for p in hard if p in superficial]
                    + [p for p in hard if p not in superficial])
        k = min(len(hard), n_pos * NEGATIVES_PER_POSITIVE)
        if k > 0:
            for j in rng.choice(len(hard), size=k, replace=False):
                d = doc_text.get(hard[j])
                if d:
                    pairs.append((q_text, d, 0))
    rng.shuffle(pairs)
    return pairs


def train_one(label_key, pairs, tokenizer):
    out_dir = RERANKER_DIR / label_key
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "final_model.pt"
    if final_path.exists():
        log(f"  Final reranker exists at {final_path} — skipping.")
        return
    if not pairs:
        log("  No training pairs — skipping.")
        return

    n_pos = sum(1 for _, _, y in pairs if y == 1)
    log(f"  Pairs: {len(pairs)} ({n_pos} positive, {len(pairs) - n_pos} hard negative)")

    ds = PairDataset(pairs, tokenizer, max_length=RERANKER_MAX_LEN)
    dl = DataLoader(ds, batch_size=RERANKER_BATCH_SIZE, shuffle=True,
                    num_workers=2, pin_memory=True)

    model = CrossEncoderReranker(RERANKER_MODEL_NAME).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=RERANKER_LR)
    # Positive:negative imbalance is 1:NEGATIVES_PER_POSITIVE by construction.
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(NEGATIVES_PER_POSITIVE), device=DEVICE))

    log_data = {"label_type": label_key, "epoch_losses": [], "num_pairs": len(pairs)}
    for epoch in range(1, RERANKER_EPOCHS + 1):
        model.train()
        total, nb = 0.0, 0
        for batch in tqdm(dl, desc=f"  Epoch {epoch}/{RERANKER_EPOCHS}"):
            kwargs = {"input_ids": batch["input_ids"].to(DEVICE),
                      "attention_mask": batch["attention_mask"].to(DEVICE)}
            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(DEVICE)
            logits = model(**kwargs)
            loss = criterion(logits, batch["label"].to(DEVICE))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            nb += 1
        avg = total / max(nb, 1)
        log(f"  Epoch {epoch} — Avg Loss: {avg:.4f}  (batches: {nb})")
        if nb == 0:
            raise RuntimeError("Reranker epoch had zero batches — refusing to save.")
        log_data["epoch_losses"].append(avg)
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict()},
                   out_dir / f"epoch_{epoch}.pt")
        with open(out_dir / "training_log.json", "w") as f:
            json.dump(log_data, f, indent=2)

    torch.save(model.state_dict(), final_path)
    log(f"  Final reranker saved to {final_path}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", nargs="*", default=list(LABEL_TYPES))
    args = ap.parse_args()

    ensure_dirs()
    banner("Stage 6 — Cross-encoder reranker training")
    log(f"Device: {DEVICE} | model {RERANKER_MODEL_NAME} | max_len {RERANKER_MAX_LEN}")

    pool_ids = set(paper_ids(CANDIDATE_PARQUET))
    doc_text = load_paper_text_map(CANDIDATE_PARQUET)
    q_text_map = load_paper_text_map(TRAIN_PARQUET)
    if ALL_PAPERS_PARQUET.exists():
        doc_text = {**load_paper_text_map(ALL_PAPERS_PARQUET), **doc_text}
    facet_map = load_facet_map(FACETS_PARQUET, text_map=q_text_map)

    tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)

    for label_key in args.types:
        banner(f"{LABEL_TYPES[label_key]['name']}", char="-")
        pairs = build_pairs(label_key, facet_map, doc_text, pool_ids)
        train_one(label_key, pairs, tokenizer)

    banner(f"Rerankers in {RERANKER_DIR}")


if __name__ == "__main__":
    main()
