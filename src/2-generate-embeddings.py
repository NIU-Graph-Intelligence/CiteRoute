"""
Stage 2 — Embedding generation
===============================
Embeds, per label type:
  * the candidate pool (document side, undecomposed title+abstract)
  * every DENSE facet view of every eval query (task, method, builds_on,
    contribution_summary) plus the undecomposed 'full' view

Facet-view embeddings are what make facet-routed dense retrieval possible in
Stage 4: each view becomes its own query vector against the same pool index.

Input:
  - OUTPUT_DIR/backbone/<type_key>/final_model.pt
  - candidate_pool / eval parquets, OUTPUT_DIR/facets/facets.parquet

Output (per label type):
  - OUTPUT_DIR/embeddings/<type_key>/candidates_embeddings.pt
  - OUTPUT_DIR/embeddings/<type_key>/candidates_paper_id_to_index.json
  - OUTPUT_DIR/embeddings/<type_key>/query_<view>_embeddings.pt
  - OUTPUT_DIR/embeddings/<type_key>/query_paper_id_to_index.json

Resume: any artifact that already exists is skipped.
"""

import argparse
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from pathlib import Path

from citeroute.config import (BACKBONE_DIR, BACKBONE_MODEL_NAME,
                             BATCH_SIZE_EMB, CANDIDATE_PARQUET, DENSE_VIEWS,
                             EMBED_DIM, EMBEDDINGS_DIR, EVAL_PARQUET,
                             FACETS_PARQUET, FULL_VIEW, LABEL_TYPES, MAX_LEN,
                             POOLING, SECONDARY_BACKBONE_DIR,
                             SECONDARY_BACKBONE_MODEL, SECONDARY_BACKBONE_NAME,
                             SECONDARY_EMBED_DIM, SECONDARY_POOLING,
                             check_inputs, describe_paths, ensure_dirs)
from citeroute.data import load_facet_map, load_paper_text_map, paper_ids
from citeroute.model import ContrastiveEncoder, TextDataset
from citeroute.utils import banner, log

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def embed(ids, text_lookup, model, tokenizer, out_path, map_path, desc,
          skip_empty=False):
    """Embed `ids`. With skip_empty=True, papers whose text for this view is
    empty (a missing or null facet) are EXCLUDED rather than embedded as an
    empty string — an empty string yields a meaningless [CLS] vector that would
    otherwise pollute that facet's ranked list."""
    if out_path.exists() and (map_path is None or map_path.exists()):
        log(f"    {out_path.name} exists — skipping.")
        return
    if skip_empty:
        get = text_lookup if callable(text_lookup) else text_lookup.get
        kept_ids = [pid for pid in ids if (get(pid) or "").strip()]
        n_drop = len(ids) - len(kept_ids)
        if n_drop:
            log(f"    {desc.strip()}: {n_drop}/{len(ids)} queries have no text "
                f"for this facet — excluded from this view.")
        ids = kept_ids
        if not ids:
            log(f"    {desc.strip()}: nothing to embed — view skipped.")
            return
    ds = TextDataset(ids, text_lookup, tokenizer, max_length=MAX_LEN)
    dl = DataLoader(ds, batch_size=BATCH_SIZE_EMB, shuffle=False,
                    num_workers=4, pin_memory=True)
    vecs, kept = [], []
    model.eval()
    for batch in tqdm(dl, desc=desc):
        emb = model(batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE))
        vecs.append(emb.cpu())
        kept.extend(batch["paper_id"])
    embeddings = torch.cat(vecs, dim=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embeddings": embeddings, "paper_ids": kept}, out_path)
    if map_path is not None:
        with open(map_path, "w") as f:
            json.dump({pid: i for i, pid in enumerate(kept)}, f)
    log(f"    Saved {embeddings.shape} -> {out_path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", nargs="*", default=list(LABEL_TYPES))
    args = ap.parse_args()

    ensure_dirs()
    banner("Stage 2 — Embedding generation (pool + per-facet query views)")
    for line in describe_paths():
        log(f"  {line}")
    check_inputs([CANDIDATE_PARQUET, EVAL_PARQUET],
                 hint="Set DATA_DIR / DATA_VERSION in .env for this server.")
    log(f"Device: {DEVICE}")
    log(f"Backbone: {BACKBONE_MODEL_NAME} ({POOLING} pooling)")
    if SECONDARY_BACKBONE_DIR:
        log(f"Secondary view: {SECONDARY_BACKBONE_NAME} "
            f"<- {SECONDARY_BACKBONE_MODEL} ({SECONDARY_POOLING} pooling)")

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_NAME)

    cand_ids = paper_ids(CANDIDATE_PARQUET)
    eval_ids = paper_ids(EVAL_PARQUET)
    cand_text = load_paper_text_map(CANDIDATE_PARQUET)
    eval_text = load_paper_text_map(EVAL_PARQUET)
    log(f"  Candidate pool: {len(cand_ids)} | Eval queries: {len(eval_ids)}")

    if not FACETS_PARQUET.exists():
        raise SystemExit(f"Missing {FACETS_PARQUET}. Run 0-consolidate-facets.py first.")
    facet_map = load_facet_map(FACETS_PARQUET, text_map=eval_text)
    log(f"  Facet map covers {len(facet_map)} papers")

    for key in args.types:
        banner(f"{LABEL_TYPES[key]['name']}", char="-")
        model_path = BACKBONE_DIR / key / "final_model.pt"
        if not model_path.exists():
            log(f"  Backbone missing ({model_path}) — skipping this type.")
            continue

        model = ContrastiveEncoder(BACKBONE_MODEL_NAME, embed_dim=EMBED_DIM).to(DEVICE)
        state = torch.load(model_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state.get("model_state_dict", state))

        out_dir = EMBEDDINGS_DIR / key
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- document side: undecomposed pool ---
        embed(cand_ids, cand_text, model, tokenizer,
              out_dir / "candidates_embeddings.pt",
              out_dir / "candidates_paper_id_to_index.json",
              desc="    Candidates")

        # --- query side: one embedding set per dense view ---
        # Each file stores its own paper_ids list, so a view that covers only a
        # subset of queries (missing facets) stays perfectly aligned downstream.
        for view in DENSE_VIEWS:
            lookup = (lambda pid, v=view: facet_map.get(pid, {}).get(v, ""))
            embed(eval_ids, lookup, model, tokenizer,
                  out_dir / f"query_{view}_embeddings.pt", None,
                  desc=f"    Query view: {view}", skip_empty=True)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- secondary encoder: one extra 'full' view (recall specialist) ---
        if SECONDARY_BACKBONE_DIR:
            sec_ckpt = Path(SECONDARY_BACKBONE_DIR)
            if sec_ckpt.is_dir():
                sec_ckpt = sec_ckpt / key / "final_model.pt"
            if not sec_ckpt.exists():
                log(f"  Secondary backbone missing ({sec_ckpt}) — skipping "
                    f"the {SECONDARY_BACKBONE_NAME} view for this type.")
                continue
            log(f"  Secondary encoder: {SECONDARY_BACKBONE_MODEL} "
                f"({SECONDARY_POOLING} pooling) from {sec_ckpt}")
            sec_tok = AutoTokenizer.from_pretrained(SECONDARY_BACKBONE_MODEL)
            sec = ContrastiveEncoder(SECONDARY_BACKBONE_MODEL,
                                     embed_dim=SECONDARY_EMBED_DIM,
                                     pooling=SECONDARY_POOLING).to(DEVICE)
            st = torch.load(sec_ckpt, map_location=DEVICE, weights_only=False)
            sec.load_state_dict(st.get("model_state_dict", st))
            tag = SECONDARY_BACKBONE_NAME
            embed(cand_ids, cand_text, sec, sec_tok,
                  out_dir / f"sec_{tag}_candidates_embeddings.pt", None,
                  desc=f"    Candidates [{tag}]")
            embed(eval_ids, (lambda pid: facet_map.get(pid, {}).get(FULL_VIEW, "")),
                  sec, sec_tok, out_dir / f"sec_{tag}_query_full_embeddings.pt", None,
                  desc=f"    Query full [{tag}]", skip_empty=True)
            del sec
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    banner(f"Embeddings written to {EMBEDDINGS_DIR}")


if __name__ == "__main__":
    main()
