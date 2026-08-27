"""
Stage 4 — Facet-routed retrieval (Sec. 3.4)
============================================
Each facet is issued to the retrieval paradigm matched to its signal type:

  entity facets  (datasets_benchmarks, key_terms, compares_against) -> BM25
  concept facets (task, method, builds_on, contribution_summary)    -> dense
  full           (undecomposed title+abstract)                      -> both

Produces one ranked list per (label type, split, view), written as a compact
NPZ run file that Stages 5-7 consume.

Input : Stage 2 embeddings, Stage 3 BM25 index, Stage 0 facets
Output: OUTPUT_DIR/runs/<type_key>/<split>_<view>.npz   (query_ids, doc_ids, ranks)

Splits:
  eval        — the benchmark queries (default)
  fusion      — a sample of TRAIN queries used to fit fusion weights (Stage 5)
  reranker    — a sample of TRAIN queries used to mine hard negatives (Stage 6)

Resume: existing run files are skipped.
"""

import argparse
import json

import numpy as np
import torch

from citeroute.config import (eval_parquet_for,  # noqa: F401
                             CANDIDATE_PARQUET, DENSE_VIEWS, EMBEDDINGS_DIR,
                             EVAL_PARQUET, FACETS_PARQUET, FULL_VIEW,
                             FUSION_TRAIN_QUERIES, LABEL_TYPES,
                             RERANKER_TRAIN_QUERIES, RETRIEVE_DEPTH, RUNS_DIR,
                             SEED, SPARSE_DIR, SPARSE_VIEWS, TRAIN_PARQUET,
                             SECONDARY_BACKBONE_DIR, SECONDARY_BACKBONE_NAME,
                             check_inputs, describe_paths, ensure_dirs,
                             sparse_run_name)
from citeroute.data import (build_ground_truth, load_facet_map,
                           load_paper_text_map, paper_ids)
from citeroute.sparse import BM25Index
from citeroute.utils import Timer, banner, log

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPLIT_SIZES = {"eval": None, "fusion": FUSION_TRAIN_QUERIES,
               "reranker": RERANKER_TRAIN_QUERIES}


def run_path(label_key, split, view):
    return RUNS_DIR / label_key / f"{split}_{view}.npz"


def save_run(path, query_ids, doc_id_matrix):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path,
                        query_ids=np.array(query_ids, dtype=object),
                        doc_ids=np.array(doc_id_matrix, dtype=object))


def load_run(path):
    d = np.load(path, allow_pickle=True)
    return list(d["query_ids"]), [list(r) for r in d["doc_ids"]]


def dense_search(query_vecs, pool_vecs, pool_ids, depth, desc):
    """Cosine top-k. Uses FAISS when available, else chunked torch matmul."""
    try:
        import faiss
        index = faiss.IndexFlatIP(pool_vecs.shape[1])
        index.add(np.ascontiguousarray(pool_vecs.astype("float32")))
        _, idx = index.search(np.ascontiguousarray(query_vecs.astype("float32")), depth)
        return [[pool_ids[j] for j in row] for row in idx]
    except ImportError:
        log(f"    (faiss unavailable — using torch matmul for {desc})")
        P = torch.from_numpy(pool_vecs).to(DEVICE)
        out = []
        step = 256
        for s in range(0, len(query_vecs), step):
            Q = torch.from_numpy(query_vecs[s:s + step]).to(DEVICE)
            sims = Q @ P.T
            idx = torch.topk(sims, k=min(depth, P.shape[0]), dim=1).indices.cpu().numpy()
            out.extend([[pool_ids[j] for j in row] for row in idx])
        return out


def l2norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def select_queries(split, label_key, pool_ids):
    """Query ids for a split; train splits are sampled and require gold labels.

    Type 4 may use its own eval slice (older queries that have followers), so
    the eval dataframe is resolved per label type.
    """
    if split == "eval":
        return paper_ids(eval_parquet_for(label_key))
    gt, _ = build_ground_truth(TRAIN_PARQUET, label_key, pool_ids=pool_ids)
    ids = sorted(gt.keys())
    n = SPLIT_SIZES[split]
    if n and len(ids) > n:
        rng = np.random.default_rng(SEED)
        ids = [ids[i] for i in rng.choice(len(ids), size=n, replace=False)]
    return sorted(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", nargs="*", default=list(LABEL_TYPES))
    ap.add_argument("--splits", nargs="*", default=["eval", "fusion", "reranker"])
    ap.add_argument("--depth", type=int, default=RETRIEVE_DEPTH)
    args = ap.parse_args()

    ensure_dirs()
    banner("Stage 4 — Facet-routed retrieval")
    for line in describe_paths():
        log(f"  {line}")
    check_inputs([CANDIDATE_PARQUET, EVAL_PARQUET, TRAIN_PARQUET, FACETS_PARQUET],
                 hint="Run Stage 0 first, and check DATA_DIR in .env.")
    log(f"Depth per view: {args.depth}")

    pool_ids_list = paper_ids(CANDIDATE_PARQUET)
    pool_ids_set = set(pool_ids_list)
    log(f"  Candidate pool: {len(pool_ids_list)}")

    # Facet text for every query paper we may touch (eval + train)
    text_map = {**load_paper_text_map(EVAL_PARQUET), **load_paper_text_map(TRAIN_PARQUET)}
    facet_map = load_facet_map(FACETS_PARQUET, text_map=text_map)

    log("  Loading BM25 index...")
    bm25 = BM25Index.load(SPARSE_DIR / "bm25_index.pkl")

    for label_key in args.types:
        banner(f"{LABEL_TYPES[label_key]['name']}", char="-")
        emb_dir = EMBEDDINGS_DIR / label_key
        cand_path = emb_dir / "candidates_embeddings.pt"
        dense_ok = cand_path.exists()
        pool_vecs = pool_ids = None
        if dense_ok:
            cand = torch.load(cand_path, map_location="cpu", weights_only=False)
            pool_vecs = l2norm(cand["embeddings"].numpy())
            pool_ids = cand["paper_ids"]
        else:
            log(f"  Missing {cand_path} — dense views skipped "
                f"(sparse views still run; run Stage 2 for the full pipeline).")

        for split in args.splits:
            query_ids = select_queries(split, label_key, pool_ids_set)
            if not query_ids:
                log(f"  [{split}] no queries — skipping.")
                continue
            log(f"  [{split}] {len(query_ids)} queries")

            # ---------- dense views ----------
            query_set = set(query_ids)
            for view in (DENSE_VIEWS if dense_ok else []):
                path = run_path(label_key, split, view)
                if path.exists():
                    log(f"    {path.name} exists — skipping.")
                    continue
                vec_path = emb_dir / f"query_{view}_embeddings.pt"
                if split == "eval" and vec_path.exists():
                    # Each embedding file carries its own paper_ids (queries with
                    # a missing facet were excluded in Stage 2), so index by that
                    # list rather than a global map.
                    data = torch.load(vec_path, map_location="cpu", weights_only=False)
                    all_vecs = l2norm(data["embeddings"].numpy())
                    emb_ids = list(data["paper_ids"])
                    keep = [i for i, q in enumerate(emb_ids) if q in query_set]
                    used = [emb_ids[i] for i in keep]
                    qv = all_vecs[keep]
                    if len(used) < len(query_ids):
                        log(f"    {view}: {len(query_ids) - len(used)} queries "
                            f"lack this facet — omitted from this view.")
                else:
                    # Train-split views are embedded on the fly (small samples).
                    try:
                        qv, used = embed_on_the_fly(label_key, query_ids, facet_map, view)
                    except Exception as e:  # noqa: BLE001
                        # One unavailable model/view must not abort a long run;
                        # the missing view is simply absent from fusion.
                        log(f"    {view}: on-the-fly embedding failed ({e}) — view skipped.")
                        continue
                if len(used) == 0:
                    log(f"    {view}: no embeddable queries — skipping.")
                    continue
                with Timer(f"    dense/{view}"):
                    ranked = dense_search(qv, pool_vecs, pool_ids, args.depth, view)
                save_run(path, used, ranked)

            # ---------- secondary encoder view ----------
            if SECONDARY_BACKBONE_DIR and split == "eval":
                tag = SECONDARY_BACKBONE_NAME
                path = run_path(label_key, split, f"{FULL_VIEW}@{tag}")
                sec_cand = emb_dir / f"sec_{tag}_candidates_embeddings.pt"
                sec_q = emb_dir / f"sec_{tag}_query_full_embeddings.pt"
                if path.exists():
                    log(f"    {path.name} exists — skipping.")
                elif sec_cand.exists() and sec_q.exists():
                    c = torch.load(sec_cand, map_location="cpu", weights_only=False)
                    q = torch.load(sec_q, map_location="cpu", weights_only=False)
                    sec_pool = l2norm(c["embeddings"].numpy())
                    sec_ids = list(q["paper_ids"])
                    keep = [i for i, x in enumerate(sec_ids) if x in query_set]
                    if keep:
                        with Timer(f"    dense/{FULL_VIEW}@{tag}"):
                            ranked = dense_search(l2norm(q["embeddings"].numpy())[keep],
                                                  sec_pool, c["paper_ids"],
                                                  args.depth, f"{FULL_VIEW}@{tag}")
                        save_run(path, [sec_ids[i] for i in keep], ranked)
                else:
                    log(f"    secondary view {tag}: embeddings not found — skipped.")

            # ---------- sparse views ----------
            for view in SPARSE_VIEWS:
                path = run_path(label_key, split, sparse_run_name(view))
                if path.exists():
                    log(f"    {path.name} exists — skipping.")
                    continue
                ranked, used = [], []
                with Timer(f"    sparse/{view}"):
                    for qid in query_ids:
                        qtext = facet_map.get(qid, {}).get(view, "")
                        if not qtext.strip():
                            continue
                        docs, _ = bm25.search(qtext, args.depth)
                        if docs:
                            ranked.append(docs)
                            used.append(qid)
                if used:
                    save_run(path, used, ranked)
                else:
                    log(f"    {view}: no non-empty queries — skipped.")

        # Release the cached on-the-fly encoder before the next label type.
        _ENCODER_CACHE.pop(label_key, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    banner(f"Runs written to {RUNS_DIR}")


_ENCODER_CACHE = {}


def _get_encoder(label_key):
    """Load (and cache) the backbone for on-the-fly embedding.

    Cached per label type: without this the model would be reloaded once per
    view per split — five needless loads of a 110M-parameter encoder.
    """
    if label_key in _ENCODER_CACHE:
        return _ENCODER_CACHE[label_key]
    from transformers import AutoTokenizer

    from citeroute.config import (BACKBONE_DIR, BACKBONE_MODEL_NAME, EMBED_DIM,
                                 POOLING)
    from citeroute.model import ContrastiveEncoder

    model_path = BACKBONE_DIR / label_key / "final_model.pt"
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_NAME)
    model = ContrastiveEncoder(BACKBONE_MODEL_NAME, embed_dim=EMBED_DIM,
                               pooling=POOLING).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()
    _ENCODER_CACHE[label_key] = (model, tokenizer)
    return model, tokenizer


def embed_on_the_fly(label_key, query_ids, facet_map, view):
    """Embed a train-split view with the label type's backbone."""
    from citeroute.config import BATCH_SIZE_EMB, EMBED_DIM, MAX_LEN

    model, tokenizer = _get_encoder(label_key)

    texts, used = [], []
    for qid in query_ids:
        t = facet_map.get(qid, {}).get(view, "")
        if t.strip():
            texts.append(t)
            used.append(qid)
    vecs = []
    with torch.no_grad():
        for s in range(0, len(texts), BATCH_SIZE_EMB):
            enc = tokenizer(texts[s:s + BATCH_SIZE_EMB], padding="max_length",
                            truncation=True, max_length=MAX_LEN,
                            return_tensors="pt").to(DEVICE)
            vecs.append(model(enc.input_ids, enc.attention_mask).cpu().numpy())
    # NOTE: the encoder stays cached in _ENCODER_CACHE for the remaining views;
    # it is released in main() once the label type is finished.
    return (l2norm(np.vstack(vecs)) if vecs else np.zeros((0, EMBED_DIM), dtype=np.float32)), used


if __name__ == "__main__":
    main()
