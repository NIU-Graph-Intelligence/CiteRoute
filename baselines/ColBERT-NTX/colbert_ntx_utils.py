import os
import sys
import time
import json
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

# ======================================================================
# Constants
# ======================================================================
CHECKPOINT = "colbert-ir/colbertv2.0"
DOC_MAXLEN = 180
NBITS = 2
TEMPERATURE = 0.07
BATCH_SIZE_FT = 16
EPOCHS = 2
LR = 1e-5
SEP = "[SEP]"

TYPE_2_THRESHOLD = 4.0
TYPE_3_THRESHOLD = 3.0

LABEL_TYPES = {
    "type_1": {
        "name": "Type 1 (Binary Relevance)",
        "field": "type_1_output",
        "threshold_fn": lambda v: float(v) == 1.0,
        "description": "binary, label == 1",
    },
    "type_2": {
        "name": "Type 2 (Usefulness)",
        "field": "type_2_output",
        "threshold_fn": lambda v: float(v) >= TYPE_2_THRESHOLD,
        "description": f"usefulness >= {TYPE_2_THRESHOLD}",
    },
    "type_3": {
        "name": "Type 3 (Relatedness)",
        "field": "type_3_output",
        "threshold_fn": lambda v: float(v) >= TYPE_3_THRESHOLD,
        "description": f"relatedness >= {TYPE_3_THRESHOLD}",
    },
}

EVAL_K_VALUES = {
    "recall": [10, 50, 100, 500],
    "ndcg": [10, 20, 30, 50],
    "hr": [10, 20],
}


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ======================================================================
# Text Loading
# ======================================================================
def load_paper_text_map(parquet_file):
    """Build mapping: paper_id (str) -> 'title [SEP] abstract'"""
    df = pl.read_parquet(parquet_file)
    paper_map = {}
    for row in df.iter_rows(named=True):
        paper_id = row["paper_id"]
        title = (row.get("title", "") or "").strip()
        abstract = (row.get("abstract", "") or "").strip()
        paper_map[paper_id] = f"{title} {SEP} {abstract}"
    return paper_map


# ======================================================================
# ColBERT Aggregation for NT-Xent
# ======================================================================
def colbert_aggregate(token_embeddings):
    """
    Mean pool ColBERT token embeddings to single doc vector.
    Excludes zero-vector tokens (padding) to prevent dilution.
    ColBERT masks padded tokens to zero, so we detect them by norm.
    """
    # token_embeddings: [batch, num_tokens, dim]
    # Detect non-padding: tokens with non-zero norm
    token_norms = token_embeddings.norm(dim=-1, keepdim=True)  # [B, T, 1]
    mask = (token_norms > 1e-8).float()                         # [B, T, 1]
    # Sum non-padding tokens and divide by count
    summed = (token_embeddings * mask).sum(dim=1)               # [B, dim]
    counts = mask.sum(dim=1).clamp(min=1.0)                     # [B, 1]
    aggregated = summed / counts
    aggregated = F.normalize(aggregated, p=2, dim=-1)
    return aggregated


class ColBERTNTXentLoss(nn.Module):
    def __init__(self, temperature=TEMPERATURE):
        super().__init__()
        # Use a learnable log-temperature to avoid collapse.
        # Initialize at log(1/temperature) so exp() gives ~1/0.07 ≈ 14.3
        # but clamp it to prevent extreme scaling.
        self.log_temperature = nn.Parameter(torch.tensor(np.log(1.0 / temperature)))
        self.ce = nn.CrossEntropyLoss()

    def forward(self, emb1, emb2):
        emb1 = emb1.float()
        emb2 = emb2.float()
        doc_emb1 = colbert_aggregate(emb1)
        doc_emb2 = colbert_aggregate(emb2)

        # Check for degenerate embeddings (all-zero after aggregation).
        # Return None so the caller skips the batch: a constant tensor here
        # is disconnected from the model, produces no gradients, and makes
        # scaler.step() raise "No inf checks were recorded".
        if doc_emb1.norm(dim=-1).min() < 1e-6 or doc_emb2.norm(dim=-1).min() < 1e-6:
            return None

        B = doc_emb1.size(0)
        Z = torch.cat([doc_emb1, doc_emb2], dim=0)
        # Clamp temperature scale to [1, 100] to prevent collapse
        temp_scale = self.log_temperature.exp().clamp(min=1.0, max=100.0)
        sim = torch.matmul(Z, Z.T) * temp_scale
        mask = torch.eye(2 * B, dtype=torch.bool, device=Z.device)
        sim = sim.masked_fill(mask, -1e9)
        labels = torch.arange(B, device=Z.device)
        labels = torch.cat([labels + B, labels], dim=0)
        return self.ce(sim, labels)


# ======================================================================
# Dataset: positive pairs for a specific label type
# ======================================================================
class ColBERTPositivePairsDataset(torch.utils.data.Dataset):
    def __init__(self, parquet_file, paper_text_map, label_type_key):
        cfg = LABEL_TYPES[label_type_key]
        field = cfg["field"]
        threshold_fn = cfg["threshold_fn"]
        df = pl.read_parquet(parquet_file)
        self.pairs = []
        for row in df.iter_rows(named=True):
            paper_id = row["paper_id"]
            paper_text = paper_text_map.get(paper_id, "")
            if not paper_text.strip():
                continue
            references = json.loads(row["references"]) if isinstance(row["references"], str) else row["references"]
            for ref in references:
                matched_paper_id = ref.get("matched_paper_id")
                if not matched_paper_id:
                    continue
                raw_val = ref.get(field)
                if raw_val is None:
                    continue
                try:
                    if threshold_fn(raw_val):
                        ref_text = paper_text_map.get(matched_paper_id)
                        if ref_text and ref_text.strip():
                            self.pairs.append((paper_text, ref_text))
                except (ValueError, TypeError):
                    continue
        log(f"    {cfg['name']}: {len(self.pairs)} positive pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


# ======================================================================
# Chunking for index building
# ======================================================================
def chunk_by_wordpieces(text, tokenizer, maxlen):
    toks = tokenizer.encode(text, add_special_tokens=False)
    for i in range(0, len(toks), maxlen):
        s = tokenizer.decode(toks[i:i+maxlen], skip_special_tokens=True).strip()
        if s:
            yield s


def load_pid_to_paperid_map(npy_path):
    if os.path.exists(npy_path):
        return np.load(npy_path, allow_pickle=True)
    else:
        raise FileNotFoundError(f"Missing {npy_path}")


def retrieve_suggestions(query_text, searcher, pid_to_paperid, k=500, query_id=None):
    try:
        pids, ranks, scores = searcher.search(query_text, k=k)
        pids = pids.tolist() if isinstance(pids, np.ndarray) else pids
    except Exception as e:
        print(f"Error during search for query_id {query_id}: {e}")
        return []
    suggestions = []
    seen = set()
    for pid in pids:
        pid = int(pid)
        if pid < 0 or pid >= len(pid_to_paperid):
            continue
        paper_id = str(pid_to_paperid[pid])
        if query_id and paper_id == query_id:
            continue
        if paper_id in seen:
            continue
        seen.add(paper_id)
        suggestions.append(paper_id)
        if len(suggestions) >= k:
            break
    return suggestions


# ======================================================================
# Ground Truth & Metrics
# ======================================================================
def extract_relevant_sets(references):
    relevant = {key: set() for key in LABEL_TYPES}
    for ref in references:
        matched_paper_id = ref.get("matched_paper_id")
        if not matched_paper_id:
            continue
        for key, cfg in LABEL_TYPES.items():
            raw_val = ref.get(cfg["field"])
            if raw_val is None:
                continue
            try:
                if cfg["threshold_fn"](raw_val):
                    relevant[key].add(matched_paper_id)
            except (ValueError, TypeError):
                continue
    return relevant


def compute_map(relevant, retrieved):
    if not relevant: return 0.0
    score, num_hits = 0.0, 0
    for i, pid in enumerate(retrieved):
        if pid in relevant:
            num_hits += 1
            score += num_hits / (i + 1)
    return score / len(relevant)

def compute_mrr(relevant, retrieved):
    for i, pid in enumerate(retrieved):
        if pid in relevant: return 1.0 / (i + 1)
    return 0.0

def compute_recall_at_k(relevant, retrieved, k):
    if not relevant: return 0.0
    return sum(1 for pid in retrieved[:k] if pid in relevant) / len(relevant)

def compute_hit_rate_at_k(relevant, retrieved, k):
    top_k = set(retrieved[:k])
    return 1.0 if any(item in top_k for item in relevant) else 0.0

def compute_dcg_at_k(relevant, retrieved, k):
    dcg = 0.0
    for i, pid in enumerate(retrieved[:k]):
        if pid in relevant: dcg += 1.0 / np.log2(i + 2)
    return dcg

def compute_ndcg_at_k(relevant, retrieved, k):
    if not relevant: return 0.0
    dcg = compute_dcg_at_k(relevant, retrieved, k)
    idcg = compute_dcg_at_k(relevant, list(relevant) + [0]*k, k)
    return dcg / idcg if idcg > 0 else 0.0

def compute_all_metrics(relevant, retrieved):
    return {
        "map": compute_map(relevant, retrieved),
        "mrr": compute_mrr(relevant, retrieved),
        "recall": {k: compute_recall_at_k(relevant, retrieved, k) for k in EVAL_K_VALUES["recall"]},
        "ndcg": {k: compute_ndcg_at_k(relevant, retrieved, k) for k in EVAL_K_VALUES["ndcg"]},
        "hr": {k: compute_hit_rate_at_k(relevant, retrieved, k) for k in EVAL_K_VALUES["hr"]},
    }