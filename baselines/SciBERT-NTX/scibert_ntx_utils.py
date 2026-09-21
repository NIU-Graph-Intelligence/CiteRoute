import os
import time
import json
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path


# ======================================================================
# Constants
# ======================================================================
MODEL_NAME = "allenai/scibert_scivocab_uncased"
EMBED_DIM = 128       # projection head output dim
MAX_LEN = 512
BATCH_SIZE_FT = 16    # fine-tuning batch size
BATCH_SIZE_EMB = 128  # embedding generation batch size
EPOCHS = 3
LR = 2e-5
TEMPERATURE = 0.07
SEP = "[SEP]"

# Label types & thresholds (consistent across all baselines)
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


def load_paper_title_map(parquet_file):
    """Load paper_id -> title mapping (for debug printing)."""
    try:
        df = pl.read_parquet(parquet_file)
        return {row["paper_id"]: row.get("title", "") for row in df.iter_rows(named=True)}
    except Exception as e:
        print(f"Warning: Could not load title map: {e}")
        return {}


# ======================================================================
# Model Definition
# ======================================================================

class ContrastiveSciBERT(nn.Module):
    """SciBERT with a 2-layer projection head for contrastive learning."""

    def __init__(self, model_name=MODEL_NAME, embed_dim=EMBED_DIM):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embed_dim),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        proj_emb = self.projection(cls_emb)
        proj_emb = nn.functional.normalize(proj_emb, dim=-1)
        return proj_emb


# ======================================================================
# NT-Xent (Normalized Temperature-scaled Cross-Entropy) Loss
# ======================================================================

class NTXentLoss(nn.Module):
    def __init__(self, temperature=TEMPERATURE):
        super().__init__()
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, emb1, emb2):
        # emb1, emb2: [B, d], already L2-normalized
        B = emb1.size(0)
        Z = torch.cat([emb1, emb2], dim=0)  # [2B, d]
        sim = torch.matmul(Z, Z.T) / self.temperature  # [2B, 2B]
        mask = torch.eye(2 * B, dtype=torch.bool, device=Z.device)
        sim = sim.masked_fill(mask, -1e9)  # remove self-similarity

        # Positive index for each row:
        # 0..B-1 -> i+B ;  B..2B-1 -> i-B
        labels = torch.arange(B, device=Z.device)
        labels = torch.cat([labels + B, labels], dim=0)

        loss = self.ce(sim, labels)
        return loss


# ======================================================================
# Fine-Tuning Dataset: positive pairs for a specific label type
# ======================================================================

class PaperPositivePairsDataset(Dataset):
    """
    Creates (anchor_text, positive_text) pairs for contrastive learning.
    Only includes references where the specified label type is positive.
    """

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
# Embedding Generation Dataset: one row per paper
# ======================================================================

class PaperTextDataset(Dataset):
    """Simple dataset: each item is (paper_id, text) for embedding generation."""

    def __init__(self, paper_ids, paper_text_map, tokenizer, max_length=MAX_LEN):
        self.paper_ids = paper_ids
        self.paper_text_map = paper_text_map
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.paper_ids)

    def __getitem__(self, idx):
        paper_id = self.paper_ids[idx]
        text = self.paper_text_map.get(paper_id, "")

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "paper_id": paper_id,
        }


# ======================================================================
# Ground Truth Extraction
# ======================================================================

def extract_relevant_sets(references):
    """Extract relevant paper_id sets for each label type from a reference list."""
    relevant = {key: set() for key in LABEL_TYPES}

    for ref in references:
        matched_paper_id = ref.get("matched_paper_id")
        if not matched_paper_id:
            continue

        for key, cfg in LABEL_TYPES.items():
            field = cfg["field"]
            raw_val = ref.get(field)
            if raw_val is None:
                continue
            try:
                if cfg["threshold_fn"](raw_val):
                    relevant[key].add(matched_paper_id)
            except (ValueError, TypeError):
                continue

    return relevant


# ======================================================================
# Metric Functions
# ======================================================================

def compute_map(relevant, retrieved):
    if not relevant:
        return 0.0
    score = 0.0
    num_hits = 0
    for i, paper_id in enumerate(retrieved):
        if paper_id in relevant:
            num_hits += 1
            score += num_hits / (i + 1)
    return score / len(relevant)


def compute_mrr(relevant, retrieved):
    for i, paper_id in enumerate(retrieved):
        if paper_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def compute_recall_at_k(relevant, retrieved, k):
    if not relevant:
        return 0.0
    hits = sum(1 for pid in retrieved[:k] if pid in relevant)
    return hits / len(relevant)


def compute_hit_rate_at_k(relevant, retrieved, k):
    top_k = set(retrieved[:k])
    return 1.0 if any(item in top_k for item in relevant) else 0.0


def compute_dcg_at_k(relevant, retrieved, k):
    dcg = 0.0
    for i, paper_id in enumerate(retrieved[:k]):
        if paper_id in relevant:
            dcg += 1.0 / np.log2(i + 2)
    return dcg


def compute_ndcg_at_k(relevant, retrieved, k):
    if not relevant:
        return 0.0
    dcg = compute_dcg_at_k(relevant, retrieved, k)
    ideal_retrieved = list(relevant) + [0] * k
    idcg = compute_dcg_at_k(relevant, ideal_retrieved, k)
    return dcg / idcg if idcg > 0 else 0.0


def compute_all_metrics(relevant, retrieved):
    return {
        "map": compute_map(relevant, retrieved),
        "mrr": compute_mrr(relevant, retrieved),
        "recall": {k: compute_recall_at_k(relevant, retrieved, k) for k in EVAL_K_VALUES["recall"]},
        "ndcg": {k: compute_ndcg_at_k(relevant, retrieved, k) for k in EVAL_K_VALUES["ndcg"]},
        "hr": {k: compute_hit_rate_at_k(relevant, retrieved, k) for k in EVAL_K_VALUES["hr"]},
    }