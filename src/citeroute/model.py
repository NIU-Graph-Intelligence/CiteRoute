"""Contrastive dense backbone (Sec. 3.2) and NT-Xent loss.

Architecture matches the SciBERT-NTX baseline exactly (encoder + 2-layer
projection head, L2-normalised output), so the backbone can be initialised
either from scratch or from an existing SciBERT-NTX checkpoint.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .config import (BACKBONE_MODEL_NAME, EMBED_DIM, MAX_LEN, POOLING,
                     TEMPERATURE)


class ContrastiveEncoder(nn.Module):
    """Transformer encoder + projection head for contrastive retrieval.

    `pooling` must match how the base model was pretrained:
      * "cls"  — BERT / SciBERT family (first token)
      * "mean" — GTE / E5 / BGE family (attention-masked mean)
    Using CLS pooling on a mean-pooled model (or vice versa) silently degrades
    retrieval badly, so the default is auto-detected from the model name in
    config.default_pooling().
    """

    def __init__(self, model_name=BACKBONE_MODEL_NAME, embed_dim=EMBED_DIM,
                 pooling=None):
        super().__init__()
        from transformers import AutoModel
        self.model_name = model_name
        self.pooling = pooling or POOLING
        if self.pooling not in ("cls", "mean"):
            raise ValueError(f"Unknown pooling {self.pooling!r} (use 'cls' or 'mean')")
        # trust_remote_code is required by the GTE v1.5 architectures; harmless
        # for standard BERT-family checkpoints.
        try:
            self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        except TypeError:
            self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )

    def _pool(self, last_hidden_state, attention_mask):
        if self.pooling == "cls":
            return last_hidden_state[:, 0, :]
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(out.last_hidden_state, attention_mask)
        return nn.functional.normalize(self.projection(pooled), dim=-1)


class NTXentLoss(nn.Module):
    """Eq. (1): normalised temperature-scaled cross-entropy with in-batch negatives."""

    def __init__(self, temperature=TEMPERATURE):
        super().__init__()
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, emb1, emb2):
        B = emb1.size(0)
        Z = torch.cat([emb1, emb2], dim=0)
        sim = torch.matmul(Z, Z.T) / self.temperature
        sim = sim.masked_fill(torch.eye(2 * B, dtype=torch.bool, device=Z.device), -1e9)
        labels = torch.arange(B, device=Z.device)
        labels = torch.cat([labels + B, labels], dim=0)
        return self.ce(sim, labels)


class PositivePairsDataset(Dataset):
    """(query_text, must_cite_text) pairs for one label type."""

    def __init__(self, train_parquet, text_map, label_type_key):
        from .config import LABEL_TYPES
        from .data import parse_references
        import polars as pl

        cfg = LABEL_TYPES[label_type_key]
        field, ok = cfg["field"], cfg["threshold_fn"]
        df = pl.read_parquet(train_parquet)
        self.pairs = []
        for row in df.iter_rows(named=True):
            q_text = text_map.get(row["paper_id"], "")
            if not q_text.strip():
                continue
            for ref in parse_references(row):
                mid = ref.get("matched_paper_id")
                raw = ref.get(field)
                if not mid or raw is None:
                    continue
                try:
                    if ok(raw):
                        r_text = text_map.get(mid)
                        if r_text and r_text.strip():
                            self.pairs.append((q_text, r_text))
                except (ValueError, TypeError):
                    continue

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


class TextDataset(Dataset):
    """Tokenised (paper_id, text) items for embedding generation."""

    def __init__(self, ids, text_lookup, tokenizer, max_length=MAX_LEN):
        self.ids = ids
        self.text_lookup = text_lookup
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        text = self.text_lookup(pid) if callable(self.text_lookup) else self.text_lookup.get(pid, "")
        enc = self.tokenizer(
            text or "",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "paper_id": pid,
        }
