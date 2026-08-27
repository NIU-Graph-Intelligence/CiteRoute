"""Cross-encoder reranker (Eqs. 6-7).

Scores (query, candidate) jointly with full cross-attention -- too slow for
the whole pool, applied to the top-C fused candidates only.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .config import RERANKER_MAX_LEN, RERANKER_MODEL_NAME


class CrossEncoderReranker(nn.Module):
    """Transformer encoder + scalar relevance head."""

    def __init__(self, model_name=RERANKER_MODEL_NAME):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        cls = out.last_hidden_state[:, 0, :]
        return self.head(self.dropout(cls)).squeeze(-1)


def build_query_text(facet_entry, full_text):
    """Query side of the pair: raw text plus the facets (Eq. 6)."""
    from .config import DENSE_FACETS, SPARSE_FACETS
    parts = [full_text or ""]
    for name in DENSE_FACETS + SPARSE_FACETS:
        val = (facet_entry or {}).get(name, "")
        if val:
            parts.append(f"{name}: {val}")
    return " ".join(parts)


class PairDataset(Dataset):
    """(query_text, doc_text, label) triples tokenised as a sentence pair."""

    def __init__(self, pairs, tokenizer, max_length=RERANKER_MAX_LEN):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        q, d, y = self.pairs[idx]
        enc = self.tokenizer(
            q, d,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(float(y), dtype=torch.float),
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)
        return item
