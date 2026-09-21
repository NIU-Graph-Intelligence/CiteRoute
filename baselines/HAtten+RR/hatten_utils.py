import os
import time
import json
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# ======================================================================
# Constants
# ======================================================================
GLOVE_DIM = 200
HIDDEN_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 1
BATCH_SIZE_FT = 64
BATCH_SIZE_EMB = 32
EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-5
MARGIN = 0.1
NITER_CHECKPOINT = 5000
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


def load_paper_title_map(parquet_file):
    try:
        df = pl.read_parquet(parquet_file)
        return {row["paper_id"]: row.get("title", "") for row in df.iter_rows(named=True)}
    except Exception as e:
        print(f"Warning: {e}")
        return {}


# ======================================================================
# Simple Tokenizer
# ======================================================================

def simple_tokenize(text):
    return text.lower().split()


# ======================================================================
# GloVe Loading
# ======================================================================

def load_glove_embeddings(glove_path=None, vocab=None, dim=GLOVE_DIM):
    if glove_path and os.path.exists(glove_path):
        log(f"Loading GloVe from {glove_path}")
        word2idx = {"<PAD>": 0, "<UNK>": 1}
        embeddings = [np.zeros(dim), np.random.randn(dim) * 0.1]
        with open(glove_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == dim + 1:
                    word = parts[0]
                    vec = np.array([float(x) for x in parts[1:]])
                    word2idx[word] = len(embeddings)
                    embeddings.append(vec)
        embeddings = np.array(embeddings, dtype=np.float32)
        log(f"  Loaded {len(word2idx)} words")
        return embeddings, word2idx
    else:
        log("GloVe not found. Using random initialization.")
        word2idx = {"<PAD>": 0, "<UNK>": 1}
        if vocab:
            for word in vocab:
                if word not in word2idx:
                    word2idx[word] = len(word2idx)
        vocab_size = max(10000, len(word2idx))
        embeddings = np.random.randn(vocab_size, dim).astype(np.float32) * 0.1
        embeddings[0] = 0
        return embeddings, word2idx


# ======================================================================
# Multi-Head Pooling
# ======================================================================

class MultiHeadPooling(nn.Module):
    def __init__(self, input_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.value_layers = nn.ModuleList([nn.Linear(input_dim, self.head_dim) for _ in range(num_heads)])
        self.attention_layers = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(num_heads)])
        self.output_layer = nn.Linear(input_dim, input_dim)

    def forward(self, x, mask=None):
        head_outputs = []
        for i in range(self.num_heads):
            values = self.value_layers[i](x)
            scores = self.attention_layers[i](x).squeeze(-1)
            if mask is not None:
                scores = scores.masked_fill(~mask, -1e9)
            attn_weights = F.softmax(scores, dim=-1).unsqueeze(-1)
            head_out = (values * attn_weights).sum(dim=1)
            head_outputs.append(head_out)
        concat = torch.cat(head_outputs, dim=-1)
        return self.output_layer(F.relu(concat))


# ======================================================================
# Paragraph Encoder
# ======================================================================

class ParagraphEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_heads,
                 num_layers=1, pretrained_embeddings=None, freeze_embeddings=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False
        self.register_buffer('pos_encoding', self._create_positional_encoding(512, embedding_dim))
        self.input_proj = nn.Linear(embedding_dim, hidden_dim) if embedding_dim != hidden_dim else nn.Identity()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=0.1, activation='relu', batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooling = MultiHeadPooling(hidden_dim, num_heads)

    def _create_positional_encoding(self, max_len, d_model):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, input_ids, mask=None):
        x = self.embedding(input_ids)
        seq_len = x.size(1)
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
        x = self.input_proj(x)
        attn_mask = ~mask if mask is not None else None
        x = self.transformer(x, src_key_padding_mask=attn_mask) if attn_mask is not None else self.transformer(x)
        return self.pooling(x, mask=mask)


# ======================================================================
# Document Encoder
# ======================================================================

class DocumentEncoder(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_paragraph_types=3, num_layers=1):
        super().__init__()
        self.type_embedding = nn.Embedding(num_paragraph_types, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=0.1, activation='relu', batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooling = MultiHeadPooling(hidden_dim, num_heads)

    def forward(self, paragraph_embeddings, paragraph_types, mask=None):
        x = paragraph_embeddings + self.type_embedding(paragraph_types)
        attn_mask = ~mask if mask is not None else None
        x = self.transformer(x, src_key_padding_mask=attn_mask) if attn_mask is not None else self.transformer(x)
        return self.pooling(x, mask=mask)


# ======================================================================
# HAtten Model
# ======================================================================

class HAttenModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_heads,
                 pretrained_embeddings=None, freeze_embeddings=True):
        super().__init__()
        self.paragraph_encoder = ParagraphEncoder(
            vocab_size, embedding_dim, hidden_dim, num_heads,
            pretrained_embeddings=pretrained_embeddings, freeze_embeddings=freeze_embeddings,
        )
        self.document_encoder = DocumentEncoder(hidden_dim, num_heads, num_paragraph_types=3)
        self.hidden_dim = hidden_dim
        self.word2idx = None

    def set_word2idx(self, word2idx):
        self.word2idx = word2idx

    def encode_paragraphs(self, paragraphs_list, max_tokens=200):
        device = next(self.parameters()).device
        batch_size = len(paragraphs_list)
        max_paragraphs = max(len(p) for p in paragraphs_list)

        all_paragraph_embs = []
        paragraph_mask = torch.zeros(batch_size, max_paragraphs, dtype=torch.bool, device=device)

        for i, paragraphs in enumerate(paragraphs_list):
            paragraph_embs = []
            for j, para in enumerate(paragraphs):
                tokens = simple_tokenize(para)[:max_tokens]
                token_ids = [self.word2idx.get(w, 1) for w in tokens]
                if len(token_ids) < max_tokens:
                    token_ids = token_ids + [0] * (max_tokens - len(token_ids))

                token_ids_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
                token_mask = torch.tensor(
                    [[1] * len(tokens) + [0] * (max_tokens - len(tokens))],
                    dtype=torch.bool, device=device,
                )
                para_emb = self.paragraph_encoder(token_ids_tensor, token_mask)
                paragraph_embs.append(para_emb)
                paragraph_mask[i, j] = True

            while len(paragraph_embs) < max_paragraphs:
                paragraph_embs.append(torch.zeros(1, self.hidden_dim, device=device))
            all_paragraph_embs.append(torch.cat(paragraph_embs, dim=0))

        return torch.stack(all_paragraph_embs), paragraph_mask

    def forward(self, paragraphs_batch, types_batch):
        device = next(self.parameters()).device
        paragraph_embeddings, paragraph_mask = self.encode_paragraphs(paragraphs_batch)
        if not isinstance(types_batch, torch.Tensor):
            types_batch = torch.tensor(types_batch, dtype=torch.long, device=device)
        else:
            types_batch = types_batch.to(device)
        return self.document_encoder(paragraph_embeddings, types_batch, paragraph_mask)


# ======================================================================
# Triplet Loss
# ======================================================================

def triplet_loss(anchor, positive, negative, margin=MARGIN):
    anchor = F.normalize(anchor, p=2, dim=-1)
    positive = F.normalize(positive, p=2, dim=-1)
    negative = F.normalize(negative, p=2, dim=-1)
    pos_sim = (anchor * positive).sum(dim=-1)
    neg_sim = (anchor * negative).sum(dim=-1)
    return F.relu(neg_sim - pos_sim + margin).mean()


# ======================================================================
# Dataset: triplets for HAtten
# ======================================================================

class HAttenTripletDataset(Dataset):
    def __init__(self, parquet_file, paper_text_map, label_type_key):
        cfg = LABEL_TYPES[label_type_key]
        field = cfg["field"]
        threshold_fn = cfg["threshold_fn"]

        df = pl.read_parquet(parquet_file)
        self.queries = []

        for row in df.iter_rows(named=True):
            paper_id = row["paper_id"]
            title = (row.get("title", "") or "").strip()
            abstract = (row.get("abstract", "") or "").strip()
            if not title and not abstract:
                continue

            references = json.loads(row["references"]) if isinstance(row["references"], str) else row["references"]

            positive_refs = []
            for ref in references:
                matched_paper_id = ref.get("matched_paper_id")
                if not matched_paper_id:
                    continue
                raw_val = ref.get(field)
                if raw_val is None:
                    continue
                try:
                    if threshold_fn(raw_val) and matched_paper_id in paper_text_map:
                        positive_refs.append(matched_paper_id)
                except (ValueError, TypeError):
                    continue

            for pos_ref in positive_refs:
                self.queries.append({
                    "query_title": title,
                    "query_abstract": abstract,
                    "pos_ref": pos_ref,
                })

        log(f"    {cfg['name']}: {len(self.queries)} query-positive pairs")

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx]


def hatten_collate_fn(batch, paper_text_map):
    """Collate function that splits text into title/abstract paragraphs."""
    query_paragraphs = []
    query_types = []
    pos_paragraphs = []
    pos_types = []

    for item in batch:
        query_paragraphs.append([item["query_title"], item["query_abstract"]])
        query_types.append([0, 1])

        pos_text = paper_text_map.get(item["pos_ref"], " ")
        parts = pos_text.split(f" {SEP} ", 1)
        pos_title = parts[0] if len(parts) > 0 else ""
        pos_abstract = parts[1] if len(parts) > 1 else ""
        pos_paragraphs.append([pos_title, pos_abstract])
        pos_types.append([0, 1])

    query_types_t = torch.tensor(query_types, dtype=torch.long)
    pos_types_t = torch.tensor(pos_types, dtype=torch.long)

    return {
        "query_paragraphs": query_paragraphs,
        "query_types": query_types_t,
        "pos_paragraphs": pos_paragraphs,
        "pos_types": pos_types_t,
    }


# ======================================================================
# Ground Truth & Metrics
# ======================================================================

def extract_relevant_sets(references):
    relevant = {key: set() for key in LABEL_TYPES}
    for ref in references:
        matched_paper_id = ref.get("matched_paper_id")
        if not matched_paper_id: continue
        for key, cfg in LABEL_TYPES.items():
            raw_val = ref.get(cfg["field"])
            if raw_val is None: continue
            try:
                if cfg["threshold_fn"](raw_val):
                    relevant[key].add(matched_paper_id)
            except (ValueError, TypeError): continue
    return relevant


def compute_map(relevant, retrieved):
    if not relevant: return 0.0
    score, num_hits = 0.0, 0
    for i, pid in enumerate(retrieved):
        if pid in relevant:
            num_hits += 1; score += num_hits / (i + 1)
    return score / len(relevant)

def compute_mrr(relevant, retrieved):
    for i, pid in enumerate(retrieved):
        if pid in relevant: return 1.0 / (i + 1)
    return 0.0

def compute_recall_at_k(relevant, retrieved, k):
    if not relevant: return 0.0
    return sum(1 for pid in retrieved[:k] if pid in relevant) / len(relevant)

def compute_hit_rate_at_k(relevant, retrieved, k):
    return 1.0 if any(item in set(retrieved[:k]) for item in relevant) else 0.0

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
