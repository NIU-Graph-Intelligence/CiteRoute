import torch
import numpy as np
import polars as pl
import json
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

SEP = "[SEP]"
MAX_LENGTH = 512
BATCH_SIZE = 128

MODEL_NAME = "allenai/scibert_scivocab_uncased"

# ======================================================================
# Dataset for papers with title, abstract
# ======================================================================
class PaperDataset(Dataset):

    def __init__(self, df, tokenizer, max_length=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.row(idx, named=True)

        # Get basic fields
        title = (row.get("title", "") or "").strip()
        abstract = (row.get("abstract", "") or "").strip()
        paper_id = row.get("paper_id", "")

        # Combine all text: Title + Abstract
        combined_text = f"{title} {SEP} {abstract}"

        # Tokenize
        encoding = self.tokenizer(
            combined_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "paper_id": paper_id
        }


def generate_embeddings(parquet_path, output_name, tokenizer, model, device):
    """Generate embeddings for a dataset"""
    print(f"\n{'='*80}")
    print(f"Processing: {parquet_path}")
    print(f"{'='*80}")

    # Load data
    print("Loading parquet file...")
    df = pl.read_parquet(parquet_path)
    print(f"Loaded {len(df)} papers")

    # Create dataset and dataloader
    dataset = PaperDataset(df, tokenizer, max_length=MAX_LENGTH)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    # Generate embeddings
    all_embeddings = []
    all_paper_ids = []

    print("Generating embeddings...")
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            paper_ids = batch["paper_id"]

            # Get embeddings
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Use [CLS] token embedding
            cls_embeddings = outputs.last_hidden_state[:, 0, :] # one whole document into a single dimensional vector of 768dim

            all_embeddings.append(cls_embeddings.cpu())
            all_paper_ids.extend(list(paper_ids))
    
    # Concatenate all embeddings
    final_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"\nGenerated embeddings shape: {final_embeddings.shape}")
    print(f"Total papers: {len(all_paper_ids)}")

    # Create paper_id to index mapping
    paper_id_to_idx = {paper_id: idx for idx, paper_id in enumerate(all_paper_ids)}

    # Save embeddings with metadata
    output_data = {
        "embeddings": final_embeddings,
        "paper_ids": all_paper_ids,
        "model_name": MODEL_NAME,
        "embedding_dim": final_embeddings.shape[1],
        "max_length": MAX_LENGTH,
    }

    return final_embeddings, all_paper_ids, paper_id_to_idx, output_data


# ======================================================================
# Helper Functions for Metrics
# ======================================================================

def average_precision(retrieved_ids, ground_truth_ids):
    """Calculates Average Precision for a single query."""
    if not ground_truth_ids:
        return 0.0

    score = 0.0
    num_hits = 0.0
    for i, item_id in enumerate(retrieved_ids):
        if item_id in ground_truth_ids:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / len(ground_truth_ids)


def recall_at_k(retrieved_ids, ground_truth_ids, k):
    """Calculates Recall@k for a single query."""
    num_retrieved = len(set(retrieved_ids[:k]) & set(ground_truth_ids))
    return num_retrieved / len(ground_truth_ids) if ground_truth_ids else 0.0


def precision_at_k(retrieved_ids, ground_truth_ids, k):
    """Calculates Precision@k for a single query."""
    num_retrieved = len(set(retrieved_ids[:k]) & set(ground_truth_ids))
    return num_retrieved / k if k > 0 else 0.0


def hit_rate_at_k(retrieved_ids, ground_truth_ids, k):
    """Calculates Hit Rate@k (whether at least one relevant item is in top-k)."""
    top_k = set(retrieved_ids[:k])
    return 1.0 if any(item in top_k for item in ground_truth_ids) else 0.0


def reciprocal_rank(retrieved_ids, ground_truth_ids):
    """Calculates Reciprocal Rank for a single query."""
    for i, item_id in enumerate(retrieved_ids):
        if item_id in ground_truth_ids:
            return 1.0 / (i + 1.0)
    return 0.0


def dcg_at_k(retrieved_ids, ground_truth_ids, k):
    """Calculates Discounted Cumulative Gain at k."""
    dcg = 0.0
    for i, item_id in enumerate(retrieved_ids[:k]):
        if item_id in ground_truth_ids:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because index starts at 0
    return dcg


def ndcg_at_k(retrieved_ids, ground_truth_ids, k):
    """Calculates Normalized Discounted Cumulative Gain at k."""
    if not ground_truth_ids:
        return 0.0

    dcg = dcg_at_k(retrieved_ids, ground_truth_ids, k)

    # Ideal DCG: all relevant items at top positions
    ideal_retrieved = list(ground_truth_ids) + [0] * k
    idcg = dcg_at_k(ideal_retrieved, ground_truth_ids, k)

    return dcg / idcg if idcg > 0 else 0.0


# ======================================================================
# LOAD PAPER TITLE MAP
# ======================================================================
def load_paper_title_map(papers_parquet_path):
    """Load paper ID to title mapping from papers.parquet"""
    try:
        papers_df = pl.read_parquet(papers_parquet_path)
        paper_map = {}
        for row in papers_df.iter_rows(named=True):
            paper_map[row["paper_id"]] = row.get("title", "")
        return paper_map
    except Exception as e:
        print(f"Warning: Could not load papers.parquet: {e}")
        return {}