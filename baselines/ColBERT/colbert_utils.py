import time
import numpy as np
from pathlib import Path


# ======================================================================
# Constants
# ======================================================================
CHECKPOINT = "colbert-ir/colbertv2.0"
DOC_MAXLEN = 180
NBITS = 2
INDEX_NAME = "colbertv2_train"


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ======================================================================
# Text Loading
# ======================================================================
def load_paper_text_map(parquet_file):
    """Build mapping: paper_id (str) -> 'title. abstract'"""
    import polars as pl

    df = pl.read_parquet(parquet_file)
    paper_map = {}
    for row in df.iter_rows(named=True):
        paper_id = row["paper_id"]
        title = (row.get("title", "") or "").strip()
        abstract = (row.get("abstract", "") or "").strip()
        paper_map[paper_id] = f"{title}. {abstract}".strip()
    return paper_map


# ======================================================================
# Chunking
# ======================================================================
def chunk_by_wordpieces(text: str, tokenizer, maxlen: int):
    """Split text into passages of at most `maxlen` wordpieces."""
    toks = tokenizer.encode(text, add_special_tokens=False)
    for i in range(0, len(toks), maxlen):
        s = tokenizer.decode(toks[i : i + maxlen], skip_special_tokens=True).strip()
        if s:
            yield s


# ======================================================================
# PID <-> Paper ID mapping
# ======================================================================
def load_pid_to_paperid_map(npy_path: Path):
    """Load the pid -> paper_id mapping (numpy array of string UUIDs)."""
    if npy_path.exists():
        return np.load(npy_path, allow_pickle=True)
    else:
        raise FileNotFoundError(f"Missing {npy_path}")


# ======================================================================
# Retrieval helper
# ======================================================================
def retrieve_suggestions(query_text, searcher, pid_to_paperid, k=500, query_id=None):
    """
    Retrieve top-k paper suggestions for a query using ColBERT searcher.
    Returns list of paper IDs (deduplicated, self-reference removed).
    """
    try:
        pids, ranks, scores = searcher.search(query_text, k=k)
        pids = pids.tolist() if isinstance(pids, np.ndarray) else pids
    except Exception as e:
        print(f"Error during search for query_id {query_id}: {e}")
        return []

    suggestions = []
    seen_paper_ids = set()
    for pid in pids:
        pid = int(pid)
        if pid < 0 or pid >= len(pid_to_paperid):
            continue
        paper_id = str(pid_to_paperid[pid])

        # Avoid duplicates and self-references
        if query_id and paper_id == query_id:
            continue
        if paper_id in seen_paper_ids:
            continue

        seen_paper_ids.add(paper_id)
        suggestions.append(paper_id)
        if len(suggestions) >= k:
            break

    return suggestions


# ======================================================================
# Label types & thresholds (shared with BM25 / SciBERT evals)
# ======================================================================
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


def extract_relevant_sets(references):
    """
    Given a list of reference dicts, extract the relevant paper_id sets
    for each label type.
    Returns dict: label_type_key -> set of relevant matched_paper_ids
    """
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
    """Mean Average Precision"""
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
    """Mean Reciprocal Rank"""
    for i, paper_id in enumerate(retrieved):
        if paper_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def compute_recall_at_k(relevant, retrieved, k):
    """Recall@k"""
    if not relevant:
        return 0.0
    hits = sum(1 for pid in retrieved[:k] if pid in relevant)
    return hits / len(relevant)


def compute_hit_rate_at_k(relevant, retrieved, k):
    """Hit Rate@k — 1 if any relevant doc is in top-k, else 0"""
    top_k = set(retrieved[:k])
    return 1.0 if any(item in top_k for item in relevant) else 0.0


def compute_dcg_at_k(relevant, retrieved, k):
    """Discounted Cumulative Gain at k"""
    dcg = 0.0
    for i, paper_id in enumerate(retrieved[:k]):
        if paper_id in relevant:
            dcg += 1.0 / np.log2(i + 2)
    return dcg


def compute_ndcg_at_k(relevant, retrieved, k):
    """Normalized Discounted Cumulative Gain at k"""
    if not relevant:
        return 0.0
    dcg = compute_dcg_at_k(relevant, retrieved, k)
    ideal_retrieved = list(relevant) + [0] * k
    idcg = compute_dcg_at_k(relevant, ideal_retrieved, k)
    return dcg / idcg if idcg > 0 else 0.0


def compute_all_metrics(relevant, retrieved):
    """Compute all metrics for a single query, returns a dict."""
    return {
        "map": compute_map(relevant, retrieved),
        "mrr": compute_mrr(relevant, retrieved),
        "recall": {k: compute_recall_at_k(relevant, retrieved, k) for k in EVAL_K_VALUES["recall"]},
        "ndcg": {k: compute_ndcg_at_k(relevant, retrieved, k) for k in EVAL_K_VALUES["ndcg"]},
        "hr": {k: compute_hit_rate_at_k(relevant, retrieved, k) for k in EVAL_K_VALUES["hr"]},
    }