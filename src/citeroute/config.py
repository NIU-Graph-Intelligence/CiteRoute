"""Central configuration for CiteCoute.

Mirrors the masterset-benchmark conventions: paths come from .env
(ROOT_DIR / OUTPUT_DIR), label thresholds and eval K values are identical
to the baselines so numbers are directly comparable.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ======================================================================
# Paths — three independent roots so the repo runs unchanged on any server
#
#   ROOT_DIR   repository root
#   DATA_DIR   where the data lives (dataframes, semantic factors)
#   OUTPUT_DIR where this repo writes its artifacts
#
# Only DATA_DIR differs between servers in practice (e.g. Lancer
# /home/ratul/citeroute/data vs Rider /mnt/data/data/data), so every data path
# below is derived from DATA_DIR and nothing is hard-coded to ROOT_DIR.
# ======================================================================
load_dotenv()
ROOT_DIR = Path(os.getenv("ROOT_DIR", ""))
DATA_DIR = Path(os.getenv("DATA_DIR", ""))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", ROOT_DIR / "output/citeroute"))
DATA_VERSION = os.getenv("DATA_VERSION", "") # use this format --> "v1.0" / "v7.0"

# Dataframes. Layout: $DATA_DIR/train_eval_set/<version>/...
# Override any single file directly with TRAIN_PARQUET / EVAL_PARQUET /
# CANDIDATE_PARQUET / ALL_PAPERS_PARQUET if a server deviates.
DATAFRAME_DIR = Path(os.getenv("DATAFRAME_DIR", DATA_DIR / "train_eval_set" / DATA_VERSION))
TRAIN_PARQUET = Path(os.getenv("TRAIN_PARQUET", DATAFRAME_DIR / f"train_{DATA_VERSION}.parquet"))
EVAL_PARQUET = Path(os.getenv("EVAL_PARQUET", DATAFRAME_DIR / f"eval_{DATA_VERSION}.parquet"))
CANDIDATE_PARQUET = Path(os.getenv("CANDIDATE_PARQUET",
                                   DATAFRAME_DIR / f"candidate_pool_{DATA_VERSION}.parquet"))
ALL_PAPERS_PARQUET = Path(os.getenv("ALL_PAPERS_PARQUET",
                                    DATAFRAME_DIR / "all_papers_with_refs_and_labels.parquet"))

# Semantic factorization JSONs produced by the semantic_factorizer package.
# Layout: $FACETS_DIR/<venue>/<year>/<paper_id>.json — locatable directly from
# the dataframe's venue / year / paper_id columns (no need for the 'path' column).
FACETS_JSON_DIR = Path(os.getenv("FACETS_DIR", DATA_DIR / "semantic_factors"))
FACETS_PARQUET = OUTPUT_DIR / "facets" / "facets.parquet"


def facet_json_path(venue, year, paper_id, facets_dir=None):
    """Direct location of one paper's factorization JSON."""
    base = Path(facets_dir) if facets_dir else FACETS_JSON_DIR
    return base / str(venue) / str(int(year)) / f"{paper_id}.json"

# Optional: reuse an already fine-tuned baseline backbone instead of training
# a new one (Stage 1 --reuse-baseline). Expected layout:
#   $BASELINE_BACKBONE_DIR/<type_key>/final_model.pt
BASELINE_BACKBONE_DIR = os.getenv("BASELINE_BACKBONE_DIR", "")

# Per-stage output roots
BACKBONE_DIR = OUTPUT_DIR / "backbone"
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"
SPARSE_DIR = OUTPUT_DIR / "sparse"
RUNS_DIR = OUTPUT_DIR / "runs"              # per-facet ranked lists
FUSION_DIR = OUTPUT_DIR / "fusion"
RERANKER_DIR = OUTPUT_DIR / "reranker"
RESULTS_DIR = OUTPUT_DIR / "evaluation_results"

# ======================================================================
# Label types — identical thresholds to the baseline suite
# ======================================================================
TYPE_2_THRESHOLD = 4.0
TYPE_3_THRESHOLD = 3.0
# Type 4 (HLM-Cite style): 2.0 = core citation, 1.0 = superficial, 0.0 = none.
# Only core citations count as positives; superficial ones are the natural
# HARD NEGATIVES for the reranker (cited, but not core).
TYPE_4_THRESHOLD = 2.0

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
    "type_4": {
        "name": "Type 4 (Core Citation)",
        "field": "type_4_output",
        "threshold_fn": lambda v: float(v) >= TYPE_4_THRESHOLD,
        "description": f"core citation (co-citation) >= {TYPE_4_THRESHOLD}",
    },
}

# Type 4 is only defined for queries that already have FOLLOWERS (later papers
# citing them), so 2026 eval queries have no type-4 labels. Its evaluation
# therefore runs on an older query slice; set TYPE_4_EVAL_PARQUET to point at
# it, otherwise the standard eval set is used and simply yields no labelled
# queries.
TYPE_4_EVAL_PARQUET = Path(os.getenv("TYPE_4_EVAL_PARQUET", "")) \
    if os.getenv("TYPE_4_EVAL_PARQUET") else None


def eval_parquet_for(label_key):
    """Eval dataframe for one label type (Type 4 may use its own slice)."""
    if label_key == "type_4" and TYPE_4_EVAL_PARQUET:
        return TYPE_4_EVAL_PARQUET
    return EVAL_PARQUET

# Union of the metric sets used by the four result tables, so one run fills
# every column. Types I-III report nDCG@{10,20,30}, R@{50,100}, HR@{10,20};
# the Type IV table instead reports nDCG@{5,10,20}, P@10, R@{10,100,500}.
EVAL_K_VALUES = {
    "recall": [10, 50, 100, 500],
    "ndcg": [5, 10, 20, 30, 50],
    "hr": [10, 20],
    "precision": [10],
}

# ======================================================================
# Facet schema and routing (Sec. 3.3-3.4 of the paper)
# ======================================================================
# Entity-bearing facets carry exact surface forms -> sparse lexical retrieval.
# Conceptual facets carry semantics -> dense retrieval with the backbone.
# "full" is the undecomposed title+abstract view, retained so factorization
# strictly ADDS signal instead of replacing the strongest baseline.
SPARSE_FACETS = ["datasets_benchmarks", "key_terms", "compares_against"]
DENSE_FACETS = ["task", "method", "builds_on", "contribution_summary"]
FULL_VIEW = "full"

# Views that are embedded and searched densely (query side).
DENSE_VIEWS = [FULL_VIEW] + DENSE_FACETS
# Views issued to BM25. The undecomposed view is also run sparsely, which
# reproduces the plain BM25 baseline as one component of the fusion; its run
# is named 'sp_full' so it does not collide with the dense 'full' view.
SPARSE_VIEWS = SPARSE_FACETS + [FULL_VIEW]
SPARSE_FULL_RUN = "sp_full"


def sparse_run_name(view):
    """Run-file name for a sparse view (avoids the dense/sparse 'full' clash)."""
    return SPARSE_FULL_RUN if view == FULL_VIEW else view


# ----------------------------------------------------------------------
# Secondary dense encoder (optional, strongly recommended).
#
# The baseline table shows a clean division of labour: SciBERT-NTX wins every
# deep-recall metric (R@100, R@500 on all three tiers) while the GTE family
# wins every top-heavy metric. Since Recall@K is the primary metric here, the
# already-trained SciBERT-NTX checkpoints are added as ONE extra fusion view
# (the undecomposed query), contributing their recall strength at zero extra
# training cost. Set SECONDARY_BACKBONE_DIR="" to disable.
# ----------------------------------------------------------------------
SECONDARY_BACKBONE_NAME = os.getenv("SECONDARY_BACKBONE_NAME", "scibert_ntx")
SECONDARY_BACKBONE_MODEL = os.getenv("SECONDARY_BACKBONE_MODEL",
                                     "allenai/scibert_scivocab_uncased")
SECONDARY_BACKBONE_DIR = os.getenv("SECONDARY_BACKBONE_DIR", "")

SECONDARY_VIEWS = ([f"{FULL_VIEW}@{SECONDARY_BACKBONE_NAME}"]
                   if SECONDARY_BACKBONE_DIR else [])

# All fusion views, in a fixed order (defines the feature-vector layout).
VIEWS = DENSE_VIEWS + SPARSE_FACETS + [SPARSE_FULL_RUN] + SECONDARY_VIEWS

# Which ground-truth tier each facet is designed to serve (Sec. 3.6).
# Used only for reporting the facet-tier alignment in the ablation table.
FACET_TARGET_TIER = {
    "compares_against": "type_1",
    "task": "type_2",
    "method": "type_2",
    "builds_on": "type_2",
    "datasets_benchmarks": "type_3",
    "key_terms": "type_3",
    "contribution_summary": "type_2",
}

# Facets that are lists in the JSON (joined with ", " to form a query string)
LIST_FACETS = {"datasets_benchmarks", "key_terms", "builds_on", "compares_against"}

# ======================================================================
# Backbone (contrastive dense encoder)
#
# Default: GTE-base. On the benchmark table GTE-base zero-shot already beats
# the fine-tuned SciBERT-NTX on every top-heavy metric (MAP/MRR/nDCG/HR) while
# being the same size as SciBERT-base, and it outperforms GTE-large and both
# v1.5 variants — so it is the strongest and cheapest starting point for
# NT-Xent adaptation.
#
# POOLING matters: GTE models are trained with MEAN pooling, BERT/SciBERT with
# CLS. Using the wrong one silently destroys retrieval quality, so it is
# auto-detected from the model name unless overridden.
# ======================================================================
BACKBONE_MODEL_NAME = os.getenv("BACKBONE_MODEL", "thenlper/gte-base")


def default_pooling(model_name):
    m = (model_name or "").lower()
    if "gte" in m or "e5" in m or "bge" in m or "sentence-transformers" in m:
        return "mean"
    return "cls"


POOLING = os.getenv("POOLING", "") or default_pooling(BACKBONE_MODEL_NAME)

EMBED_DIM = int(os.getenv("EMBED_DIM", "128"))
MAX_LEN = 512

# Secondary-encoder pooling/dim (defined here so both are available below).
SECONDARY_POOLING = (os.getenv("SECONDARY_POOLING", "")
                     or default_pooling(os.getenv("SECONDARY_BACKBONE_MODEL",
                                                  "allenai/scibert_scivocab_uncased")))
SECONDARY_EMBED_DIM = int(os.getenv("SECONDARY_EMBED_DIM", str(EMBED_DIM)))
BATCH_SIZE_FT = 16
BATCH_SIZE_EMB = 128
EPOCHS = int(os.getenv("EPOCHS", "3"))
LR = 2e-5
TEMPERATURE = 0.07
SEP = "[SEP]"

# ======================================================================
# Retrieval / fusion / reranking
# ======================================================================
RETRIEVE_DEPTH = int(os.getenv("RETRIEVE_DEPTH", "1000"))   # per-view depth
RRF_K = 60
RERANK_DEPTH = int(os.getenv("RERANK_DEPTH", "200"))        # top-C reranked
FUSION_TRAIN_QUERIES = int(os.getenv("FUSION_TRAIN_QUERIES", "2000"))

# Cross-encoder reranker
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL", "allenai/scibert_scivocab_uncased")
RERANKER_MAX_LEN = int(os.getenv("RERANKER_MAX_LEN", "256"))
RERANKER_EPOCHS = int(os.getenv("RERANKER_EPOCHS", "1"))
RERANKER_LR = 2e-5
RERANKER_BATCH_SIZE = int(os.getenv("RERANKER_BATCH_SIZE", "32"))
RERANKER_INFER_BATCH_SIZE = int(os.getenv("RERANKER_INFER_BATCH_SIZE", "256"))
RERANKER_TRAIN_QUERIES = int(os.getenv("RERANKER_TRAIN_QUERIES", "8000"))
NEGATIVES_PER_POSITIVE = int(os.getenv("NEGATIVES_PER_POSITIVE", "4"))

SEED = 42


def ensure_dirs():
    for d in [BACKBONE_DIR, EMBEDDINGS_DIR, SPARSE_DIR, RUNS_DIR,
              FUSION_DIR, RERANKER_DIR, RESULTS_DIR, FACETS_PARQUET.parent]:
        Path(d).mkdir(parents=True, exist_ok=True)


def describe_paths():
    """One-line-per-path summary, printed by every stage so a wrong .env on a
    given server is obvious in the first lines of the log."""
    return [
        f"ROOT_DIR   : {ROOT_DIR}",
        f"DATA_DIR   : {DATA_DIR}",
        f"OUTPUT_DIR : {OUTPUT_DIR}",
        f"dataframes : {DATAFRAME_DIR}  (version {DATA_VERSION})",
        f"facets     : {FACETS_JSON_DIR}",
    ]


def check_inputs(paths, hint=""):
    """Fail fast with an actionable message when a server's .env is wrong."""
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        lines = ["Missing required input(s):"] + [f"  - {m}" for m in missing]
        lines.append("")
        lines.extend(describe_paths())
        if hint:
            lines.append("")
            lines.append(hint)
        raise SystemExit("\n".join(lines))
