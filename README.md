# CiteRoute

Facet-routed retrieval for **core citation recommendation** in the AI/ML literature.

> ### 📦 Data: **https://huggingface.co/datasets/trratul/CiteRoute**
> Corpus, labels, semantic factorizations, and the CoreML-168K splits.
> Everything needed to reproduce the paper is under `train_eval_set/` and `semantic_factors/`.

**Paper:** CiteRoute: Facet-Routed Retrieval for Core Citation Recommendation — submitted to the 2026 IEEE International Conference on Big Data (IEEE BigData 2026).

**Code:** this repository — the CiteRoute method, the **CoreML-168K** evaluation
setting, and the scripts to reproduce every result in the paper.

---

## Overview

Citation recommendation systems return papers that are *related* to a query. What
a researcher actually needs is the small set they **must cite** — experimental
baselines, directly extended methods, core datasets. CiteRoute retrieves that set
from a paper's title and abstract alone.

The central idea: citation importance is not one signal but several. A baseline
citation is announced by exact method and dataset names; a core-relevance
citation by conceptual overlap. A single query embedding averages the two.
CiteRoute instead **factorizes** the query into typed facets and **routes** each
to the retrieval paradigm that matches its signal.

- **168,837 papers** from 15 peer-reviewed venues
- **3,628,410 citation instances** labelled under four definitions of importance
- **CoreML-168K** — a temporally decoupled evaluation setting: 145,948-paper candidate pool, 2026 query papers, no overlap between training and retrieval
- Every learned and generative component is **open-weight** and runs on commodity GPUs

### Results

Recall@100 against the strongest of 24 baselines, under all four definitions:

| Definition | Best baseline | CiteRoute | Gain |
|---|---|---|---|
| Type I — experimental baseline | 0.401 | **0.444** | +0.043 (+10.7%) |
| Type II — core relevance | 0.370 | **0.424** | +0.054 (+14.6%) |
| Type III — mention frequency | 0.539 | **0.594** | +0.055 (+10.2%) |
| Type IV — core citation | 0.448 | **0.509** | +0.061 (+13.6%) |

CiteRoute leads on Recall@100 and Recall@500 under every definition, and is the
only system in the suite that leads on head quality (nDCG@10) and coverage at the
same time.

---

## Four Definitions of Citation Importance

Being cited says nothing about how much a paper mattered, so the target set needs
a definition. CiteRoute is evaluated against four — three read the query paper's
own text, one reads the behaviour of the authors who later built on it.

| Type | Label | Must-cite when |
|---|---|---|
| I | Experimental baseline (binary) | The query uses the cited paper as a direct comparison, takes a dataset or benchmark from it, or builds on a task or method it defines |
| II | Core relevance (1–5) | Score ≥ 4 — the cited work is central to the query's own task and method |
| III | Mention frequency | The cited paper is mentioned ≥ 3 times within the query paper |
| IV | Core citation (co-citation) | At least one later paper citing the query also cites it; *superficial* otherwise |

None of the four subsumes the others, and a method is scored against each
independently.

Types I and II are annotated at corpus scale by **locally hosted open-weight
judges** — `gpt-oss-120b` for Type I, `Gemma 4 31B` for Type II — chosen from 33
candidate models by a controlled comparison and validated against six human
annotators on a held-out sample. Types III and IV are computed directly from the
citation graph and need no LLM.

---

## Method

Four components, run in order:

1. **Semantic factorization.** An open-weight LLM rewrites the query's title and
   abstract into typed facets: `task`, `method`, `datasets_benchmarks`,
   `key_terms`, `contribution_summary`, and `compares_against` — the methods a
   paper of this type, on this task, at this date would be expected to benchmark
   against.
2. **Retrieval backbones.** A BM25 index and a contrastively fine-tuned dense
   encoder (GTE-base, NT-Xent, 128-d), each built once over the candidate pool
   and shared by every query and facet.
3. **Facet-routed retrieval and weighted fusion.** Entity-bearing facets are
   issued to BM25, conceptual facets to the dense encoder, and the undecomposed
   query is kept on both routes. The resulting 8 rankings are merged by
   reciprocal rank fusion with weights learned per definition.
4. **Cross-encoder reranking.** The top 200 fused candidates are rescored by a
   cross-encoder reading query, facets, and candidate jointly.

---

## Data

All data is hosted on the Hugging Face Hub:
[**huggingface.co/datasets/trratul/CiteRoute**](https://huggingface.co/datasets/trratul/CiteRoute/tree/main)

### Splits

CoreML-168K decouples training from retrieval — the candidate pool is not the
training split. Two split families are released, re-sliced from the *same*
annotated corpus at different year boundaries. Nothing is re-annotated between
them: a given reference carries identical Type I–IV labels in both.

| | **CoreML-168K** (Types I–III) | **CoreML-168K-T4** (Type IV) |
|---|---|---|
| Eval queries | ICML/ICLR/CVPR 2026 — 15,761 | ICML/ICLR/CVPR 2023 — 5,383 (5,033 usable) |
| Candidate pool | ≤ 2025 — 145,948 | ≤ 2022 — 77,354 |
| Train (core set) | ≤ 2025 — 103,096 | ≤ 2022 — 47,437 |
| Label evidence | text of the query | followers, 2024–2026 |

Type IV labels are defined by *followers* — later papers that cite the query — so
that tier needs its own year-shifted split. The 2024–2026 papers supplying those
labels fall outside the T4 pool by the same ≤ 2022 cut that defines it, so no
method can retrieve one and no future information reaches a ranking.

### Expected local layout

```
citeroute/
├── data/
│   ├── papers                    # source PDFs (not redistributed — see License)
│   ├── metadata                  # Provided on Hugging Face
│   ├── grobid_output             # Generated by the extraction pipeline
│   ├── citation_contexts         # Generated by the extraction pipeline
│   ├── prompt_scores             # Provided on Hugging Face
│   ├── semantic_factors          # Provided on Hugging Face
│   │   └── <venue>/<year>/<paper_id>.json
│   └── train_eval_set            # Provided on Hugging Face
│       ├── v7.0                  # must match DATA_VERSION in your .env
│       │   ├── all_papers_with_refs_and_labels.parquet
│       │   ├── train_v7.0.parquet
│       │   ├── eval_v7.0.parquet
│       │   ├── candidate_pool_v7.0.parquet
│       │
│       └── v7.0-t4
│       │   ├── train_v7.0-t4.parquet
│       │   ├── eval_v7.0-t4.parquet
│       │   ├── candidate_pool_v7.0-t4.parquet
│
└── CiteRoute                     # this repo
```

> **To reproduce the reported results you need only `train_eval_set` and
> `semantic_factors`.** The other folders are required only if you want to
> rebuild the corpus, citation contexts, and labels from scratch.

We release metadata, citation graphs, semantic factorizations, and
LLM-generated labels only; source PDFs are not redistributed.

---

## Quick Start

```bash
git clone https://github.com/NIU-Graph-Intelligence/CiteRoute.git
cd CiteRoute
pip install -r requirements.txt
cp .env.example .env      # then edit the paths below
```

### Configuration

All paths come from `.env`, so the repo runs unchanged on any machine. In
practice only `DATA_DIR` differs between servers.

| Variable | Meaning |
|---|---|
| `ROOT_DIR` | Repository root |
| `DATA_DIR` | Where the data lives (dataframes, semantic factors) |
| `OUTPUT_DIR` | Where this repo writes checkpoints, indices, runs, and results |
| `DATA_VERSION` | Split version, e.g. `v2.0` — must match the directory under `train_eval_set/` |
| `FACETS_DIR` | Semantic factorization JSONs (defaults to `$DATA_DIR/semantic_factors`) |
| `TYPE_4_EVAL_PARQUET` | Path to the Type IV eval slice (CoreML-168K-T4) |
| `BACKBONE_MODEL` | Dense backbone, default `thenlper/gte-base` |
| `RETRIEVE_DEPTH` / `RERANK_DEPTH` | Per-view retrieval depth (1000) and rerank window (200) |

Every stage prints its resolved paths in the first lines of its log, so a
misconfigured `.env` is obvious immediately.

### Running the pipeline

```bash
./run_all.sh                  # all four definitions
./run_all.sh type_1           # a single definition
```

Every stage is resumable — re-running after an interruption picks up where it
stopped. Logs land in `./logs/`.

| Stage | Script | What it does |
|---|---|---|
| 0 | `0-consolidate-facets.py` | Consolidate per-paper factorization JSONs into one parquet |
| 1 | `1-finetune-backbone.py` | Fine-tune the dense backbone with NT-Xent, one per definition |
| 2 | `2-generate-embeddings.py` | Embed the candidate pool and all query facets |
| 3 | `3-build-sparse-index.py` | Build the BM25 index over the pool |
| 4 | `4-facet-retrieval.py` | Retrieve one ranked list per facet view (depth 1,000) |
| 5 | `5-fit-fusion-weights.py` | Fit per-definition fusion weights λ on held-out training queries |
| 6 | `6-train-reranker.py` | Train the cross-encoder on mined hard negatives |
| 7 | `7-evaluate.py` | Fuse, rerank, and write the main results |
| 8 | `8-ablations.py` | Component, leave-one-facet-out, and leakage ablations |

Useful flags:

```bash
python 7-evaluate.py --types type_1 type_2 --no-reranker   # fusion-only
python 7-evaluate.py --uniform-rrf                         # ignore learned λ
python 8-ablations.py --skip-rerank                        # faster facet ablations
```

### Regenerating the semantic factorizations

The released factorizations were produced with a locally hosted open-weight model
at temperature 0 under schema-constrained decoding. To regenerate them — for a
new corpus, or to test a different factorizer — write the JSONs into a separate
`FACETS_DIR` and re-run stages 0, 2, 4, and 7 against it.

---

## Supported Venues

NeurIPS, ICML, ICLR, AAAI, IJCAI, CVPR, ICCV, ECCV, ACL, EMNLP, NAACL, COLT, UAI,
AISTATS, and JMLR. Coverage begins with the deep-learning era (most venues from
2012–2013 onward) and extends through the 2026 proceedings.

---

## License

Source code is licensed under the MIT License.

The dataset is released under CC BY 4.0. This project does not claim ownership of
paper metadata, abstracts, or PDFs retrieved from public conference and publisher
websites; those materials remain subject to the rights and terms of their
respective authors, publishers, and source websites.