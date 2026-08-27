# CiteCoute

Retrieval method for **must-cite citation recommendation**: given a paper's
title and abstract, find the papers it *has* to cite.

The idea in one sentence: instead of squeezing the whole query into one
embedding, an LLM splits it into **facets** (task, method, datasets, key
terms, competitors), each facet goes to the retriever that handles its kind of
signal best, and the separate result lists are merged and reordered.

---

## Part 1 — What the pipeline does, step by step

Nine scripts, run in order. Each one is resumable: re-running skips work that
is already done.

### Step 0 · Collect the factorizations — `0-consolidate-facets.py`

**What it does.** Your `semantic_factorizer` package already wrote one small
JSON per paper. This step gathers all ~168K of them into a single table so
later steps don't have to open 168K files.

**How it finds them.** From the dataframes it reads `venue`, `year` and
`paper_id`, then looks directly at
`$DATA_DIR/semantic_factors/<venue>/<year>/<paper_id>.json`. Anything not
found there is picked up by one fallback scan (a few papers sit under a
different year folder because the factorizer grouped by metadata-file year
while the dataframe stores the paper's own year).

**What to check in the log.** It prints how many papers have a factorization
and the fill rate of each individual facet. Anything below 80% is flagged.

**Output.** `$OUTPUT_DIR/facets/facets.parquet` and `facets/coverage.json`.

---

### Step 1 · Train the dense encoder — `1-finetune-backbone.py`

**What it does.** Fine-tunes a text encoder so that a paper's embedding lands
near the embeddings of the papers it must cite. Trained separately for each
label type, on the training set only.

**How.** NT-Xent contrastive loss: pull each query towards its must-cite
papers, push it away from the other papers in the batch.

**Model.** GTE-base by default (see Part 3 for why), with **mean pooling** —
GTE is trained that way, and using CLS pooling instead would quietly ruin
retrieval quality. Pooling is auto-detected from the model name.

**Shortcut.** If you already have a fine-tuned checkpoint of the *same*
architecture, adopt it instead of training:

```bash
BACKBONE_MODEL=allenai/scibert_scivocab_uncased \
python 1-finetune-backbone.py --reuse-baseline \
    --baseline-dir /home/ratul/citeroute/masterset-benchmark/output/dense/SciBERT-NTXent
```

`BACKBONE_MODEL` must match the checkpoint — adopting SciBERT weights while
the default GTE architecture is configured will fail loudly (by design).
The path may point at the model root, at `fine_tuned_models/`, or at a single
`.pt` file; if `final_model.pt` is missing, the highest `epoch_N.pt` is used
and logged. Types with no checkpoint are reported as MISSING and skipped, so a
type still in training does not block the others.

### Running two backbones for the ablation

Use a separate `OUTPUT_DIR` per backbone so the two never overwrite each
other's embeddings:

```bash
# primary: GTE-base, trained here
OUTPUT_DIR=.../output/citeroute ./run_all.sh

# ablation: adopt the SciBERT-NTX checkpoints
OUTPUT_DIR=.../output/mustcite_scibert \
BACKBONE_MODEL=allenai/scibert_scivocab_uncased \
BASELINE_BACKBONE_DIR=.../SciBERT-NTXent \
REUSE_BASELINE=1 ./run_all.sh
```

**Output.** `$OUTPUT_DIR/backbone/<type>/final_model.pt`

---

### Step 2 · Turn text into vectors — `2-generate-embeddings.py`

**What it does.** Two things, per label type:

1. Embeds all 145,948 candidate-pool papers (title + abstract). This is the
   searchable index on the document side.
2. Embeds each **conceptual facet** of every evaluation query separately —
   so the query's `task` text becomes one vector, its `method` text another,
   and so on, plus one for the undecomposed title+abstract.

**Missing facets.** If a query has no text for a facet (null field, or no
factorization at all), it is *left out of that facet's file* rather than
embedded as an empty string — an empty string produces a meaningless vector
that would still get searched and pollute the results. Each file therefore
carries its own list of paper IDs.

**Output.** `$OUTPUT_DIR/embeddings/<type>/candidates_embeddings.pt` and
`query_<facet>_embeddings.pt`

---

### Step 3 · Build the keyword index — `3-build-sparse-index.py`

**What it does.** Builds a BM25 index over the same candidate pool, using
title + abstract. BM25 matches *exact words*, which is what finds dataset and
method names.

Implemented on scipy sparse matrices — no Java, no extra dependency — and
verified against a reference BM25 implementation.

**Output.** `$OUTPUT_DIR/sparse/bm25_index.pkl` (shared by all label types)

---

### Step 4 · Search once per facet — `4-facet-retrieval.py`

**The core step.** Each facet is sent to the retriever suited to it, and each
produces its own ranked list of 1,000 papers:

| Facet (query side) | Goes to | Because |
|---|---|---|
| `datasets_benchmarks` | BM25 | dataset names are exact strings |
| `key_terms` | BM25 | canonical terminology is exact |
| `compares_against` | BM25 | method names are exact |
| `sp_full` (whole query) | BM25 | reproduces the plain BM25 baseline |
| `task` | dense encoder | conceptual, wording varies |
| `method` | dense encoder | conceptual |
| `contribution_summary` | dense encoder | conceptual |
| `full` (whole query) | dense encoder | keeps the strongest single-vector baseline in the mix |

That is **8 lists per query** with the v1 factorizations you are running.

It does this for three groups of queries:
- `eval` — the benchmark queries, for the final numbers
- `fusion` — a sample of 2,000 training queries, used by Step 5
- `reranker` — a sample of 8,000 training queries, used by Step 6

**Output.** `$OUTPUT_DIR/runs/<type>/<split>_<view>.npz`

---

### Step 5 · Learn how much to trust each facet — `5-fit-fusion-weights.py`

**What it does.** Merges the 8 lists into one using Reciprocal Rank Fusion: a
paper's score is the sum of `1/(60 + rank)` over the lists it appears in, so
papers that several facets agree on rise to the top.

Not all facets deserve equal say, and which facet matters depends on the label
type. So a logistic regression on the training queries learns one weight per
facet **per label type**. This is also the interpretable result for the paper:
the weights show which facet serves which tier.

**Output.** `$OUTPUT_DIR/fusion/<type>_weights.json`

Optional — if you skip this step, Step 7 falls back to equal weights.

---

### Step 6 · Train the reranker — `6-train-reranker.py`

**Why.** Steps 2–4 score the query and each candidate *separately* (fast, but
coarse). A cross-encoder reads the query and one candidate **together**, which
is far more accurate but too slow for 145K papers — so it is applied only to
the top 200.

**Training data.** Positives are the real must-cite papers. Negatives are
*hard* ones: papers the pipeline retrieved near the top but that are not
must-cite — exactly the mistakes it needs to learn to avoid.

**Type 4 gets better negatives.** Its labels distinguish *core* citations
(2.0) from *superficial* ones (1.0) — papers the query really does cite, but
which no later paper co-cites. Telling those two apart is precisely the task,
so superficial citations are drawn first into the negative pool.

**Output.** `$OUTPUT_DIR/reranker/<type>/final_model.pt`

Optional — skip it and Step 7 reports fusion-only results.

---

### Step 7 · Final results — `7-evaluate.py`

Runs the whole path on the evaluation queries: fuse the 8 lists with the
learned weights, rerank the top 200, then score the ranking.

Reports MAP, MRR, nDCG@{5,10,20,30,50}, Recall@{10,50,100,500}, P@10, and
HR@{10,20} — the union of every metric your four result tables use, so one run
fills all of them.

Gold references that are not in the candidate pool cannot be retrieved by
anyone; they are excluded and the count is logged (report it in the paper).

**Output.** `$OUTPUT_DIR/evaluation_results/mustcite_evaluation_metrics.{txt,json}`
— same format as your baseline result files.

---

### Step 8 · Ablations — `8-ablations.py`

Re-scores the Step-4 lists under different settings. **No retraining**, since
every ablation is just a subset of facets or a switch turned off:

- **Components:** full / without reranker / without learned weights / without
  factorization (undecomposed views only)
- **One facet at a time removed**, per label type — this is the facet-to-tier
  alignment evidence
- **Leakage control:** remove the facets where the LLM infers beyond the given
  text (`compares_against`, and `builds_on` under v2)

**Output.** `$OUTPUT_DIR/evaluation_results/ablations.{txt,json}`

---

## Part 2 — Running it

### Install and configure

```bash
pip install -r requirements.txt
cp .env.example .env      # then edit for this machine
```

Only three paths matter, and only `DATA_DIR` normally differs between servers:

```properties
# Lancer
ROOT_DIR=/home/ratul/citeroute
DATA_DIR=/home/ratul/citeroute/data
OUTPUT_DIR=/home/ratul/citeroute/citeroute-recommendation/output

# Rider
ROOT_DIR=/home/ratul/citeroute
DATA_DIR=/mnt/data/data/data
OUTPUT_DIR=/home/ratul/citeroute/citeroute-recommendation/output
```

Every stage prints the paths it resolved in its first lines, and stops with a
clear message naming the missing file if something is wrong.

Expected inputs:

| Path | Produced by |
|---|---|
| `$DATA_DIR/train_eval_set/$DATA_VERSION/{train,eval,candidate_pool}_$DATA_VERSION.parquet` | `dataframe_builder` script 4 |
| `$DATA_DIR/train_eval_set/$DATA_VERSION/all_papers_with_refs_and_labels.parquet` | `dataframe_builder` script 3 |
| `$DATA_DIR/semantic_factors/<venue>/<year>/<paper_id>.json` | `semantic_factorizer` |

### Run

```bash
./run_all.sh                       # everything, all four label types
./run_all.sh type_1                # one label type
REUSE_BASELINE=1 ./run_all.sh      # adopt SciBERT-NTX instead of training
```

Or stage by stage:

```bash
python 0-consolidate-facets.py
python 1-finetune-backbone.py
python 2-generate-embeddings.py
python 3-build-sparse-index.py
python 4-facet-retrieval.py
python 5-fit-fusion-weights.py
python 6-train-reranker.py
python 7-evaluate.py
python 8-ablations.py
```

Every script accepts `--types type_1 type_2 ...` to run a subset, useful for
splitting across GPUs (`CUDA_VISIBLE_DEVICES=0 python ... --types type_1`).

### Suggested first run

1. `python 0-consolidate-facets.py` — read the coverage report before anything
   else. If a facet's fill rate is very low, that facet will contribute little.
2. `REUSE_BASELINE=1 ./run_all.sh type_1` — one type end to end, to confirm the
   plumbing.
3. Then the remaining types.

---

## Part 3 — Design decisions and why

### The four label types

Every step runs independently per type: own encoder, own weights, own
reranker, own evaluation.

| Type | A paper counts as positive when | Where it comes from |
|---|---|---|
| `type_1` | it is used as an experimental baseline | MasterSet |
| `type_2` | its core-relevance score is >= 4 | MasterSet |
| `type_3` | it is mentioned >= 3 times in the citing paper | MasterSet |
| `type_4` | it is a **core citation** (label 2.0): some later paper citing the query also cites it | HLM-Cite style |

Types 1–3 are **textual** — they come from how the query paper talks about the
cited work. Type 4 is **structural** — it comes from how the community later
treats the pair. Agreement between them is evidence that must-cite status is
not an artefact of either signal.

### Type 4 needs its own query set

A query only has a Type-4 label if later papers have already cited it. The 2026
evaluation queries have no such followers, so Type 4 must be evaluated on an
older slice (your Table V uses `v7.0-T4`: 5,033 queries, 77,354 candidates):

```properties
TYPE_4_EVAL_PARQUET=${DATA_DIR}/train_eval_set/v7.0/eval_type4.parquet
```

Every stage resolves the eval dataframe per type, so Types 1–3 keep using the
2026 queries while Type 4 uses its own. Leave it unset and Type 4 simply
reports no labelled queries.

### Backbone: GTE-base

Chosen from your own baseline tables. Across all four types, GTE-base
zero-shot beats the *fine-tuned* SciBERT-NTX on every top-heavy metric
(MAP, MRR, nDCG, P@10), at the same 110M size, and it beats GTE-large and both
v1.5 variants. On Type 4 it is statistically tied with the proprietary
OpenAI-3-large (R@100 0.4256 vs 0.4267) and actually wins R@500 — a good
open-weight story for the paper.

Since NT-Xent fine-tuning raised SciBERT's Recall@100 roughly twentyfold, the
same treatment on a much better starting point is the natural backbone.

**Pooling matters.** GTE uses mean pooling, BERT/SciBERT use CLS. Auto-detected
from the model name; override with `POOLING=mean|cls`.

### Optional: a second encoder in the mix

On Types 1–3 the split is clean — SciBERT-NTX wins every deep-recall metric
(R@100, R@500) while GTE wins every top-heavy one. Since Recall@K is the
headline metric, you can add your existing SciBERT-NTX checkpoints as one
extra fusion view at no training cost:

```properties
SECONDARY_BACKBONE_DIR=/home/ratul/citeroute/masterset-benchmark/output/dense/SciBERT-NTXent/fine_tuned_models
```

This adds a 9th view, `full@scibert_ntx`; the learned weights decide per type
how much to trust it.

### Factorization schema: v1 vs v2

| Schema | Fields | Effect |
|---|---|---|
| **v1** — *what you are running* | `expected_baselines` | Read as the `compares_against` facet (same meaning). 8 views total. |
| **v2** — *not currently used* | `builds_on` + `compares_against` | Adds a 9th view, `builds_on`, separating works a paper *extends* from works it *competes with*. |

Both are read automatically and Step 0 reports which it found. Running v1 costs
exactly one of nine possible views. Regenerating with v2 is optional and makes
a clean ablation: point `FACETS_DIR` at the new tree and re-run steps 0, 2, 4, 7.

### Missing or incomplete factorizations

| Situation | What happens |
|---|---|
| Paper has no JSON at all | Still retrieved through the undecomposed `full` and `sp_full` views |
| Paper has a JSON but one facet is empty | Excluded from *that facet's view only*; keeps every other view |
| A candidate-pool paper lacks facets | Irrelevant — the document side is indexed by title+abstract, which always exists |

No query is ever dropped. Verified on a corpus with 15% of JSONs deleted and
20% partially emptied: every query was still scored, and no facet-less query
leaked into a facet-specific run.

---

## Part 4 — Reference

### Outputs

```
$OUTPUT_DIR/
├── facets/facets.parquet               all factorizations + coverage.json
├── backbone/<type>/final_model.pt      dense encoder, per label type
├── embeddings/<type>/                  candidates_*.pt, query_<facet>_*.pt
├── sparse/bm25_index.pkl               BM25 index (shared)
├── runs/<type>/<split>_<view>.npz      one ranked list per facet
├── fusion/<type>_weights.json          learned facet weights, per type
├── reranker/<type>/final_model.pt      cross-encoder
└── evaluation_results/
    ├── mustcite_evaluation_metrics.{txt,json}
    └── ablations.{txt,json}
```

### Settings worth knowing

| Variable | Default | Meaning |
|---|---|---|
| `BACKBONE_MODEL` | `thenlper/gte-base` | dense encoder |
| `POOLING` | auto | `mean` for GTE/E5/BGE, `cls` for BERT/SciBERT |
| `RETRIEVE_DEPTH` | 1000 | results per facet |
| `RERANK_DEPTH` | 200 | how many the cross-encoder reorders (keep > 100 so Recall@100 can improve) |
| `EPOCHS` | 3 | encoder training epochs |
| `RERANKER_EPOCHS` | 1 | cross-encoder epochs |
| `RERANKER_MAX_LEN` | 256 | lower is faster |
| `NEGATIVES_PER_POSITIVE` | 4 | hard negatives per positive |
| `FUSION_TRAIN_QUERIES` | 2000 | training queries for the weights |
| `RERANKER_TRAIN_QUERIES` | 8000 | training queries for the reranker |
| `TYPE_4_EVAL_PARQUET` | unset | Type 4's own evaluation slice |
| `SECONDARY_BACKBONE_DIR` | unset | adds the `full@scibert_ntx` view |

### Package layout

```
citeroute/
├── config.py       paths, label types, facets, all settings
├── data.py         parquet + factorization loading, ground truth
├── metrics.py      MAP, MRR, nDCG, Recall, Precision, HR
├── model.py        dense encoder (mean/CLS pooling) + NT-Xent loss
├── sparse.py       BM25 on scipy sparse matrices
├── fusion.py       RRF + weight fitting
├── rerank.py       cross-encoder
├── runs.py         reading the per-facet run files
├── pipeline.py     shared fuse -> rerank -> score path
└── utils.py        logging helpers
```
