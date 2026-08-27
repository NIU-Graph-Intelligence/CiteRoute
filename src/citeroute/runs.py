"""Reading per-view run files produced by Stage 4."""

from pathlib import Path

import numpy as np

from .config import RUNS_DIR, VIEWS
from .utils import log


def run_path(label_key, split, view):
    return Path(RUNS_DIR) / label_key / f"{split}_{view}.npz"


def load_run(path):
    d = np.load(path, allow_pickle=True)
    return list(d["query_ids"]), [list(r) for r in d["doc_ids"]]


def load_view_runs(label_key, split, views=None, quiet=False):
    """{query_id: {view: [doc_id, ...]}} for all available views."""
    views = views or VIEWS
    per_query = {}
    found = []
    for view in views:
        p = run_path(label_key, split, view)
        if not p.exists():
            if not quiet:
                log(f"    missing run: {p.name}")
            continue
        found.append(view)
        qids, ranked = load_run(p)
        for qid, docs in zip(qids, ranked):
            per_query.setdefault(qid, {})[view] = docs
    if not quiet:
        log(f"    loaded {len(found)} views for {len(per_query)} queries: {found}")
    return per_query, found
