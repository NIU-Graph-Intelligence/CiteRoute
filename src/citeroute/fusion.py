"""Rank fusion (Eq. 5): reciprocal rank fusion with optional learned weights."""

import numpy as np

from .config import RRF_K, VIEWS
from .utils import log


def rrf_fuse(view_runs, weights=None, k=RRF_K, depth=None):
    """Fuse per-view ranked lists into one ranking.

    view_runs : {view_name: [paper_id, ...]} ranked best-first
    weights   : {view_name: lambda_i}; defaults to 1.0 (classical RRF)
    """
    weights = weights or {}
    scores = {}
    for view, ranked in view_runs.items():
        w = float(weights.get(view, 1.0))
        if w == 0.0:
            continue
        for rank, pid in enumerate(ranked, start=1):
            scores[pid] = scores.get(pid, 0.0) + w / (k + rank)
    fused = sorted(scores.items(), key=lambda kv: -kv[1])
    if depth:
        fused = fused[:depth]
    return [pid for pid, _ in fused], {pid: s for pid, s in fused}


def rrf_features(view_runs, candidates, k=RRF_K, views=None):
    """Per-candidate feature vector of reciprocal ranks, one column per view."""
    views = views or VIEWS
    pos = {v: {pid: r for r, pid in enumerate(view_runs.get(v, []), start=1)} for v in views}
    X = np.zeros((len(candidates), len(views)), dtype=np.float32)
    for i, pid in enumerate(candidates):
        for j, v in enumerate(views):
            r = pos[v].get(pid)
            if r is not None:
                X[i, j] = 1.0 / (k + r)
    return X


def fit_weights(samples, views=None, seed=42):
    """Learn per-view weights lambda_i by logistic regression on RRF features.

    samples: list of (view_runs, gold_set) from TRAIN queries only.
    Weights are clipped at 0 (Eq. 5 requires lambda_i >= 0) and normalised so
    the mean weight is 1, which keeps fused scores on the classical-RRF scale.
    """
    views = views or VIEWS
    Xs, ys = [], []
    for view_runs, gold in samples:
        cands = sorted({pid for r in view_runs.values() for pid in r})
        if not cands or not gold:
            continue
        Xs.append(rrf_features(view_runs, cands, views=views))
        ys.append(np.array([1 if pid in gold else 0 for pid in cands], dtype=np.int8))
    if not Xs:
        log("  No usable fusion-training samples; falling back to uniform weights.")
        return {v: 1.0 for v in views}

    X = np.vstack(Xs)
    y = np.concatenate(ys)
    log(f"  Fitting fusion weights on {X.shape[0]} candidate rows "
        f"({int(y.sum())} positives, {len(views)} views)")
    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        clf.fit(X, y)
        raw = clf.coef_.ravel()
    except ImportError:
        # Correlation fallback keeps the pipeline runnable without sklearn.
        log("  scikit-learn not available; using correlation-based weights.")
        raw = np.array([np.corrcoef(X[:, j], y)[0, 1] if X[:, j].std() > 0 else 0.0
                        for j in range(X.shape[1])])
        raw = np.nan_to_num(raw)

    w = np.clip(raw, 0.0, None)
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.mean()
    weights = {v: float(w[j]) for j, v in enumerate(views)}
    for v, val in sorted(weights.items(), key=lambda kv: -kv[1]):
        log(f"    lambda[{v:22s}] = {val:.3f}")
    return weights
