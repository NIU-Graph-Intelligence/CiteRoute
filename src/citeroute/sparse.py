"""BM25 sparse retrieval (Eq. 3), implemented on scipy sparse matrices.

Self-contained on purpose: no Java (Pyserini) and no extra pip dependency
beyond scipy, and fast enough for the 145,948-paper pool because scoring a
query reduces to summing precomputed weight columns.
"""

import pickle
import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .utils import log

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")

K1 = 1.5
B = 0.75


def tokenize(text):
    return TOKEN_RE.findall((text or "").lower())


class BM25Index:
    """Precomputed BM25 weight matrix: score(q, d) = sum_{w in q} W[d, w]."""

    def __init__(self, doc_ids, weights, vocab):
        self.doc_ids = doc_ids
        self.weights = weights.tocsc()   # column slicing per query term
        self.vocab = vocab

    # ---------------- build ----------------
    @classmethod
    def build(cls, doc_ids, texts, k1=K1, b=B):
        log(f"  Tokenising {len(doc_ids)} documents...")
        vocab, indptr, indices, data = {}, [0], [], []
        doc_lens = np.zeros(len(doc_ids), dtype=np.float32)
        for i, text in enumerate(texts):
            counts = {}
            for tok in tokenize(text):
                tid = vocab.setdefault(tok, len(vocab))
                counts[tid] = counts.get(tid, 0) + 1
            doc_lens[i] = sum(counts.values())
            indices.extend(counts.keys())
            data.extend(counts.values())
            indptr.append(len(indices))
        n_docs, n_terms = len(doc_ids), len(vocab)
        tf = sp.csr_matrix(
            (np.array(data, dtype=np.float32), np.array(indices), np.array(indptr)),
            shape=(n_docs, n_terms),
        )
        log(f"  Vocabulary: {n_terms} terms, {tf.nnz} postings")

        # IDF (Robertson/Okapi with +0.5 smoothing, floored at 0)
        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

        # BM25 weights, computed in place on the CSR data array
        avgdl = float(doc_lens.mean()) if n_docs else 1.0
        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        denom_norm = k1 * (1.0 - b + b * doc_lens[rows] / max(avgdl, 1e-6))
        w = tf.data * (k1 + 1.0) / (tf.data + denom_norm)
        w *= idf[tf.indices]
        weights = sp.csr_matrix((w.astype(np.float32), tf.indices, tf.indptr),
                                shape=(n_docs, n_terms))
        log(f"  BM25 index built (avgdl={avgdl:.1f})")
        return cls(list(doc_ids), weights, vocab)

    # ---------------- query ----------------
    def score(self, query_text):
        tids = [self.vocab[t] for t in tokenize(query_text) if t in self.vocab]
        if not tids:
            return None
        sub = self.weights[:, tids]
        return np.asarray(sub.sum(axis=1)).ravel()

    def search(self, query_text, depth):
        scores = self.score(query_text)
        if scores is None:
            return [], []
        depth = min(depth, len(self.doc_ids))
        top = np.argpartition(-scores, depth - 1)[:depth]
        top = top[np.argsort(-scores[top])]
        return [self.doc_ids[i] for i in top], scores[top].tolist()

    # ---------------- persistence ----------------
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"doc_ids": self.doc_ids,
                         "weights": self.weights.tocsr(),
                         "vocab": self.vocab}, f, protocol=4)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["doc_ids"], d["weights"], d["vocab"])
