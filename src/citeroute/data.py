"""Data loading: parquets, text maps, facets, and ground truth."""

import json
from pathlib import Path

import polars as pl

from .config import (DENSE_FACETS, FULL_VIEW, LABEL_TYPES, LIST_FACETS, SEP,
                     SPARSE_FACETS)
from .utils import log


# ======================================================================
# Text
# ======================================================================
def load_paper_text_map(parquet_file):
    """paper_id -> 'title [SEP] abstract' (same format as the baselines)."""
    df = pl.read_parquet(parquet_file)
    out = {}
    for row in df.iter_rows(named=True):
        title = (row.get("title", "") or "").strip()
        abstract = (row.get("abstract", "") or "").strip()
        out[row["paper_id"]] = f"{title} {SEP} {abstract}"
    return out


def load_paper_title_map(parquet_file):
    try:
        df = pl.read_parquet(parquet_file)
        return {r["paper_id"]: r.get("title", "") for r in df.iter_rows(named=True)}
    except Exception as e:  # noqa: BLE001
        log(f"Warning: could not load title map: {e}")
        return {}


def paper_ids(parquet_file):
    return pl.read_parquet(parquet_file)["paper_id"].to_list()


def parse_references(row):
    refs = row["references"]
    return json.loads(refs) if isinstance(refs, str) else (refs or [])


# ======================================================================
# Ground truth
# ======================================================================
def extract_relevant_sets(references):
    """references -> {label_key: set(matched_paper_id)} using the shared thresholds."""
    relevant = {key: set() for key in LABEL_TYPES}
    for ref in references:
        mid = ref.get("matched_paper_id")
        if not mid:
            continue
        for key, cfg in LABEL_TYPES.items():
            raw = ref.get(cfg["field"])
            if raw is None:
                continue
            try:
                if cfg["threshold_fn"](raw):
                    relevant[key].add(mid)
            except (ValueError, TypeError):
                continue
    return relevant


def build_ground_truth(parquet_file, label_key, pool_ids=None, restrict_to=None):
    """{query_paper_id: set(gold_ids)} for one label type.

    Gold references outside the candidate pool are unretrievable by
    construction; they are dropped here and counted, exactly as the
    baseline eval scripts do.
    """
    df = pl.read_parquet(parquet_file)
    gt, outside = {}, 0
    for row in df.iter_rows(named=True):
        pid = row["paper_id"]
        if restrict_to is not None and pid not in restrict_to:
            continue
        rel = extract_relevant_sets(parse_references(row))[label_key]
        if pool_ids is not None:
            outside += len(rel - pool_ids)
            rel = rel & pool_ids
        if rel:
            gt[pid] = rel
    return gt, outside


# ======================================================================
# Facets
# ======================================================================
FACET_COLUMNS = [FULL_VIEW] + DENSE_FACETS + SPARSE_FACETS


def facet_to_text(name, value):
    """Render one facet value as a query string."""
    if value is None:
        return ""
    if name in LIST_FACETS:
        if not isinstance(value, list):
            return str(value).strip()
        return ", ".join(str(v).strip() for v in value if str(v).strip())
    return str(value).strip()


#: Facet-name aliases across factorization schema versions.
#: v1 emits a single `expected_baselines` list ("existing methods a paper like
#: this would most likely compare against"). That is exactly the semantics of
#: v2's `compares_against`, so v1 files populate that facet; v2's `builds_on`
#: has no v1 counterpart and stays empty, which the pipeline handles as a
#: normal missing facet (the view is simply absent for those queries).
FACET_ALIASES = {
    "compares_against": ["compares_against", "expected_baselines"],
    "builds_on": ["builds_on"],
}


def _facet_value(factors, name):
    for key in FACET_ALIASES.get(name, [name]):
        if key in factors and factors.get(key) not in (None, "", []):
            return factors.get(key)
    return factors.get(name)


def detect_schema(factors):
    """'v2' if the split fields are present, else 'v1'."""
    if "builds_on" in factors or "compares_against" in factors:
        return "v2"
    if "expected_baselines" in factors:
        return "v1"
    return "unknown"


def _facet_row(payload, pid):
    factors = payload.get("factors", {}) or {}
    row = {"paper_id": pid}
    for name in DENSE_FACETS + SPARSE_FACETS:
        row[name] = facet_to_text(name, _facet_value(factors, name))
    row["paper_type"] = factors.get("paper_type") or ""
    row["schema"] = payload.get("prompt_version") or detect_schema(factors)
    return row


def consolidate_facet_jsons(facets_dir, out_parquet, paper_index):
    """Collect factorization JSONs for the papers we actually need.

    `paper_index` is a list of (paper_id, venue, year) taken from the
    dataframes, so each JSON is addressed DIRECTLY at
        <facets_dir>/<venue>/<year>/<paper_id>.json
    rather than by walking the tree.

    One wrinkle is handled explicitly: semantic_factorizer groups output by the
    metadata-FILE year, while the dataframe stores the paper-level year, and
    the two can differ for a handful of papers. Any paper missed by the direct
    path is resolved by a single rglob pass that indexes paper_id -> file, so a
    year mismatch never looks like a missing factorization.
    """
    facets_dir = Path(facets_dir)
    rows, bad = [], 0
    misses = []

    for pid, venue, year in paper_index:
        fp = None
        try:
            cand = facets_dir / str(venue) / str(int(year)) / f"{pid}.json"
            if cand.exists():
                fp = cand
        except (TypeError, ValueError):
            pass
        if fp is None:
            misses.append(pid)
            continue
        try:
            rows.append(_facet_row(json.loads(fp.read_text(encoding="utf-8")), pid))
        except (json.JSONDecodeError, OSError):
            bad += 1

    log(f"  Direct path hit for {len(rows)}/{len(paper_index)} papers")

    # Fallback pass: resolve anything the direct path missed (e.g. a paper
    # whose factorization sits under a different year directory).
    if misses:
        log(f"  Resolving {len(misses)} misses via a one-time index of {facets_dir} ...")
        index = {}
        for fp in facets_dir.rglob("*.json"):
            index.setdefault(fp.stem, fp)
        recovered = 0
        for pid in misses:
            fp = index.get(pid)
            if fp is None:
                continue
            try:
                rows.append(_facet_row(json.loads(fp.read_text(encoding="utf-8")), pid))
                recovered += 1
            except (json.JSONDecodeError, OSError):
                bad += 1
        log(f"    Recovered {recovered} (mismatched venue/year directory); "
            f"{len(misses) - recovered} genuinely absent")

    if bad:
        log(f"  Skipped {bad} unreadable JSONs")
    if not rows:
        raise RuntimeError(
            f"No usable factorization JSONs under {facets_dir}. Check FACETS_DIR "
            f"in .env — expected <venue>/<year>/<paper_id>.json")

    df = pl.DataFrame(rows).unique(subset=["paper_id"], keep="first")

    # Report the factorization schema actually found on disk. v1 has no
    # builds_on facet, so that view will be absent — expected, not a fault.
    schemas = df["schema"].value_counts().sort("count", descending=True)
    for row in schemas.iter_rows(named=True):
        log(f"  Factorization schema {row['schema']!r}: {row['count']} papers")
    if any(str(r["schema"]).startswith("v1") for r in schemas.iter_rows(named=True)):
        log("  NOTE: v1 schema detected — 'expected_baselines' is mapped to the "
            "compares_against facet; the builds_on facet will be empty and its "
            "view simply omitted from fusion.")

    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_parquet)
    log(f"  Wrote {df.height} facet rows -> {out_parquet}")
    return df


def paper_index_from(parquets):
    """[(paper_id, venue, year)] over several dataframes, de-duplicated."""
    seen, out = set(), []
    for pq in parquets:
        if not Path(pq).exists():
            continue
        df = pl.read_parquet(pq)
        cols = df.columns
        if not {"paper_id", "venue", "year"} <= set(cols):
            log(f"  Warning: {pq} lacks venue/year columns; skipping for indexing")
            continue
        for row in df.select(["paper_id", "venue", "year"]).iter_rows(named=True):
            pid = row["paper_id"]
            if pid in seen:
                continue
            seen.add(pid)
            out.append((pid, row["venue"], row["year"]))
    return out


def load_facet_map(facets_parquet, text_map=None):
    """paper_id -> {view_name: text}.

    The FULL view falls back to the raw 'title [SEP] abstract' string, so a
    paper with no factorization still participates in retrieval through the
    undecomposed view (graceful degradation, never a crash).
    """
    df = pl.read_parquet(facets_parquet)
    out = {}
    for row in df.iter_rows(named=True):
        pid = row["paper_id"]
        entry = {name: (row.get(name) or "") for name in DENSE_FACETS + SPARSE_FACETS}
        entry[FULL_VIEW] = (text_map or {}).get(pid, "")
        out[pid] = entry
    if text_map:
        for pid, txt in text_map.items():
            if pid not in out:
                out[pid] = {name: "" for name in DENSE_FACETS + SPARSE_FACETS}
                out[pid][FULL_VIEW] = txt
    return out
