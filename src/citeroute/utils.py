"""Logging and small shared helpers."""

import json
import time
from pathlib import Path


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def banner(msg, char="=", width=80):
    log(char * width)
    log(msg)
    log(char * width)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class Timer:
    """Context manager that logs elapsed wall time."""

    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        log(f"  {self.label} took {time.time() - self.t0:.1f}s")
        return False
