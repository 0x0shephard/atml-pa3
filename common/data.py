from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: str | Path) -> dict:
    path = repo_path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg.get("base_config"):
        base = load_yaml(cfg["base_config"])
        cfg = _deep_merge(base, {k: v for k, v in cfg.items() if k != "base_config"})
    return cfg


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with repo_path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    p = repo_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_suffix(p.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(p)


def course_data_dir() -> Path:
    """Locate the supplied ATML-PA3-Data folder locally or as a mounted Kaggle dataset."""
    candidates = []
    if os.environ.get("PA3_COURSE_DATA"):
        candidates.append(Path(os.environ["PA3_COURSE_DATA"]))
    candidates.append(repo_path("course_data"))
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        candidates.append(kaggle)  # The supplied README.MD below marks the dataset root.
    for candidate in candidates:
        if candidate.exists() and any(candidate.rglob("generations.parquet")):
            # The supplied README sits beside the per-model folders.
            readmes = sorted(candidate.rglob("README.MD")) + sorted(candidate.rglob("README.md"))
            return readmes[0].parent if readmes else candidate
    raise FileNotFoundError("ATML-PA3-Data not found; place it in course_data/ or set PA3_COURSE_DATA")
