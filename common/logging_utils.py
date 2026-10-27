from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common.data import repo_path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_parent(path: str | Path) -> Path:
    p = repo_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    p = ensure_parent(path)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_json(path: str | Path, obj: Any) -> None:
    p = ensure_parent(path)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def wall_timer():
    start = time.perf_counter()
    return lambda: time.perf_counter() - start
