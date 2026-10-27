"""Run provenance, atomic checkpoints, and numerical safeguards shared by tasks."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from common.data import repo_path

PACKAGES = ("torch", "transformers", "peft", "tokenizers", "accelerate", "datasets", "lm_eval", "math-verify")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def source_fingerprint():
    source = hashlib.sha256()
    root = repo_path(".")
    for file in sorted(root.rglob("*.py")):
        relative = file.relative_to(root)
        if any(part.startswith(".") or part in {"outputs", "remote", "hf_cache", "kaggle_downloads", "kaggle_jobs"}
               for part in relative.parts):
            continue
        source.update(str(relative).encode())
        source.update(file.read_bytes())
    return source.hexdigest()


def package_versions():
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_path("."),
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def atomic_json(path, value):
    path = repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def completed(output):
    path = repo_path(output) / "run.json"
    return path.exists() and json.loads(path.read_text()).get("status") == "complete"


def trainable_state(model):
    return {name: param.detach().cpu().clone() for name, param in model.named_parameters()
            if param.requires_grad}


def restore_parameters(model, state):
    params = dict(model.named_parameters())
    expected = {name for name, param in params.items() if param.requires_grad}
    if expected != set(state):
        raise ValueError("Checkpoint trainable parameter names differ from the current model")
    with torch.no_grad():
        for name, value in state.items():
            params[name].copy_(value.to(params[name].device))


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Exact training resume requires the same GPU count")
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def file_bytes(path):
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


class Run:
    """One resumable training run.

    Every update is logged to ``train.jsonl``; full recovery state (parameters, optimizer,
    scheduler, scaler, RNG, data position) is written only at the configured checkpoint
    updates, overwriting the previous one so Kaggle output stays small. Sizes of every
    checkpoint are recorded in ``run.json``.
    """

    def __init__(self, output, cfg, example_ids, name, resume=False):
        self.output = repo_path(output)
        self.path = self.output / "training_state.pt"
        self.cfg = cfg
        self.run_id = fingerprint({"config": cfg, "examples": fingerprint(example_ids), "name": name})
        self.start = time.perf_counter()
        self.elapsed = 0.0
        self.previous_peaks = {}
        self.state = None
        if self.path.exists():
            if not resume:
                raise FileExistsError(f"Existing run at {self.output}; use --resume")
            self.state = torch.load(self.path, map_location="cpu", weights_only=False)
            if self.state["run_id"] != self.run_id:
                raise ValueError("Resume config or input data differs from saved run")
            self.elapsed = self.state.get("elapsed_seconds", 0.0)
            self.previous_peaks = self.state.get("peak_vram_bytes", {})
        elif self.output.exists() and any(self.output.iterdir()):
            if not resume:
                raise FileExistsError(f"Output is nonempty without a resumable checkpoint: {self.output}")
            shutil.rmtree(self.output)
        self.output.mkdir(parents=True, exist_ok=True)
        previous = []
        if self.state and (self.output / "run.json").exists():
            previous = json.loads((self.output / "run.json").read_text()).get("checkpoints", [])
        self.metadata = {"name": name, "run_id": self.run_id, "status": "running", "config": cfg,
                         "source_commit": git_commit(), "source_sha256": source_fingerprint(),
                         "versions": package_versions(), "examples_sha256": fingerprint(example_ids),
                         "num_examples": len(example_ids),
                         "checkpoints": [c for c in previous if self.state and c["update"] <= self.state["progress"]["update"]]}
        if self.state and self.state.get("source_sha256") != self.metadata["source_sha256"]:
            raise ValueError("Source changed since checkpoint; start a new run instead of mixing code versions")
        reset_peaks()
        atomic_json(self.output / "run.json", self.metadata)

    def seconds(self):
        if torch.cuda.is_initialized():
            for device in range(torch.cuda.device_count()):
                torch.cuda.synchronize(device)
        return self.elapsed + time.perf_counter() - self.start

    def peaks(self):
        if not torch.cuda.is_initialized():
            return dict(self.previous_peaks)
        return {str(i): max(self.previous_peaks.get(str(i), 0), torch.cuda.max_memory_allocated(i))
                for i in range(torch.cuda.device_count())}

    def restore(self, models, optimizers, schedulers, scaler):
        if not self.state:
            return {"update": 0, "cursor": 0}
        for name, model in models.items():
            restore_parameters(model, self.state["models"][name])
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(self.state["optimizers"][name])
        for name, scheduler in schedulers.items():
            scheduler.load_state_dict(self.state["schedulers"][name])
        scaler.load_state_dict(self.state["scaler"])
        restore_rng(self.state["rng"])
        # A crash can leave log rows beyond the last atomic checkpoint.
        log = self.output / "train.jsonl"
        if log.exists():
            rows = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
            rows = [x for x in rows if x["update"] <= self.state["progress"]["update"]]
            log.write_text("".join(json.dumps(x) + "\n" for x in rows))
        return self.state["progress"]

    def log(self, record):
        # Overflowed (skipped) updates report non-finite gradient norms; store them as null.
        record = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in record.items()}
        with (self.output / "train.jsonl").open("a") as file:
            file.write(json.dumps(record, allow_nan=False) + "\n")

    def save(self, models, optimizers, schedulers, scaler, progress, extra=None):
        seconds = self.seconds()
        state = {"run_id": self.run_id, "source_sha256": self.metadata["source_sha256"],
                 # Device tensors are serialized storage by storage, avoiding a full host copy.
                 "models": {k: {n: p.detach() for n, p in v.named_parameters() if p.requires_grad}
                            for k, v in models.items()},
                 "optimizers": {k: v.state_dict() for k, v in optimizers.items()},
                 "schedulers": {k: v.state_dict() for k, v in schedulers.items()},
                 "scaler": scaler.state_dict(), "rng": rng_state(), "progress": progress,
                 "elapsed_seconds": seconds, "peak_vram_bytes": self.peaks(), **(extra or {})}
        started = time.perf_counter()
        if self.cfg.get("train", {}).get("replace_checkpoint_in_place"):
            # Large full-model states: drop the old one first so two never coexist on Kaggle's
            # 20 GB output disk. A crash during this write falls back to the previous session.
            self.path.unlink(missing_ok=True)
            torch.save(state, self.path)
        else:
            temporary = self.path.with_suffix(".tmp")
            torch.save(state, temporary)
            os.replace(temporary, self.path)
        record = {"update": progress["update"], "examples_seen": progress["cursor"],
                  "bytes": self.path.stat().st_size, "save_seconds": time.perf_counter() - started}
        self.metadata["checkpoints"].append(record)
        atomic_json(self.output / "run.json", self.metadata)
        print(json.dumps({"checkpoint": record}), flush=True)

    def finish(self, results_dir, **extra):
        final = {**self.metadata, **extra, "status": "complete",
                 "wall_clock_seconds": self.seconds(), "peak_vram_bytes": self.peaks()}
        atomic_json(self.output / "run.json", final)
        results = repo_path(results_dir)
        results.mkdir(parents=True, exist_ok=True)
        atomic_json(results / f"{self.metadata['name']}_run.json", final)
        if (self.output / "train.jsonl").exists():
            shutil.copyfile(self.output / "train.jsonl", results / f"{self.metadata['name']}_train.jsonl")
        return final


def optimizer_step(parameters, optimizer, scaler, max_norm, extra_grad=None):
    """Unscale, optionally add an analytic gradient term, clip, and step.

    Returns (grad_norm, skipped). With dynamic loss scaling an overflow skips the update
    instead of failing; callers log it.
    """
    scaler.unscale_(optimizer)
    if extra_grad is not None:
        extra_grad()
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    scale_before = scaler.get_scale() if scaler.is_enabled() else 1.0
    scaler.step(optimizer)
    scaler.update()
    skipped = bool(scaler.is_enabled() and scaler.get_scale() < scale_before)
    if not scaler.is_enabled() and not torch.isfinite(norm):
        raise FloatingPointError("Nonfinite gradient without loss scaling")
    return float(norm), skipped


def reset_peaks():
    """Peak-memory counters exist only once CUDA is initialized; resetting earlier raises."""
    if torch.cuda.is_initialized():
        for device in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device)


def cuda_peaks():
    if not torch.cuda.is_initialized():
        return {}
    return {str(i): {"allocated": torch.cuda.max_memory_allocated(i), "reserved": torch.cuda.max_memory_reserved(i)}
            for i in range(torch.cuda.device_count())}


def numeric_stats(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"mean": None, "std": None, "count": 0}
    return {"mean": float(values.mean()), "std": float(values.std()), "count": len(values)}
