"""Build private, self-contained Kaggle notebooks and operate separate CLI profiles.

No credential values are ever embedded in source snapshots or notebook metadata.
Each run is self-contained: it prepares its own inputs, so no kernel output is mounted
by another kernel. Runs may split work into GPU lanes that share one T4x2 session.
"""
from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments" / "runs.json"
ACCOUNTS = ROOT / "configs" / "kaggle_accounts.yaml"
SNAPSHOT_DIRS = {"common", "configs", "scripts", "task1_finetune", "task2_vlm", "task3_ttc", "tests", "experiments"}
SNAPSHOT_FILES = {"requirements.txt", "README.md"}
# Kept small evidence only; heavy reproducible inputs are deleted before Kaggle saves output.
CLEANUP = ("checkpoints", "cached", "data", "remote", ".cache")


def read_accounts():
    if not ACCOUNTS.exists():  # Inside a Kaggle job: account settings are local-only.
        return {"accounts": {}, "weekly_gpu_hours": 0, "course_dataset": None}
    cfg = yaml.safe_load(ACCOUNTS.read_text())
    for name, account in cfg["accounts"].items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(f"Invalid Kaggle username {name}")
        if account.get("profile"):
            account["profile"] = str((ROOT / account["profile"]).resolve())
    return cfg


def read_manifest():
    manifest = json.loads(MANIFEST.read_text())
    ids = [run["id"] for run in manifest["runs"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate run IDs")
    accounts = read_accounts()["accounts"]
    lookup = {run["id"]: run for run in manifest["runs"]}
    for run in manifest["runs"]:
        if any(dependency not in ids for dependency in run.get("depends_on", [])):
            raise ValueError(f"Unknown dependency for {run['id']}")
        if accounts and run.get("account") and run["account"] not in accounts:
            raise ValueError(f"Unknown account pin for {run['id']}")
        lanes = run.get("lanes", [])
        if not lanes or not all(isinstance(lane, list) for lane in lanes):
            raise ValueError(f"Run {run['id']} needs a list of lanes")
        if len(lanes) > (2 if run.get("gpu", True) else 1):
            raise ValueError(f"Too many lanes for {run['id']}")
    visiting, visited = set(), set()
    def visit(key):
        if key in visiting:
            raise ValueError("Run dependency cycle")
        if key in visited:
            return
        visiting.add(key)
        for dependency in lookup[key].get("depends_on", []):
            visit(dependency)
        visiting.remove(key)
        visited.add(key)
    for key in ids:
        visit(key)
    return manifest


def find_run(run_id):
    run = next((r for r in read_manifest()["runs"] if r["id"] == run_id), None)
    if run is None:
        raise ValueError(f"Unknown run ID: {run_id}")
    return run


def source_snapshot():
    stream = io.BytesIO()
    # Allowlist avoids including checkpoints, assets, credentials, environments or result records.
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(ROOT.rglob("*")):
            relative = file.relative_to(ROOT)
            if not file.is_file() or file.is_symlink() or any(x.startswith(".") or x == "__pycache__" for x in relative.parts):
                continue
            include = ((relative.parts[0] in SNAPSHOT_DIRS and file.suffix in {".py", ".yaml", ".json"})
                       or str(relative) in SNAPSHOT_FILES)
            if relative.parts[0] == "configs" and relative.name == "kaggle_accounts.yaml":
                include = False  # Local profile paths are not needed remotely.
            if include:
                info = zipfile.ZipInfo(relative.as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, file.read_bytes())
    payload = stream.getvalue()
    return base64.b64encode(payload).decode(), hashlib.sha256(payload).hexdigest()


def notebook_source(run, payload, digest):
    return f'''# Generated from a fixed source snapshot; experiment logic lives in Python modules.
import base64, hashlib, json, os, pathlib, shutil, subprocess, sys, time, zipfile, io
root = pathlib.Path("/kaggle/working/pa3")
root.mkdir(parents=True, exist_ok=True)
payload = base64.b64decode({payload!r})
assert hashlib.sha256(payload).hexdigest() == {digest!r}
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    archive.extractall(root)
os.chdir(root)
os.environ["HF_HOME"] = "/kaggle/temp/pa3-hf-cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MPLCONFIGDIR"] = "/kaggle/temp/pa3-matplotlib"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
def execute(command):
    print("RUN", command, flush=True)
    subprocess.run(command, check=True, cwd=root)
started = time.time()
(root / "job_status.json").write_text(json.dumps({{"run_id": {run['id']!r}, "status": "running", "source_sha256": {digest!r}}}))
try:
    # Resume only a matching run/source snapshot from the previous attempt's saved output.
    if {run.get('resume', False)!r}:
        previous = []
        for status_file in pathlib.Path("/kaggle/input").rglob("job_status.json"):
            status = json.loads(status_file.read_text())
            if status.get("run_id") == {run['id']!r}:
                # A different snapshot is accepted only when named explicitly at build time (e.g. an evaluation-only fix).
                if status.get("source_sha256") not in {[digest] + ([run["resume_source"]] if run.get("resume_source") else [])!r}:
                    raise RuntimeError("Resume source snapshot differs")
                previous.append(status_file.parent)
        if len(previous) != 1: raise RuntimeError("Need exactly one previous output for this run")
        for prefix in ("outputs", "results", "manifests", "remote", "data"):
            if (previous[0] / prefix).exists():
                shutil.copytree(previous[0] / prefix, root / prefix, dirs_exist_ok=True)
    execute([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    execute([sys.executable, "-m", "scripts.check_environment"])
    execute([sys.executable, "-m", "unittest", "discover", "-s", "tests"])
    execute([sys.executable, "-m", "scripts.download_assets"])
    execute([sys.executable, "-m", "scripts.run_lanes", {run['id']!r}])
    missing = [p for p in {run['artifacts']!r} if not (root / p).exists()]
    if missing: raise RuntimeError(f"Job finished without required artifacts: {{missing}}")
    # Only after success: a failed session keeps recovery inputs for a resumed attempt.
    for name in {CLEANUP!r}:
        shutil.rmtree(root / name, ignore_errors=True)
    for state in (root / "outputs").rglob("training_state.pt"):
        state.unlink()
    (root / "job_status.json").write_text(json.dumps({{"run_id": {run['id']!r}, "status": "complete", "source_sha256": {digest!r}, "resume_source_sha256": {run.get("resume_source")!r}, "wall_clock_seconds": time.time() - started}}))
except BaseException as exc:
    (root / "job_status.json").write_text(json.dumps({{"run_id": {run['id']!r}, "status": "failed", "error": str(exc)[-4000:], "source_sha256": {digest!r}, "resume_source_sha256": {run.get("resume_source")!r}, "wall_clock_seconds": time.time() - started}}))
    raise
'''


def kernel_slug(run_id, attempt=1):
    title = f"PA3 {run_id.replace('_', ' ')}" + (f" attempt {attempt}" if attempt > 1 else "")
    return title, title.lower().replace(" ", "-")


def build(run_id, username, resume_kernel=None, attempt=1, resume_source=None, resume_dataset=None):
    accounts = read_accounts()
    if username not in accounts["accounts"]:
        raise ValueError(f"Unknown account {username}")
    run = find_run(run_id)
    if resume_kernel and resume_dataset:
        raise ValueError("Resume from a kernel or a dataset, not both")
    if (resume_kernel or resume_dataset) and attempt < 2:
        raise ValueError("Use --attempt 2 (or higher) with --resume-kernel to avoid self-referencing the prior notebook")
    if resume_source and not (resume_kernel or resume_dataset):
        raise ValueError("--resume-source needs --resume-kernel or --resume-dataset")
    run = {**run, "resume": bool(resume_kernel or resume_dataset), "resume_source": resume_source}
    payload, digest = source_snapshot()
    title, slug = kernel_slug(run_id, attempt)
    directory = ROOT / "kaggle_jobs" / username / run_id
    if attempt > 1:
        directory = directory / f"attempt_{attempt}"
    directory.mkdir(parents=True, exist_ok=True)
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                "cells": [{"cell_type": "code", "id": "launch", "metadata": {}, "execution_count": None,
                           "outputs": [], "source": notebook_source(run, payload, digest).splitlines(True)}]}
    (directory / "run.ipynb").write_text(json.dumps(notebook, indent=2))
    datasets = [f"{username}/{accounts['course_dataset']}"] if run.get("course_data") else []
    if resume_dataset:
        datasets.append(resume_dataset)
    metadata = {"id": f"{username}/{slug}", "title": title, "code_file": "run.ipynb", "language": "python",
                "kernel_type": "notebook", "is_private": True, "enable_gpu": run.get("gpu", True),
                "enable_internet": True, "dataset_sources": datasets,
                "kernel_sources": [resume_kernel] if resume_kernel else [],
                "competition_sources": [], "model_sources": []}
    if run.get("gpu", True):
        metadata["machine_shape"] = "NvidiaTeslaT4"
    (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    (directory / "snapshot.json").write_text(json.dumps({"sha256": digest, "run_id": run_id}, indent=2))
    return directory


def cli_launcher(cli, profile):
    """Run the Kaggle CLI; a non-default profile goes through the credential-isolating wrapper."""
    env = os.environ.copy()
    if profile:
        env["PA3_KAGGLE_PROFILE"] = profile
        env["KAGGLE_CONFIG_DIR"] = profile
        return [str(Path(cli).parent / "python"), str(ROOT / "scripts/kaggle_profile_cli.py")], env
    return [cli], env


def kaggle(cli, args, profile=None, timeout=1800):
    launcher, env = cli_launcher(cli, profile)
    result = subprocess.run(launcher + args, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    output = result.stdout + result.stderr
    if result.returncode:
        raise RuntimeError(output[-2000:])
    return output


def push_args(directory, gpu):
    args = ["kernels", "push", "-p", str(directory), "-t", "43200"]
    if gpu:
        args += ["--accelerator", "NvidiaTeslaT4"]
    return args


def iso_week(timestamp):
    year, week, _ = datetime.datetime.fromisoformat(timestamp).isocalendar()
    return f"{year}-W{week:02d}"


def gpu_hours_used(owner, week):
    """GPU-hour ledger from local receipts (completed and failed GPU sessions)."""
    total = 0.0
    for receipt in (ROOT / "results" / "jobs").glob("*.json"):
        data = json.loads(receipt.read_text())
        if data.get("owner") == owner and data.get("gpu") and data.get("finished") and iso_week(data["finished"]) == week:
            total += data.get("wall_clock_seconds", 0.0) / 3600
    return total


def import_output(directory, run_id, owner=None, kernel=None):
    run = find_run(run_id)
    candidates = list(Path(directory).rglob("job_status.json"))
    statuses = [(file, json.loads(file.read_text())) for file in candidates]
    matches = [(file, status) for file, status in statuses if status.get("run_id") == run_id]
    if len(matches) != 1:
        raise ValueError("Need exactly one matching job_status.json")
    status_path, status = matches[0]
    if status["status"] != "complete":
        raise ValueError(f"Job was not successful: {status}")
    source_root = status_path.parent
    if any(not (source_root / artifact).exists() for artifact in run["artifacts"]):
        raise ValueError("Downloaded job is missing required artifacts")
    files = [file for prefix in ("outputs", "results", "manifests") for file in (source_root / prefix).rglob("*") if file.is_file()]
    for file in files:
        if file.is_symlink():
            raise ValueError("Downloaded artifacts must not contain symlinks")
        target = ROOT / file.relative_to(source_root)
        if target.exists() and target.read_bytes() != file.read_bytes():
            raise FileExistsError(f"Conflicting artifact: {target}; preserve separate run versions")
    for file in files:
        target = ROOT / file.relative_to(source_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)
    write_receipt(run_id, status, owner, kernel, run.get("gpu", True))
    return len(files)


def write_receipt(run_id, status, owner, kernel, gpu):
    receipt = ROOT / "results" / "jobs" / f"{run_id}.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    if status.get("status") != "complete":  # Failed sessions keep separate receipts (still counted for quota).
        receipt = receipt.with_name(f"{run_id}.{status.get('status', 'failed')}.{datetime.datetime.now():%Y%m%d%H%M%S}.json")
    record = {**status, "owner": owner, "kernel": kernel, "gpu": gpu,
              "finished": datetime.datetime.now().astimezone().isoformat()}
    receipt.write_text(json.dumps(record, indent=2))


def error_pages(directory):
    """Files the CLI saved from an HTML error response instead of the real content."""
    return [f for f in directory.rglob("*") if f.is_file()
            and f.read_bytes()[:64].lstrip().lower().startswith((b"<html", b"<!doctype"))]


def fetch_clean(cli, kernel, directory, pattern, profile, tries=3):
    kaggle(cli, ["kernels", "output", kernel, "-p", str(directory), "--file-pattern", pattern, "-q"], profile, timeout=7200)
    for _ in range(tries):
        bad = error_pages(directory)
        if not bad:
            break
        for f in bad:
            relative = f.relative_to(directory).as_posix().removeprefix("pa3/")
            kaggle(cli, ["kernels", "output", kernel, "-p", str(directory), "--file-pattern",
                         ".*" + re.escape(relative) + "$", "-q"], profile, timeout=7200)
    return error_pages(directory)


def stage_resume(cli, run_id, kernel, attempt, profile, fill_from=None):
    """Kaggle refuses errored notebooks as kernel sources, so a failed attempt's recovery files are
    re-uploaded as a private dataset owned by the same account and mounted by the next attempt."""
    owner = kernel.split("/")[0]
    slug = f"pa3-{run_id.replace('_', '-')}-resume-a{attempt}"
    directory = ROOT / "kaggle_downloads" / f"{run_id}_resume_a{attempt}"
    shutil.rmtree(directory, ignore_errors=True)
    bad = fetch_clean(cli, kernel, directory, ".*(outputs/|results/|manifests/|remote/|job_status).*", profile)
    for f in bad:  # Bad on the kernel itself (e.g. copied from an earlier corrupt staging): refetch from fill_from.
        if not fill_from:
            break
        relative = f.relative_to(directory).as_posix().removeprefix("pa3/")
        f.unlink()
        fetch_clean(cli, fill_from, directory, ".*" + re.escape(relative) + "$", profile)
    for log in directory.glob("*.log"):  # Kernel logs are not recovery inputs.
        log.unlink()
    bad = [f for f in error_pages(directory) if f.suffix != ".log"]
    if bad:
        raise RuntimeError(f"Downloads are HTML error pages: {bad}")
    for f in error_pages(directory):  # A corrupt lane log is not worth blocking on.
        f.unlink()
    status = json.loads((directory / "pa3" / "job_status.json").read_text())
    if status.get("run_id") != run_id:
        raise ValueError(f"{kernel} output belongs to {status.get('run_id')}")
    metadata = {"id": f"{owner}/{slug}", "title": slug.replace("-", " "), "licenses": [{"name": "other"}],
                "description": f"Private recovery files from {kernel} for resuming {run_id}."}
    (directory / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))
    kaggle(cli, ["datasets", "create", "-p", str(directory), "--dir-mode", "zip"], profile, timeout=7200)
    return {"dataset": f"{owner}/{slug}", "source_sha256": status.get("source_sha256")}


def default_cli():
    candidate = ROOT.parent / "PA2" / ".kaggle-cli" / "bin" / "kaggle"
    return str(candidate) if candidate.exists() else (shutil.which("kaggle") or "kaggle")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cli", default=default_cli())
    sub = ap.add_subparsers(dest="action", required=True)
    sub.add_parser("list")
    build_parser = sub.add_parser("build")
    build_parser.add_argument("run_id")
    build_parser.add_argument("--username", required=True)
    build_parser.add_argument("--resume-kernel")
    build_parser.add_argument("--attempt", type=int, default=1)
    build_parser.add_argument("--resume-source", help="Also accept a previous output from this source sha256")
    build_parser.add_argument("--resume-dataset", help="owner/slug from stage-resume (failed kernels cannot be mounted)")
    staged = sub.add_parser("stage-resume", help="Copy a failed kernel's outputs into a private dataset for resuming")
    staged.add_argument("run_id")
    staged.add_argument("kernel")
    staged.add_argument("--attempt", type=int, required=True, help="The attempt that will resume from it")
    staged.add_argument("--fill-from", help="Earlier kernel to refetch files that are corrupt on this one")
    push = sub.add_parser("push")
    push.add_argument("directory", type=Path)
    for name in ("status", "output"):
        p = sub.add_parser(name)
        p.add_argument("kernel")
        if name == "output":
            p.add_argument("--directory", type=Path, required=True)
    imported = sub.add_parser("import")
    imported.add_argument("run_id")
    imported.add_argument("directory", type=Path)
    imported.add_argument("--kernel")
    sub.add_parser("quota")
    args = ap.parse_args()
    accounts = read_accounts()
    profile_of = lambda owner: accounts["accounts"][owner].get("profile")
    if args.action == "list":
        for run in read_manifest()["runs"]:
            print(run["id"], "account=" + str(run.get("account")), "gpu=" + str(run.get("gpu", True)),
                  "est_h=" + str(run.get("est_gpu_hours")), "depends=" + ",".join(run.get("depends_on", [])))
    elif args.action == "build":
        print(build(args.run_id, args.username, args.resume_kernel, args.attempt, args.resume_source, args.resume_dataset))
    elif args.action == "stage-resume":
        owner = args.kernel.split("/")[0]
        print(stage_resume(args.cli, args.run_id, args.kernel, args.attempt, profile_of(owner), args.fill_from))
    elif args.action == "push":
        metadata = json.loads((args.directory / "kernel-metadata.json").read_text())
        if metadata.get("is_private") is not True:
            raise ValueError("Runner only submits private notebooks")
        print(kaggle(args.cli, push_args(args.directory, metadata.get("enable_gpu")), profile_of(metadata["id"].split("/")[0])))
    elif args.action in ("status", "output"):
        owner = args.kernel.split("/")[0]
        command = ["kernels", "status", args.kernel] if args.action == "status" else \
            ["kernels", "output", args.kernel, "-p", str(args.directory)]
        print(kaggle(args.cli, command, profile_of(owner)))
    elif args.action == "import":
        owner = args.kernel.split("/")[0] if args.kernel else None
        print("Imported files:", import_output(args.directory, args.run_id, owner, args.kernel))
    elif args.action == "quota":
        week = iso_week(datetime.datetime.now().astimezone().isoformat())
        for owner in accounts["accounts"]:
            used = gpu_hours_used(owner, week)
            print(owner, week, f"used={used:.2f}h", f"limit={accounts['weekly_gpu_hours']}h")


if __name__ == "__main__":
    main()
