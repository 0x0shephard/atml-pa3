"""Hourly, restartable controller for the private PA3 experiment queue across Kaggle accounts."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import time

from scripts.kaggle_runner import (ROOT, build, default_cli, gpu_hours_used, import_output, iso_week, kaggle,
                                   push_args, read_accounts, read_manifest, write_receipt)

STATE_DIR = ROOT / "kaggle_jobs" / "autopilot"
ACTIVE = ("submitted", "submitting", "download")


def log(message):
    print(datetime.datetime.now().astimezone().isoformat(), message, flush=True)


def tick(state, cli, only=None):
    cfg = read_accounts()
    accounts = cfg["accounts"]
    runs = [r for r in read_manifest()["runs"] if only is None or r["id"] in only]
    jobs = state["jobs"]
    def save():
        temp = STATE_DIR / "state.tmp"
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(STATE_DIR / "state.json")
    for run in runs:
        if run["id"] not in jobs:
            receipt = ROOT / "results" / "jobs" / f"{run['id']}.json"
            done = receipt.exists() and json.loads(receipt.read_text()).get("status") == "complete"
            jobs[run["id"]] = {"status": "complete" if done else "pending"}
    save()
    for run in runs:
        key, job = run["id"], jobs[run["id"]]
        if job["status"] not in ACTIVE:
            continue
        owner = job["kernel"].split("/")[0]
        profile = accounts[owner].get("profile")
        try:
            output = kaggle(cli, ["kernels", "status", job["kernel"]], profile)
            match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", output) or re.search(r'status "?([A-Za-z_]+)"?', output)
            if not match:
                raise RuntimeError("Unrecognized status: " + output)
            status = match[1].upper()
            log(f"{key}: {status}")
            job["remote_status"] = status
            if status == "COMPLETE":
                job["status"] = "download"
                save()
                directory = ROOT / "kaggle_downloads" / (key + "_auto")
                kaggle(cli, ["kernels", "output", job["kernel"], "-p", str(directory),
                             "--file-pattern", ".*(outputs/|results/|manifests/|job_status).*", "-q"], profile)
                import_output(directory, key, owner, job["kernel"])
                job["status"] = "complete"
                job.pop("error", None)
                log(f"{key}: imported successfully")
            elif status in ("ERROR", "CANCELLED", "CANCELED", "CANCEL_ACKNOWLEDGED"):
                job["status"] = "failed"
                save()
                # One folder per attempt, emptied first, so a stale job_status is never re-read.
                directory = ROOT / "kaggle_downloads" / f"{key}_failed_auto_a{job.get('attempt', 1)}"
                shutil.rmtree(directory, ignore_errors=True)
                kaggle(cli, ["kernels", "output", job["kernel"], "-p", str(directory),
                             "--file-pattern", ".*(job_status|lanes/).*", "-q"], profile)
                statuses = list(directory.rglob("job_status.json"))
                failure = json.loads(statuses[0].read_text()) if statuses else {"run_id": key, "status": "failed",
                                                                                  "error": "no job_status.json in output"}
                write_receipt(key, failure, owner, job["kernel"], run.get("gpu", True))
        except Exception as exc:
            job["error"] = str(exc)[-2000:]
            log(f"{key}: {exc}")
        save()
    active = [r for r in runs if jobs[r["id"]]["status"] in ACTIVE]
    counts = {owner: {True: 0, False: 0} for owner in accounts}
    for r in active:
        counts[jobs[r["id"]]["kernel"].split("/")[0]][r.get("gpu", True)] += 1
    week = iso_week(datetime.datetime.now().astimezone().isoformat())
    # Resumed attempts of failed jobs, added by hand once the failure is fixed:
    # {run_id: {"attempt": 2, "resume_dataset" | "resume_kernel": "owner/slug", "resume_source": sha256 | null,
    #           "est_gpu_hours": h}}  (failed kernels must go through `kaggle_runner stage-resume` first)
    # A fresh rerun names {"attempt": 2, "account": owner} with no resume source.
    retries_path = STATE_DIR / "retries.json"
    retries = json.loads(retries_path.read_text()) if retries_path.exists() else {}
    for run in runs:
        key, job = run["id"], jobs[run["id"]]
        retry = retries.get(key)
        if retry and job["status"] == "failed" and retry["attempt"] > job.get("attempt", 1):
            # Kernel outputs are private, so a resumed attempt runs on the account that owns them.
            resume = retry.get("resume_dataset") or retry.get("resume_kernel")
            run = {**run, "account": resume.split("/")[0] if resume else retry.get("account", run.get("account")),
                   "est_gpu_hours": retry.get("est_gpu_hours", run.get("est_gpu_hours", 0))}
        elif job["status"] != "pending":
            continue
        else:
            retry = None
        if any(jobs.get(d, {}).get("status") != "complete" for d in run.get("depends_on", [])):
            continue
        if run.get("hold"):  # Waiting on inputs the user must supply (e.g. course data).
            continue
        gpu = run.get("gpu", True)
        candidates = [run["account"]] if run.get("account") else list(accounts)
        available = []
        for owner in candidates:
            if counts[owner][gpu] >= (2 if gpu else 1):
                continue
            # Count running sessions as already spent so concurrent jobs cannot overrun the weekly quota.
            reserved = sum(r.get("est_gpu_hours", 0) for r in active if jobs[r["id"]]["kernel"].split("/")[0] == owner)
            if gpu and gpu_hours_used(owner, week) + reserved + run.get("est_gpu_hours", 0) > cfg["weekly_gpu_hours"]:
                log(f"{key}: {owner} lacks GPU quota this week")
                continue
            available.append(owner)
        if not available:
            continue
        owner = min(available, key=lambda a: (counts[a][gpu], gpu_hours_used(a, week)))
        profile = accounts[owner].get("profile")
        try:
            directory = (build(key, owner, retry.get("resume_kernel"), retry["attempt"], retry.get("resume_source"),
                               retry.get("resume_dataset")) if retry else build(key, owner))
            metadata = json.loads((directory / "kernel-metadata.json").read_text())
            job.update(status="submitting", kernel=metadata["id"], attempt=retry["attempt"] if retry else 1)
            save()  # Crash-safe: query this slug before any further submission.
            output = kaggle(cli, push_args(directory, gpu), profile)
            # Kaggle still pushes when it drops an input it cannot mount; treat that as a rejection.
            if ("successfully pushed" not in output.lower() and "kernel version" not in output.lower()) \
                    or "not valid" in output.lower():
                job.update(status="pending", error=output)
                log(f"{key}: submission rejected: {output}")
                save()
                break
            job["status"] = "submitted"
            job.pop("error", None)
            log(f"{key}: submitted {job['kernel']}")
            counts[owner][gpu] += 1
            active.append(run)
        except Exception as exc:
            # Ambiguous submission stays 'submitting' to prevent duplicate pushes.
            job["error"] = str(exc)[-2000:]
            log(f"{key}: {exc}")
            save()
            break
        save()
    state["last_poll"] = datetime.datetime.now().astimezone().isoformat()
    save()
    return all(jobs[r["id"]]["status"] in ("complete", "failed") or r.get("hold") for r in runs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", default=default_cli())
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--only", nargs="*", help="Restrict to these run IDs (for staged queues)")
    parser.add_argument("--interval", type=int, default=3600)
    args = parser.parse_args()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    lock = (STATE_DIR / "lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (STATE_DIR / "pid").write_text(str(os.getpid()))
    path = STATE_DIR / "state.json"
    state = json.loads(path.read_text()) if path.exists() else {"jobs": {}}
    while True:
        try:
            if tick(state, str(Path(args.cli).resolve()), set(args.only) if args.only else None):
                log("Queue finished.")
                break
        except Exception as exc:
            log(f"Controller error: {exc}")
        if args.once:
            break
        log(f"Next poll in {args.interval}s")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
