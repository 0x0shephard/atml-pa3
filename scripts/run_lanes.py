"""Run a manifest entry's setup commands, then its lanes in parallel, one GPU per lane.

Kaggle bills a T4x2 session by wall-clock, so two independent experiments share one
session. Each lane runs its commands sequentially with CUDA_VISIBLE_DEVICES pinned and
writes its own log under results/jobs/lanes/. The job fails if any lane fails.
"""
import argparse
import os
import subprocess
import sys
import time

from scripts.kaggle_runner import ROOT, find_run


def run_lane(index, commands, log_dir, gpu):
    env = os.environ.copy()
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(index)
    log = (log_dir / f"lane{index}.log").open("a")
    script = " && ".join(" ".join([sys.executable] + command) for command in commands)
    return subprocess.Popen(["bash", "-o", "pipefail", "-c", f"({script}) 2>&1 | tee -a {log.name}"],
                            cwd=ROOT, env=env), log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    args = parser.parse_args()
    run = find_run(args.run_id)
    log_dir = ROOT / "results" / "jobs" / "lanes" / args.run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    for command in run.get("setup", []):
        print("SETUP", command, flush=True)
        subprocess.run([sys.executable] + command, check=True, cwd=ROOT)
    started = time.time()
    processes = [run_lane(i, lane, log_dir, run.get("gpu", True)) for i, lane in enumerate(run["lanes"]) if lane]
    failures = []
    for index, (process, log) in enumerate(processes):
        code = process.wait()
        log.close()
        print(f"lane {index} exited {code} after {time.time() - started:.0f}s", flush=True)
        if code:
            failures.append(index)
    if failures:
        raise SystemExit(f"Lanes failed: {failures}")


if __name__ == "__main__":
    main()
