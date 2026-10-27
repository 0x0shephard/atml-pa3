"""Check that every non-smoke manifest artifact and job receipt exists locally before reporting."""
import argparse
import json
import sys

from scripts.kaggle_runner import ROOT, read_manifest


def validate():
    issues = []
    for run in read_manifest()["runs"]:
        if run["id"] == "smoke":
            continue
        receipt = ROOT / "results" / "jobs" / f"{run['id']}.json"
        if not receipt.exists() or json.loads(receipt.read_text()).get("status") != "complete":
            issues.append(f"{run['id']}: no completed job receipt")
        issues += [f"{run['id']}: missing {a}" for a in run["artifacts"] if not (ROOT / a).exists()]
    for name in ("full_ft", "l2sp", "lora4", "lora16", "oplora16"):
        path = ROOT / "results" / "task1" / f"{name}_run.json"
        if path.exists():
            data = json.loads(path.read_text())
            if data.get("updates") != 1250 or data.get("examples") != 40000 or data.get("smoke"):
                issues.append(f"task1 {name}: budget is not 1,250 updates / 40,000 examples")
    for name in ("vlm196", "vlm49"):
        path = ROOT / "results" / "task2" / name / f"{name}_run.json"
        if path.exists() and json.loads(path.read_text()).get("example_presentations") != 72000:
            issues.append(f"task2 {name}: budget is not 72,000 example presentations")
    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    issues = validate()
    print(json.dumps({"complete": not issues, "issues": issues}, indent=2))
    if issues and not args.allow_incomplete:
        sys.exit(1)


if __name__ == "__main__":
    main()
