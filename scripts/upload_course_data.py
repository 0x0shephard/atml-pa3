"""Upload course_data/ (ATML-PA3-Data) as a private Kaggle dataset on every configured account.

Private datasets are only visible to their owner, so each account gets its own copy under the
same slug; jobs mount <owner>/<course_dataset>. Never passes --public.
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

from scripts.kaggle_runner import ROOT, default_cli, kaggle, read_accounts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", default=default_cli())
    parser.add_argument("--source", type=Path, default=ROOT / "course_data")
    parser.add_argument("--accounts", nargs="*")
    args = parser.parse_args()
    cfg = read_accounts()
    if not any(args.source.rglob("generations.parquet")):
        raise FileNotFoundError(f"No generations.parquet under {args.source}")
    for owner in args.accounts or list(cfg["accounts"]):
        profile = cfg["accounts"][owner].get("profile")
        with tempfile.TemporaryDirectory() as staging:
            staging = Path(staging)
            for item in args.source.iterdir():
                if item.name.startswith("."):
                    continue
                target = staging / item.name
                shutil.copytree(item, target) if item.is_dir() else shutil.copy2(item, target)
            metadata = {"id": f"{owner}/{cfg['course_dataset']}", "title": cfg["course_dataset"].replace("-", " "),
                        "licenses": [{"name": "other"}],
                        "description": "Private copy of the course-supplied ATML PA3 rollout data for coursework jobs."}
            (staging / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))
            print(owner, kaggle(args.cli, ["datasets", "create", "-p", str(staging), "--dir-mode", "zip"], profile, timeout=7200))


if __name__ == "__main__":
    main()
