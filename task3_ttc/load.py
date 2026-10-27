"""Load and join the supplied rollout tables, verify token alignment, and fix the question splits."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd

from common.data import course_data_dir, load_yaml, repo_path
from common.runtime import atomic_json

KEYS = ["dataset", "sample_id", "rollout_id"]


def dataset_folder(cfg, name):
    root = course_data_dir()
    hints = [h.lower() for h in cfg["datasets"][name]["folder_hint"]]
    folders = sorted({p.parent for p in root.rglob("generations.parquet")})
    matches = [f for f in folders if any(h in str(f.relative_to(root)).lower() for h in hints)]
    if name == "math":
        matches = [f for f in matches if not (f / "attention_metrics.parquet").exists()]
    if name == "mathvista":
        matches = [f for f in matches if (f / "attention_metrics.parquet").exists()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Ambiguous or missing folder for {name}: {matches}")
    return matches[0]


def resolve(frame, cfg, logical, required=True):
    for candidate in cfg["schema"][logical]:
        if candidate in frame.columns:
            return candidate
    if required:
        raise KeyError(f"No column for {logical}; tried {cfg['schema'][logical]} in {list(frame.columns)}")
    return None


@dataclass
class Question:
    qid: str
    rollout_ids: np.ndarray
    lengths: np.ndarray
    answers: list
    gold: object
    question: str | None
    kl: list            # per-rollout token arrays of D_KL(U || p)
    logp: list          # per-rollout token arrays of sampled-token log-probability
    vis: list | None = None   # per-rollout token arrays of normalized visual engagement v_t
    text: list = field(default_factory=list)
    groups: np.ndarray | None = None
    correct: np.ndarray | None = None
    no_answer: np.ndarray | None = None

    @cached_property
    def L(self):
        return np.array([x.mean() if len(x) else -np.inf for x in self.logp])

    @cached_property
    def K(self):
        return np.array([x.mean() if len(x) else -np.inf for x in self.kl])

    @cached_property
    def V(self):
        return np.array([x.mean() if len(x) else 0.0 for x in self.vis])


def as_array(value):
    return np.asarray(value if value is not None else [], dtype=np.float64)


def load_dataset(cfg, name, keep_text=True):
    folder = dataset_folder(cfg, name)
    gen = pd.read_parquet(folder / "generations.parquet")
    prob = pd.read_parquet(folder / "probability_metrics.parquet")
    frames = [gen, prob]
    attention = None
    if (folder / "attention_metrics.parquet").exists():
        attention = pd.read_parquet(folder / "attention_metrics.parquet")
        frames.append(attention)
    keys = [resolve(gen, cfg, k) for k in ("dataset", "sample_id", "rollout_id")]
    merged = gen
    for frame in frames[1:]:
        extra = [c for c in frame.columns if c not in merged.columns or c in keys]
        merged = merged.merge(frame[extra], on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(gen):
        raise ValueError(f"Join lost rows: {len(gen)} generations, {len(merged)} joined")
    col = {k: resolve(merged, cfg, k, required=k in ("sample_id", "rollout_id", "logp", "kl_u_p", "extracted", "gold"))
           for k in cfg["schema"]}
    report = {"folder": str(folder.name), "rows": len(merged), "columns": list(merged.columns), "resolved": col,
              "alignment": {"checked": 0, "token_minus_measure": {}, "mismatch": 0}}
    questions = {}
    for qid, rows in merged.groupby(col["sample_id"], sort=True):
        rows = rows.sort_values(col["rollout_id"])
        kl = [as_array(v) for v in rows[col["kl_u_p"]]]
        logp = [as_array(v) for v in rows[col["logp"]]]
        tokens = [len(v) for v in rows[col["token_ids"]]] if col["token_ids"] else [len(x) for x in logp]
        vis = None
        if attention is not None:
            vis = []
            for a, p, g in zip(rows[col["attn_image"]], rows[col["attn_prompt"]], rows[col["attn_generated"]]):
                a, p, g = as_array(a), as_array(p), as_array(g)
                vis.append(a / np.maximum(a + p + g, 1e-12))
        for i in range(len(rows)):
            report["alignment"]["checked"] += 1
            measured = {len(kl[i]), len(logp[i])} | ({len(vis[i])} if vis is not None else set())
            if len(measured) != 1:
                report["alignment"]["mismatch"] += 1
            diff = str(tokens[i] - len(logp[i]))
            report["alignment"]["token_minus_measure"][diff] = report["alignment"]["token_minus_measure"].get(diff, 0) + 1
        lengths = rows[col["num_tokens"]].to_numpy() if col["num_tokens"] else np.array(tokens)
        answers = [None if (a is None or (isinstance(a, float) and np.isnan(a)) or str(a).strip() == "") else str(a)
                   for a in rows[col["extracted"]]]
        golds = rows[col["gold"]].tolist()
        questions[str(qid)] = Question(
            qid=str(qid), rollout_ids=rows[col["rollout_id"]].to_numpy(), lengths=np.asarray(lengths, dtype=np.int64),
            answers=answers, gold=golds[0], question=rows[col["question"]].iloc[0] if col["question"] else None,
            kl=kl, logp=logp, vis=vis, text=rows[col["text"]].tolist() if keep_text and col["text"] else [])
    if report["alignment"]["mismatch"]:
        raise ValueError(f"Token measurement lists disagree in length: {report['alignment']}")
    report["questions"] = len(questions)
    report["rollouts_per_question"] = sorted({len(q.rollout_ids) for q in questions.values()})
    return questions, report


def splits(cfg, question_ids):
    """Sort IDs lexicographically, permute with default_rng(seed); first 200 dev, rest held out."""
    ordered = sorted(question_ids)
    perm = np.random.default_rng(cfg["seed"]).permutation(len(ordered))
    permuted = [ordered[i] for i in perm]
    n = cfg["splits"]["dev_questions"]
    return {"permutation": permuted, "dev": permuted[:n], "eval": permuted[n:]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task3.yaml")
    parser.add_argument("--inspect", action="store_true", help="Print raw schemas and exit")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    if args.inspect:
        for path in sorted(course_data_dir().rglob("*.parquet")):
            frame = pd.read_parquet(path)
            print(path.relative_to(course_data_dir()), len(frame))
            print(frame.dtypes.to_string())
            print(frame.head(1).T.to_string()[:2000])
        return
    for name in cfg["datasets"]:
        questions, report = load_dataset(cfg, name, keep_text=False)
        split = splits(cfg, list(questions))
        if name == "math":
            split["live"] = split["eval"][: cfg["splits"]["live_questions"]]
        atomic_json(repo_path(cfg["results_dir"]) / name / "splits.json", split)
        atomic_json(repo_path(cfg["results_dir"]) / name / "load_report.json", report)
        print(name, json.dumps({k: v for k, v in report.items() if k != "columns"})[:1500])


if __name__ == "__main__":
    main()
