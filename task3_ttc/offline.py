"""Offline Task 3 analysis on the supplied pools: repeated sampling, selection, signals, adaptive stopping.

All outcomes average over ten fixed rollout orders within a question, then across questions. Gold
answers enter only through the precomputed correctness labels used for scoring and retrospective
analysis, never in selection, stopping or normalization.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common.data import load_yaml, repo_path
from common.runtime import atomic_json
from task3_ttc import answers as answer_rules
from task3_ttc.load import load_dataset, splits
from task3_ttc.selection import METHODS, calibrated_confidence, rwc_margin, rwc_weights, scores_for, select, vote_winner


def orders_for(cfg, questions):
    """Ten seeded permutations of the 32 rollouts per question, reused by every method and policy."""
    rng = np.random.default_rng(cfg["seed"])
    out = {}
    for qid in sorted(questions):
        n = len(questions[qid].rollout_ids)
        out[qid] = np.stack([rng.permutation(n) for _ in range(cfg["offline"]["orders"])])
    return out


def run_selection(cfg, questions, orders, methods):
    rows = []
    for qid, q in questions.items():
        for o, order in enumerate(orders[qid]):
            for n in cfg["offline"]["budgets"]:
                cand = order[:n]
                cost_all = int(q.lengths[cand].sum())
                coverage = bool(q.correct[cand].any())
                distinct = len(set(q.groups[cand].tolist()))
                for method in methods:
                    group = select(q, cand, method)
                    correct = bool(q.correct[cand][q.groups[cand] == group][0])
                    position = int(np.flatnonzero(q.groups[cand] == group)[0])
                    rows.append({"qid": qid, "order": o, "n": n, "method": method, "group": group,
                                 "rollout_id": int(q.rollout_ids[cand[position]]), "correct": correct,
                                 "oracle": coverage, "distinct": distinct,
                                 "cost": int(q.lengths[cand[0]]) if method == "first" else cost_all})
    return pd.DataFrame(rows)


def per_question(frame):
    return frame.groupby(["qid", "n", "method"], as_index=False)[["correct", "oracle", "distinct", "cost"]].mean()


def summarize(pq, ids):
    sub = pq[pq.qid.isin(ids)]
    table = sub.groupby(["n", "method"])[["correct", "oracle", "distinct", "cost"]].mean().reset_index()
    table = table.rename(columns={"correct": "accuracy", "oracle": "oracle_coverage", "cost": "mean_tokens"})
    # Failure decomposition: sampling (no correct candidate) vs selection (correct available but missed).
    sub = sub.assign(sampling_failure=1 - sub.oracle, selection_failure=sub.oracle - sub.correct)
    fails = sub.groupby(["n", "method"])[["sampling_failure", "selection_failure"]].mean().reset_index()
    return table.merge(fails, on=["n", "method"])


def paired_bootstrap(pq, ids, a, b, n, iterations, seed):
    sub = pq[(pq.n == n) & pq.qid.isin(ids)].pivot(index="qid", columns="method", values="correct")
    diff = (sub[a] - sub[b]).to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(iterations, len(diff)))
    means = diff[idx].mean(1)
    return {"a": a, "b": b, "n": n, "diff_pp": 100 * float(diff.mean()),
            "ci95_pp": [100 * float(np.percentile(means, 2.5)), 100 * float(np.percentile(means, 97.5))]}


def signal_analysis(questions, ids, kind):
    """At N = 32: top-vs-bottom quartile correctness gaps within questions, length associations, and
    (MathVista) confidence-vision relationships and the tokenwise/rollout-level fusion gap."""
    names = ["logp", "self_certainty", "shortest"] + (["vision", "visconf", "visconf_token"] if kind == "mathvista" else [])
    gaps = {k: [] for k in names}
    gaps_mixed = {k: [] for k in names}
    length_corr, corr_tokens, covariances = [], [], []
    for qid in ids:
        q = questions[qid]
        cand = np.arange(len(q.rollout_ids))
        correct = q.correct.astype(float)
        mixed = 0 < correct.mean() < 1
        for name in names:
            s = scores_for(q, cand, name)
            order = np.argsort(-s, kind="stable")
            k = len(cand) // 4
            gap = correct[order[:k]].mean() - correct[order[-k:]].mean()
            gaps[name].append(gap)
            if mixed:
                gaps_mixed[name].append(gap)
        if mixed:
            length_corr.append(spearmanr(q.lengths, correct).statistic)
        if kind == "mathvista":
            _, tokens = calibrated_confidence(q, cand)
            for c, i in zip(tokens, cand):
                v = q.vis[i]
                if c.size > 2 and c.std() > 0 and v.std() > 0:
                    corr_tokens.append((qid, float(np.corrcoef(c, v)[0, 1])))
                if c.size:
                    covariances.append(float((c * v).mean() - c.mean() * v.mean()))
    out = {"quartile_gap_all": {k: float(np.mean(v)) for k, v in gaps.items()},
           "quartile_gap_mixed": {k: float(np.mean(v)) if v else None for k, v in gaps_mixed.items()},
           "mixed_questions": len(next(iter(gaps_mixed.values()))),
           "length_correctness_spearman_mixed": {"mean": float(np.nanmean(length_corr)) if length_corr else None,
                                                 "n": len(length_corr)}}
    if kind == "mathvista":
        per_q = pd.DataFrame(corr_tokens, columns=["qid", "r"]).groupby("qid").r.mean()
        out["confidence_vision_corr"] = {"rollout_mean": float(np.mean([r for _, r in corr_tokens])),
                                         "question_mean": float(per_q.mean()), "question_median": float(per_q.median()),
                                         "question_q25_q75": [float(per_q.quantile(.25)), float(per_q.quantile(.75))],
                                         "rollouts": len(corr_tokens)}
        out["tokenwise_minus_rollout_fusion"] = {"mean_cov": float(np.mean(covariances)),
                                                 "mean_abs_cov": float(np.mean(np.abs(covariances)))}
    return out


def adaptive(cfg, questions, orders, ids, kind, tau, gamma):
    """Simulated stopping between completed rollouts; always returns the current SC answer."""
    a = cfg["adaptive"]
    score_name = "self_certainty" if kind == "math" else "visconf"
    rows = []
    for qid in ids:
        q = questions[qid]
        for o, order in enumerate(orders[qid]):
            stop = a["max_candidates"]
            for n in a["checkpoints"]:
                cand = order[:n]
                groups = q.groups[cand]
                sc, votes = vote_winner(groups, np.ones(n))
                top = sorted(votes.values(), reverse=True)
                unique = len(top) == 1 or top[0] > top[1]
                agreed = unique and not q.no_answer[cand][groups == sc].any() and votes[sc] >= tau * n - 1e-9
                if agreed and gamma is not None:
                    rwc, totals = vote_winner(groups, rwc_weights(scores_for(q, cand, score_name)))
                    agreed = rwc == sc and rwc_margin(totals) >= gamma - 1e-12
                if agreed:
                    stop = n
                    break
            cand = order[:stop]
            group = select(q, cand, "self_consistency")
            rows.append({"qid": qid, "order": o, "stop": stop, "cost": int(q.lengths[cand].sum()),
                         "correct": bool(q.correct[cand][q.groups[cand] == group][0]),
                         "fixed32_correct": bool(q.correct[q.groups == select(q, order, "self_consistency")][0]),
                         "pool_correct_fraction": float(q.correct.mean())})
    return pd.DataFrame(rows)


def choose_policy(cfg, questions, orders, dev, kind):
    a = cfg["adaptive"]
    fixed = adaptive(cfg, questions, orders, dev, kind, tau=2.0, gamma=None)  # never stops early
    target = fixed.correct.mean() - a["tolerance_pp"] / 100
    grid = []
    for family in ("agreement", "quality"):
        candidates = [(None, None)]  # always-32 fallback
        for tau in a["taus"]:
            for gamma in (a["gammas"] if family == "quality" else [None]):
                candidates.append((tau, gamma))
        for tau, gamma in candidates:
            sim = fixed if tau is None else adaptive(cfg, questions, orders, dev, kind, tau, gamma)
            per_q = sim.groupby("qid")[["correct", "cost", "stop"]].mean()
            grid.append({"family": family, "tau": tau, "gamma": gamma, "accuracy": float(per_q.correct.mean()),
                         "mean_cost": float(per_q.cost.mean()), "mean_rollouts": float(per_q.stop.mean())})
    grid = pd.DataFrame(grid)
    chosen = {}
    for family, sub in grid.groupby("family"):
        ok = sub[sub.accuracy >= target - 1e-12].copy()
        ok["tau_key"] = ok.tau.fillna(np.inf)
        ok["gamma_key"] = ok.gamma.fillna(-1 if family == "agreement" else np.inf)
        best = ok.sort_values(["mean_cost", "accuracy", "tau_key", "gamma_key"], ascending=[True, False, True, True]).iloc[0]
        chosen[family] = {"tau": None if pd.isna(best.tau) else float(best.tau),
                          "gamma": None if pd.isna(best.gamma) else float(best.gamma),
                          "dev_accuracy": float(best.accuracy), "dev_mean_cost": float(best.mean_cost)}
    return grid, chosen, float(fixed.correct.mean())


def evaluate_policy(cfg, questions, orders, ids, kind, policy):
    tau = policy["tau"] if policy["tau"] is not None else 2.0
    sim = adaptive(cfg, questions, orders, ids, kind, tau, policy["gamma"] if policy["tau"] is not None else None)
    per_q = sim.groupby("qid")[["correct", "cost", "stop"]].mean()
    early_wrong = sim[(sim.stop < cfg["adaptive"]["max_candidates"]) & ~sim.correct]
    return sim, {"accuracy": float(per_q.correct.mean()), "mean_cost": float(per_q.cost.mean()),
                 "mean_rollouts": float(per_q.stop.mean()),
                 "stop_distribution": {str(k): float(v) for k, v in sim.stop.value_counts(normalize=True).sort_index().items()},
                 "incorrect_early_stops": int(len(early_wrong)),
                 "incorrect_early_stops_recoverable_at_32": int(early_wrong.fixed32_correct.sum()),
                 "early_stop_rate": float((sim.stop < cfg["adaptive"]["max_candidates"]).mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task3.yaml")
    parser.add_argument("--datasets", nargs="*")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    for name in args.datasets or list(cfg["datasets"]):
        kind = cfg["datasets"][name]["kind"]
        out = repo_path(cfg["results_dir"]) / name
        questions, report = load_dataset(cfg, name, keep_text=False)
        metadata = answer_rules.load_mathvista_metadata(cfg) if kind == "mathvista" else None
        answer_rules.label(questions, kind, metadata)
        split = splits(cfg, list(questions))
        if name == "math":
            split["live"] = split["eval"][: cfg["splits"]["live_questions"]]
        atomic_json(out / "splits.json", split)
        atomic_json(out / "load_report.json", {**report, "scorer": answer_rules.scorer_versions()})
        labels = pd.DataFrame([{"qid": q.qid, "rollout_id": int(r), "group": int(g), "correct": bool(c),
                                "no_answer": bool(na), "tokens": int(t), "answer": a}
                               for q in questions.values()
                               for r, g, c, na, t, a in zip(q.rollout_ids, q.groups, q.correct, q.no_answer, q.lengths, q.answers)])
        labels.to_parquet(out / "rollout_labels.parquet")
        orders = orders_for(cfg, questions)
        atomic_json(out / "orders.json", {k: v.tolist() for k, v in orders.items()})
        frame = run_selection(cfg, questions, orders, METHODS[kind])
        frame.to_parquet(out / "selections.parquet")
        pq = per_question(frame)
        pq.to_parquet(out / "per_question.parquet")
        tables = {s: summarize(pq, split[s]) for s in ("dev", "eval")}
        for s, table in tables.items():
            table.to_csv(out / f"selection_summary_{s}.csv", index=False)
        comparisons = [("self_consistency", "first"), ("self_certainty", "logp"), ("self_certainty:rwc", "self_consistency"),
                       ("self_certainty:rwc", "self_certainty"), ("shortest", "self_consistency")]
        if kind == "mathvista":
            comparisons += [("visconf:rwc", "self_consistency"), ("visconf:rwc", "visconf"), ("visconf", "self_certainty"),
                            ("visconf_token:rwc", "visconf:rwc"), ("visconf_token", "visconf"), ("vision:rwc", "self_consistency")]
        boot = [paired_bootstrap(pq, split["eval"], a, b, n, cfg["offline"]["bootstrap"], cfg["seed"])
                for (a, b) in comparisons for n in (4, 8, 32)]
        atomic_json(out / "bootstrap_eval.json", boot)
        atomic_json(out / "signals_eval.json", signal_analysis(questions, split["eval"], kind))
        grid, chosen, fixed_dev = choose_policy(cfg, questions, orders, split["dev"], kind)
        grid.to_csv(out / "adaptive_dev_grid.csv", index=False)
        held = {}
        for family, policy in chosen.items():
            sim, held[family] = evaluate_policy(cfg, questions, orders, split["eval"], kind, policy)
            sim.to_parquet(out / f"adaptive_eval_{family}.parquet")
        fixed = tables["eval"][tables["eval"].method == "self_consistency"][["n", "accuracy", "mean_tokens"]]
        atomic_json(out / "adaptive_summary.json", {"selected": chosen, "fixed32_dev_accuracy": fixed_dev,
                                                     "held_out": held, "fixed_sc_eval": fixed.to_dict("records")})
        print(name, tables["eval"][tables["eval"].n.isin([1, 8, 32])].to_string(index=False))
        print(json.dumps(held, indent=1))


if __name__ == "__main__":
    main()
