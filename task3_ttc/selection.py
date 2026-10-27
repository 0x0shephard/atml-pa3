"""Quality scores and answer aggregation over a revealed candidate prefix.

Ties: score or vote ties go to the earliest supporting rollout in the current order; RWC uses
average ranks for tied scores. VisConf calibration uses only the revealed candidates' tokens.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def calibrated_confidence(q, cand):
    """C_i = mean_t clip((k_t - Q05) / max(Q95 - Q05, 1e-8), 0, 1), percentiles pooled over cand."""
    pooled = np.concatenate([q.kl[i] for i in cand]) if len(cand) else np.array([])
    if pooled.size == 0:
        return np.zeros(len(cand)), [np.zeros(0) for _ in cand]
    lo, hi = np.percentile(pooled, [5, 95])  # linear interpolation (NumPy default)
    width = max(hi - lo, 1e-8)
    tokens = [np.clip((q.kl[i] - lo) / width, 0.0, 1.0) for i in cand]
    return np.array([t.mean() if t.size else 0.0 for t in tokens]), tokens


def scores_for(q, cand, name):
    if name == "logp":
        return q.L[cand]
    if name == "self_certainty":
        return q.K[cand]
    if name == "shortest":
        return -q.lengths[cand].astype(float)
    if name == "vision":
        return q.V[cand]
    C, tokens = calibrated_confidence(q, cand)
    if name == "visconf":
        return q.V[cand] * C
    if name == "visconf_token":
        return np.array([(c * q.vis[i]).mean() if c.size else 0.0 for c, i in zip(tokens, cand)])
    raise KeyError(name)


def highest(scores):
    """Index of the best score; np.argmax returns the earliest position among ties."""
    return int(np.argmax(scores))


def vote_winner(groups, weights):
    """Group with the largest total weight; ties go to the group whose first supporter comes first."""
    totals, first = {}, {}
    for position, (g, w) in enumerate(zip(groups, weights)):
        totals[g] = totals.get(g, 0.0) + w
        first.setdefault(g, position)
    best = max(totals.values())
    winners = [g for g in totals if np.isclose(totals[g], best, rtol=0, atol=1e-12)]
    return min(winners, key=lambda g: first[g]), totals


def rwc_weights(scores):
    n = len(scores)
    if n == 1:
        return np.ones(1)
    ranks = rankdata(-np.asarray(scores), method="average")  # rank 1 = best
    return (n - ranks) / (n - 1)


def rwc_margin(totals):
    values = sorted(totals.values(), reverse=True)
    total = sum(values)
    if total <= 0:
        return 0.0
    return (values[0] - (values[1] if len(values) > 1 else 0.0)) / total


def select(q, cand, method):
    """Returns (selected group, position in cand used for the answer or None for votes)."""
    groups = q.groups[cand]
    if len(cand) == 1 or method == "first":
        return int(groups[0])
    if method == "self_consistency":
        return int(vote_winner(groups, np.ones(len(cand)))[0])
    score_name, aggregation = method.rsplit(":", 1) if ":" in method else (method, "best")
    scores = scores_for(q, cand, score_name)
    if aggregation == "best":
        return int(groups[highest(scores)])
    if aggregation == "rwc":
        return int(vote_winner(groups, rwc_weights(scores))[0])
    raise KeyError(method)


METHODS = {
    "math": ["first", "self_consistency", "logp", "self_certainty", "self_certainty:rwc", "shortest"],
    "mathvista": ["first", "self_consistency", "logp", "self_certainty", "self_certainty:rwc", "shortest",
                  "vision", "vision:rwc", "visconf", "visconf:rwc", "visconf_token", "visconf_token:rwc"],
}
