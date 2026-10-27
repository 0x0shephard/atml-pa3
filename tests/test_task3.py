import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from common.data import load_yaml
from task3_ttc import answers
from task3_ttc.live import last_boxed
from task3_ttc.load import Question, load_dataset, splits
from task3_ttc.offline import adaptive
from task3_ttc.selection import calibrated_confidence, rwc_margin, rwc_weights, scores_for, select, vote_winner


def question(groups, K, correct=None, lengths=None, vis=None, kl=None):
    n = len(groups)
    kl = kl if kl is not None else [np.full(3, k) for k in K]
    q = Question(qid="q", rollout_ids=np.arange(n), lengths=np.asarray(lengths if lengths is not None else [10] * n),
                 answers=[str(g) for g in groups], gold="0", question="?", kl=kl,
                 logp=[np.full(3, -1.0) for _ in range(n)], vis=vis)
    q.groups = np.asarray(groups)
    q.correct = np.asarray(correct if correct is not None else [g == 0 for g in groups])
    q.no_answer = np.zeros(n, dtype=bool)
    return q


class Task3(unittest.TestCase):
    def test_boxed_extraction_balances_braces(self):
        self.assertEqual(last_boxed(r"so \boxed{1} then \boxed{\frac{a}{b}} done"), r"\frac{a}{b}")
        self.assertIsNone(last_boxed("no box"))
        self.assertIsNone(last_boxed(r"\boxed{unclosed"))

    def test_mathvista_normalization_and_groups(self):
        meta = {"question_type": "free_form", "answer_type": "float", "precision": 2, "answer": "0.01"}
        groups, correct = answers.group_mathvista(["0.0115 \\, \\text{m}", "0.01", None, "10-20 years"], meta)
        self.assertEqual(groups[0], groups[1])
        self.assertTrue(correct[int(groups[0])])
        self.assertEqual(groups[2], -1)  # missing extraction -> NO_ANSWER
        self.assertFalse(correct[int(groups[3])])  # unparseable keeps its own incorrect group
        mc = {"question_type": "multi_choice", "answer_type": "text", "choices": ["3", "4"], "answer": "4"}
        self.assertEqual(answers.gold_mathvista(mc), "B")
        self.assertEqual(answers.normalize_mathvista("(B) 4", mc), "B")

    def test_rwc_weights_use_average_ranks(self):
        np.testing.assert_allclose(rwc_weights([3.0, 1.0, 2.0]), [1.0, 0.0, 0.5])
        np.testing.assert_allclose(rwc_weights([2.0, 2.0, 1.0]), [0.75, 0.75, 0.0])

    def test_vote_ties_go_to_earliest_supporter(self):
        winner, totals = vote_winner([2, 1, 1, 2], np.ones(4))
        self.assertEqual(winner, 2)
        self.assertAlmostEqual(rwc_margin({0: 3.0, 1: 1.0}), 0.5)
        self.assertAlmostEqual(rwc_margin({0: 2.0}), 1.0)

    def test_selection_rules(self):
        q = question([1, 0, 0, 1, 2], K=[0.1, 0.9, 0.2, 0.3, 0.95])
        order = np.arange(5)
        self.assertEqual(select(q, order, "first"), 1)
        self.assertEqual(select(q, order, "self_consistency"), 1)  # 2-2 tie, group 1 appears first
        self.assertEqual(select(q, order, "self_certainty"), 2)
        # RWC weights (5 - rank) / 4: group 0 gets 0.75 + 0.25, group 1 0.0 + 0.5, group 2 1.0;
        # the 1.0 tie goes to group 0, whose first supporter comes earlier.
        self.assertEqual(select(q, order, "self_certainty:rwc"), 0)
        self.assertEqual(select(q, order[:1], "self_certainty:rwc"), 1)  # N=1 returns the only candidate

    def test_visconf_calibrates_on_revealed_prefix_only(self):
        kl = [np.array([0.0, 10.0]), np.array([5.0, 5.0]), np.array([100.0, 100.0])]
        vis = [np.array([1.0, 0.0]), np.array([0.5, 0.5]), np.array([0.2, 0.2])]
        q = question([0, 1, 2], K=[0, 0, 0], kl=kl, vis=vis)
        C2, _ = calibrated_confidence(q, np.array([0, 1]))
        lo, hi = np.percentile([0, 10, 5, 5], [5, 95])
        expected = [np.clip((np.array([0, 10]) - lo) / (hi - lo), 0, 1).mean(), np.clip((5 - lo) / (hi - lo), 0, 1)]
        np.testing.assert_allclose(C2, expected)
        z = scores_for(q, np.array([0, 1]), "visconf")
        np.testing.assert_allclose(z, q.V[[0, 1]] * C2)
        token = scores_for(q, np.array([0, 1]), "visconf_token")
        _, tokens = calibrated_confidence(q, np.array([0, 1]))
        cov = [(c * vis[i]).mean() - c.mean() * vis[i].mean() for c, i in zip(tokens, [0, 1])]
        np.testing.assert_allclose(token - z, cov)  # Z_token - Z = Cov_t(c, v)

    def test_adaptive_stops_on_agreement_and_never_on_no_answer(self):
        cfg = {"adaptive": {"checkpoints": [4, 8, 16], "max_candidates": 32}}
        groups = [0] * 4 + [1] * 28
        q = question(groups, K=np.linspace(0, 1, 32))
        orders = {"q": np.arange(32)[None]}
        sim = adaptive(cfg, {"q": q}, orders, ["q"], "math", tau=1.0, gamma=None)
        self.assertEqual(int(sim.stop[0]), 4)
        q.no_answer = q.groups == 0
        sim = adaptive(cfg, {"q": q}, orders, ["q"], "math", tau=1.0, gamma=None)
        self.assertEqual(int(sim.stop[0]), 32)

    def test_loading_and_splits(self):
        cfg = load_yaml("configs/task3.yaml")
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / "qwen25-math"
            folder.mkdir()
            (Path(root) / "README.MD").write_text("synthetic")
            rows = [{"dataset": "math", "sample_id": f"id{q}", "rollout_id": r, "generated_token_ids": [1, 2, 3, 0],
                     "n_generated_text_tokens": 4, "extracted_answer": str(r % 2) if r != 3 else None,
                     "gold_answer": "1", "question_text": "1+0?", "generated_text": "..."}
                    for q in range(5) for r in range(4)]
            pd.DataFrame(rows).to_parquet(folder / "generations.parquet")
            pd.DataFrame([{**{k: r[k] for k in ("dataset", "sample_id", "rollout_id")}, "logp": [-0.1, -0.2, -0.3],
                           "kl_u_p": [1.0, 2.0, 3.0]} for r in rows]).to_parquet(folder / "probability_metrics.parquet")
            os.environ["PA3_COURSE_DATA"] = root
            try:
                qs, report = load_dataset(cfg, "math")
            finally:
                del os.environ["PA3_COURSE_DATA"]
            self.assertEqual(report["alignment"]["token_minus_measure"], {"1": 20})
            answers.label(qs, "math")
            q = qs["id0"]
            self.assertEqual(q.correct.tolist(), [False, True, False, False])
            self.assertTrue(q.no_answer[3])
            split = splits({"seed": 6304, "splits": {"dev_questions": 2}}, list(qs))
            self.assertEqual(len(split["dev"]), 2)
            self.assertEqual(sorted(split["dev"] + split["eval"]), sorted(qs))
            self.assertEqual(split, splits({"seed": 6304, "splits": {"dev_questions": 2}}, list(reversed(list(qs)))))


if __name__ == "__main__":
    unittest.main()
