import unittest

import numpy as np
import torch

from task2_vlm.evaluate import mismatched_mapping
from task2_vlm.metrics import consensus, normalize_answer, pair_bootstrap, representative, soft_accuracy
from task2_vlm.model import VLM, Connector, Formatter, collate, pool_features
from task2_vlm.prepare_data import plural, with_article


class FakeTokenizer:
    """Character-level stand-in exposing the pieces Formatter needs."""
    pad_token_id = 0

    def apply_chat_template(self, messages, **_):
        return f"<u>{messages[0]['content']}</u><a>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [10 + (ord(c) % 80) for c in text]}

    def convert_tokens_to_ids(self, token):
        return {"<|im_end|>": 3, "<|image_pad|>": 5}[token]


CFG = {"caps": {"question": 80, "caption_target": 64, "short_answer_target": 16},
       "prompts": {"caption": "Describe.", "short_answer_suffix": "One word."}}


def tiny_lm():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=97, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=8)
    return Qwen3ForCausalLM(config)


class Task2(unittest.TestCase):
    def test_vqa_scoring(self):
        answers = ["two"] * 3 + ["2"] * 2 + ["3"] * 5
        self.assertEqual(normalize_answer("Two."), "2")
        self.assertEqual(normalize_answer("the red one!"), "red 1")
        # Five humans said 2: every leave-one-out subset keeps >= 3 matches except none -> score 1.
        self.assertAlmostEqual(soft_accuracy("2", answers), 1.0)
        self.assertAlmostEqual(soft_accuracy("4", answers), 0.0)
        one = ["yes"] + ["no"] * 9
        self.assertAlmostEqual(soft_accuracy("yes", one), 0.9 * (1 / 3))
        self.assertEqual(consensus(["Yes", "yes", "no"]), ("yes", 2))
        self.assertEqual(representative(["Yes", "yes", "yes", "no"], "yes"), "yes")

    def test_templates(self):
        self.assertEqual(plural("person"), "people")
        self.assertEqual(plural("bus"), "buses")
        self.assertEqual(plural("teddy bear"), "teddy bears")
        self.assertEqual(with_article("umbrella"), "an umbrella")

    def test_no_image_drops_only_visual_block(self):
        fmt = Formatter(FakeTokenizer(), CFG, 4)
        ex = {"kind": "vqa", "question": "What?", "target": "cat"}
        with_image, without = fmt.encode(ex), fmt.encode(ex, image=False)
        visual = [5] * 4 + fmt.newline
        start = len(fmt.head)
        self.assertEqual(with_image["input_ids"][start:start + len(visual)], visual)
        self.assertEqual(with_image["input_ids"][:start] + with_image["input_ids"][start + len(visual):], without["input_ids"])
        self.assertEqual(sum(x != -100 for x in with_image["labels"]), len(fmt.ids("cat")) + 1)
        self.assertEqual(sum(with_image["nll_mask"]), len(fmt.ids("cat")))  # termination excluded

    def test_left_padding_positions(self):
        batch = collate([{"input_ids": [1, 2, 3], "labels": [-100] * 3, "nll_mask": [0] * 3},
                         {"input_ids": [4], "labels": [-100], "nll_mask": [0]}], 0, left=True)
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1], [0, 0, 1]])
        self.assertEqual(batch["position_ids"].tolist(), [[0, 1, 2], [0, 0, 0]])

    def test_pooling(self):
        x = torch.arange(196 * 2, dtype=torch.float32).view(1, 196, 2)
        pooled = pool_features(x, 49)
        grid = x.view(14, 14, 2)
        torch.testing.assert_close(pooled[0, 0], grid[:2, :2].reshape(4, 2).mean(0))
        self.assertEqual(pooled.shape, (1, 49, 2))

    def test_connector_receives_gradients_through_frozen_lm(self):
        lm = tiny_lm()
        for p in lm.parameters():
            p.requires_grad_(False)
        fmt = Formatter(FakeTokenizer(), CFG, 4)
        vlm = VLM(lm, Connector(8, 16, 16), fmt)
        ex = {"kind": "vqa", "question": "Q?", "target": "a"}
        batch = collate([fmt.encode(ex), fmt.encode(ex)], 0)
        batch["input_ids"] = batch["input_ids"] % 97
        batch["input_ids"][batch["labels"] == -100] %= 97
        loss = vlm.example_mean_loss(batch, torch.randn(2, 4, 8))
        loss.mean().backward()
        self.assertTrue(all(p.grad is not None and p.grad.abs().sum() > 0 for p in vlm.connector.parameters()))
        self.assertTrue(all(p.grad is None for p in lm.parameters()))

    def test_mismatched_mapping_and_bootstrap(self):
        pairs = [{"pair_id": f"p{i}"} for i in range(10)]
        mapping = mismatched_mapping(pairs, 6304)
        self.assertTrue(all(k != v for k, v in mapping.items()))
        self.assertEqual(mapping, mismatched_mapping(pairs, 6304))
        result = pair_bootstrap(np.ones(20), 100, 0)
        self.assertEqual(result["ci95"], [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
