import copy
import unittest

import torch
from torch import nn

from task1_finetune import adapters
from task1_finetune.data import IGNORE, batches, normalize_text
from task1_finetune.train import L2SP, response_loss_sum, unique_parameters


def tiny_qwen():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8, tie_word_embeddings=True)
    return Qwen3ForCausalLM(config).eval()


class Task1(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize_text("  Janet’s ducks lay 16 eggs!\n"), normalize_text("janet s ducks lay 16 eggs"))

    def test_response_only_loss_matches_masked_cross_entropy(self):
        model = tiny_qwen()
        rows = [{"input_ids": [1, 2, 3, 4, 5], "labels": [IGNORE, IGNORE, 3, 4, 5]},
                {"input_ids": [6, 7, 8], "labels": [IGNORE, 7, 8]}]
        batch = next(batches(rows, 2, 0))
        total, tokens = response_loss_sum(model, batch, chunk=2)
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        expected = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 97), batch["labels"][:, 1:].reshape(-1),
                                               ignore_index=IGNORE, reduction="sum")
        self.assertEqual(tokens, 5)  # padding and prompt positions are excluded
        torch.testing.assert_close(total, expected, rtol=1e-5, atol=1e-5)

    def test_oplora_constraints_and_lora_matches_peft(self):
        torch.manual_seed(1)
        base = nn.Linear(24, 20, bias=False)
        lora = adapters.LowRankLinear(copy.deepcopy(base), 4, 8, 0.0, 0, torch.Generator().manual_seed(3))
        op = adapters.LowRankLinear(copy.deepcopy(base), 4, 8, 0.0, 6, torch.Generator().manual_seed(3))
        torch.testing.assert_close(lora.lora_A, op.lora_A)  # identical unprojected initialization
        for m in (lora, op):
            m.lora_B.data.normal_()
        dw = op.delta_weight()
        self.assertLess(float((op.U_k.T @ dw).norm()), 1e-4)
        self.assertLess(float((dw @ op.V_k).norm()), 1e-4)
        # Plain LoRA equals PEFT's LoRA with the same factors.
        from peft import LoraConfig, get_peft_model
        wrapper = nn.Sequential()
        wrapper.add_module("proj", nn.Linear(24, 20, bias=False))
        wrapper.proj.weight.data.copy_(base.weight.data)
        peft_model = get_peft_model(wrapper, LoraConfig(r=4, lora_alpha=8, target_modules=["proj"], lora_dropout=0.0))
        layer = peft_model.base_model.model.proj
        layer.lora_A["default"].weight.data.copy_(lora.lora_A.data)
        layer.lora_B["default"].weight.data.copy_(lora.lora_B.data)
        x = torch.randn(5, 24)
        torch.testing.assert_close(lora(x), peft_model(x), rtol=1e-5, atol=1e-5)

    def test_merge_matches_unmerged_and_export_is_projected(self):
        model = tiny_qwen()
        for p in model.parameters():
            p.requires_grad_(False)
        adapters.attach(model, ["q_proj", "down_proj"], 4, 8, 0.0, 3, seed=6304)
        for m in adapters.adapter_modules(model).values():
            m.lora_B.data.normal_(std=0.1)
        state = adapters.adapter_state(model)
        merged = adapters.merge_into(tiny_qwen(), state, 2.0)
        ids = torch.randint(0, 97, (2, 6))
        torch.testing.assert_close(model(ids).logits, merged(ids).logits, rtol=1e-4, atol=1e-4)
        unmerged = adapters.load_exported(tiny_qwen(), state, 4, 8)
        torch.testing.assert_close(unmerged(ids).logits, merged(ids).logits, rtol=1e-4, atol=1e-4)

    def test_l2sp_gradient_equals_autograd_penalty(self):
        model = tiny_qwen()
        params = unique_parameters(model)
        self.assertEqual(len(params), len({id(p) for p in params}))  # tied embedding counted once
        penalty = L2SP(params, 1e-3, "same")
        with torch.no_grad():
            for p in params:
                p.add_(torch.randn_like(p) * 0.01)
        for p in params:
            p.grad = torch.zeros_like(p)
        omega = penalty.omega_and_grad(True)
        manual = [p.grad.clone() for p in params]
        for p in params:
            p.grad = None
        expected = sum(((p - r) ** 2).sum() for p, r in zip(params, penalty.reference)) / 2
        (1e-3 * expected).backward()
        self.assertAlmostEqual(omega, float(expected), places=4)
        for g, p in zip(manual, params):
            torch.testing.assert_close(g, p.grad, rtol=1e-4, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
