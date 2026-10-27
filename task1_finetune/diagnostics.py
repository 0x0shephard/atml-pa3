"""Weight-space and activation diagnostics on down_proj in layers 0, 13 and 27.

rho_k = ||U_k U_k^T dW||_F^2 / ||dW||_F^2 (share of the update in W0's leading output subspace),
eta   = ||dW||_F / ||W0||_F, and R(z) = ||dW z|| / (||W0 z|| + 1e-8) on starting-model inputs z
cached at the last prompt token of fixed math (GSM8K) and non-math (HellaSwag) probes. All FP32.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from common.data import load_yaml, repo_path
from common.runtime import atomic_json
from task1_finetune.data import load_tokenizer, prompt_text, sample_probes
from task1_finetune.train import load_model


def probe_prompts(cfg, tokenizer):
    """Benchmark prompt formats: GSM8K as rendered for chat evaluation, HellaSwag as a stock context."""
    from lm_eval.tasks import TaskManager, get_task_dict
    probes = sample_probes(cfg)
    tasks = get_task_dict([cfg["eval"]["gsm8k_task"], "hellaswag"], TaskManager())
    gsm_task, hella_task = tasks[cfg["eval"]["gsm8k_task"]], tasks["hellaswag"]
    gsm_docs, hella_docs = list(gsm_task.test_docs()), list(hella_task.validation_docs())
    prompts = [("math", i, prompt_text(tokenizer, gsm_task.doc_to_text(gsm_docs[i]))) for i in probes["gsm8k_test"]]
    prompts += [("non_math", i, hella_task.doc_to_text(hella_docs[i])) for i in probes["hellaswag_validation"]]
    return prompts


@torch.no_grad()
def cache_inputs(cfg, model, tokenizer):
    layers = cfg["diagnostics"]["layers"]
    captured = {}
    def hook(layer):
        def fn(_, inputs):
            captured[layer] = inputs[0][0, -1].detach().float().cpu()
        return fn
    handles = [model.model.layers[l].mlp.down_proj.register_forward_pre_hook(hook(l)) for l in layers]
    cache = {l: [] for l in layers}
    kinds, ids = [], []
    for kind, doc_id, text in probe_prompts(cfg, tokenizer):
        batch = tokenizer([text], return_tensors="pt", add_special_tokens=False).to(model.device)
        model(**batch)  # Single unpadded prompt: the last position is the last prompt token.
        for l in layers:
            cache[l].append(captured[l])
        kinds.append(kind)
        ids.append(doc_id)
    for h in handles:
        h.remove()
    return {l: torch.stack(v) for l, v in cache.items()}, np.array(kinds), ids


def delta_for(cfg, condition, base_weights):
    """dW per layer for a saved condition (Full FT: trained - W0; adapters: exported B~A~ * alpha/r)."""
    name, stage = condition
    saved = torch.load(repo_path(cfg["remote_dir"]) / name / stage / "down_proj.pt", map_location="cpu")
    deltas = {}
    for layer, w0 in base_weights.items():
        if f"layer{layer}.delta" in saved:
            deltas[layer] = saved[f"layer{layer}.delta"].float()
        else:
            deltas[layer] = saved[f"layer{layer}.weight"].float() - w0
    return deltas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["midpoint", "final"], required=True)
    parser.add_argument("--name")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    if args.name:
        cfg["name"] = args.name
    tokenizer = load_tokenizer(cfg)
    model = load_model(cfg, torch.float32).eval()
    layers, k = cfg["diagnostics"]["layers"], cfg["diagnostics"]["protect_k"]
    base = {l: model.model.layers[l].mlp.down_proj.weight.detach().float().cpu() for l in layers}
    z, kinds, ids = cache_inputs(cfg, model, tokenizer)
    deltas = delta_for(cfg, (cfg["name"], args.stage), base)
    result = {"condition": f"{cfg['name']}_{args.stage}", "protect_k": k, "layers": {},
              "probe_ids": {"math": [i for i, t in zip(ids, kinds) if t == "math"],
                            "non_math": [i for i, t in zip(ids, kinds) if t == "non_math"]}}
    for l in layers:
        w0, dw = base[l], deltas[l]
        U = torch.linalg.svd(w0, full_matrices=False)[0][:, :k]
        norm2 = float(dw.pow(2).sum())
        rho = float((U @ (U.T @ dw)).pow(2).sum() / norm2) if norm2 > 0 else None
        eta = float(dw.norm() / w0.norm())
        ratios = ((z[l] @ dw.T).norm(dim=1) / ((z[l] @ w0.T).norm(dim=1) + 1e-8)).numpy()
        entry = {"rho_k": rho, "eta": eta, "delta_fro": norm2 ** 0.5, "w0_fro": float(w0.norm())}
        for kind in ("math", "non_math"):
            values = ratios[kinds == kind]
            entry[f"R_{kind}"] = {"mean": float(values.mean()), "median": float(np.median(values)),
                                  "std": float(values.std()), "values": values.tolist()}
        # Projection constraint check, separate from behavioral evidence.
        V = torch.linalg.svd(w0, full_matrices=False)[2][:k].T
        entry["output_subspace_leak"] = float((U.T @ dw).norm() / max(dw.norm(), 1e-30))
        entry["input_subspace_leak"] = float((dw @ V).norm() / max(dw.norm(), 1e-30))
        result["layers"][str(l)] = entry
    atomic_json(repo_path(cfg["results_dir"]) / "diagnostics" / f"{result['condition']}.json", result)
    print({l: {k2: v for k2, v in e.items() if not isinstance(v, dict)} for l, e in result["layers"].items()})


if __name__ == "__main__":
    main()
