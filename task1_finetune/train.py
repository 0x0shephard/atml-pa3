"""Supervised math fine-tuning of Qwen3-0.6B: Full FT, Full FT + L2-SP, LoRA, OPLoRA.

Mixed precision follows the assignment: Full FT keeps FP32 master parameters under FP16
autocast; adapter runs freeze an FP16 backbone and train FP32 adapters. Optimizer states are
FP32, gradients are unscaled before clipping, and dynamic loss scaling may skip updates.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from common.data import load_yaml, repo_path
from common.logging_utils import set_seed
from common.runtime import Run, atomic_json, completed, file_bytes, optimizer_step
from task1_finetune import adapters
from task1_finetune.data import IGNORE, batches, load_tokenizer, prepare


def load_model(cfg, dtype, device="cuda"):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(cfg["models"]["qwen3"], revision=cfg["revisions"]["qwen3"],
                                                 torch_dtype=dtype, attn_implementation="sdpa")
    return model.to(device)


def unique_parameters(model):
    seen, params = set(), []
    for _, p in model.named_parameters():
        if id(p) not in seen:
            seen.add(id(p))
            params.append(p)
    return params


def response_loss_sum(model, batch, chunk):
    """Sum of next-token cross-entropy over supervised positions.

    Logits are formed only at supervised positions and recomputed in chunks during backward,
    so a 2 x 2048-token microbatch never materializes full-vocabulary logits.
    """
    hidden = model.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
    targets = batch["labels"][:, 1:]
    keep = targets != IGNORE
    states, targets = hidden[:, :-1][keep], targets[keep]
    head = model.lm_head
    def chunk_loss(h, y):
        return F.cross_entropy(head(h).float(), y, reduction="sum")
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, len(targets), chunk):
        total = total + checkpoint(chunk_loss, states[start:start + chunk], targets[start:start + chunk],
                                   use_reentrant=False)
    return total, int(keep.sum())


class L2SP:
    """Omega = 1/2 sum_j (theta_j - theta0_j)^2 over unique parameters; its gradient lambda*(theta - theta0)
    is added to the unscaled gradients, which equals backpropagating lambda*Omega."""

    def __init__(self, params, lam, device="same"):
        """device="same" keeps theta0 beside the parameters; "cpu" offloads it to pinned host memory
        and streams it per parameter (transfer time is recorded)."""
        started = time.perf_counter()
        self.params, self.lam = params, lam
        self.offload = device == "cpu" and params[0].is_cuda
        self.device = "cpu" if self.offload else str(params[0].device)
        self.reference = [p.detach().float().clone() for p in params]
        if self.offload:
            self.reference = [r.cpu().pin_memory() for r in self.reference]
        self.reference_bytes = sum(r.numel() * r.element_size() for r in self.reference)
        self.setup_seconds = time.perf_counter() - started
        self.penalty_seconds = 0.0
        self.transfer_seconds = 0.0
        self.last_omega = 0.0

    @torch.no_grad()
    def omega_and_grad(self, apply_grad):
        started = time.perf_counter()
        omega = torch.zeros((), dtype=torch.float64, device=self.params[0].device)
        for p, ref in zip(self.params, self.reference):
            if self.offload:
                moved = time.perf_counter()
                ref = ref.to(p.device, non_blocking=True)
                torch.cuda.synchronize()
                self.transfer_seconds += time.perf_counter() - moved
            diff = p.detach().float() - ref
            omega += diff.double().pow(2).sum()
            if apply_grad and p.grad is not None:
                p.grad.add_(diff.to(p.grad.dtype), alpha=self.lam)
        if self.params[0].is_cuda:
            torch.cuda.synchronize()
        self.penalty_seconds += time.perf_counter() - started
        self.last_omega = float(omega / 2)
        return self.last_omega


@torch.no_grad()
def displacement(params, reference):
    total = 0.0
    for p, ref in zip(params, reference):
        total += float((p.detach().float() - ref.to(p.device)).double().pow(2).sum())
    return total ** 0.5


@torch.no_grad()
def save_down_proj(model, cfg, path):
    """FP32 trained down_proj weights for the weight-space and activation diagnostics."""
    weights = {}
    for layer in cfg["diagnostics"]["layers"]:
        module = model.model.layers[layer].mlp.down_proj
        if isinstance(module, adapters.LowRankLinear):
            weights[f"layer{layer}.delta"] = module.delta_weight().cpu()
        else:
            weights[f"layer{layer}.weight"] = module.weight.detach().float().cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights, path)


def save_full_snapshot(model, tokenizer, path):
    """FP16 copy of a fully fine-tuned model for evaluation (remote-only; not committed)."""
    started = time.perf_counter()
    state = {k: v.detach().half() for k, v in model.state_dict().items()}
    model.save_pretrained(path, state_dict=state, safe_serialization=True)
    tokenizer.save_pretrained(path)
    return time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-updates", type=int, help="Smoke test: stop after this many updates")
    parser.add_argument("--stress", action="store_true", help="Smoke test: longest examples first")
    parser.add_argument("--name")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    if args.name:
        cfg["name"] = args.name
    name, train_cfg = cfg["name"], cfg["train"]
    output = repo_path(cfg["outputs_dir"]) / name
    remote = repo_path(cfg["remote_dir"]) / name
    if completed(output):
        print("already complete", output)
        return
    set_seed(cfg["seed"])
    tokenizer = load_tokenizer(cfg)
    rows = prepare(cfg)
    if args.stress:
        rows = sorted(rows, key=lambda r: -len(r["input_ids"]))
    per_update = train_cfg["microbatch"] * train_cfg["grad_accum"]
    updates = args.max_updates or train_cfg["updates"]
    if not args.max_updates and len(rows) != updates * per_update:
        raise ValueError("Training budget mismatch")
    rows = rows[: updates * per_update]
    full = cfg["method"] == "full"
    model = load_model(cfg, torch.float32 if full else torch.float16)
    model.config.use_cache = False
    overhead = {}
    if full:
        for p in model.parameters():
            p.requires_grad_(True)
    else:
        for p in model.parameters():
            p.requires_grad_(False)
        a = cfg["adapter"]
        replaced, seconds = adapters.attach(model, a["targets"], a["rank"], a["alpha"], a["dropout"],
                                            a["protect_k"], cfg["seed"])
        overhead.update(adapted_modules=len(replaced), adapter_setup_seconds=seconds)
        if a["protect_k"]:
            overhead.update(svd_preprocessing_seconds=seconds, basis_bytes=adapters.basis_bytes(model))
    if train_cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    params = [p for p in unique_parameters(model) if p.requires_grad]
    trainable = sum(p.numel() for p in params)
    groups = [{"params": [p for p in params if p.ndim >= 2], "weight_decay": train_cfg["weight_decay"]},
              {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0}]
    # Fused AdamW avoids full-size foreach temporaries, which matters for Full FT on a 15 GB T4.
    optimizer = torch.optim.AdamW(groups, lr=cfg["learning_rate"], betas=tuple(train_cfg["betas"]),
                                  eps=train_cfg["eps"], fused=torch.cuda.is_available())
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, train_cfg["warmup_updates"], train_cfg["updates"])
    scaler = torch.amp.GradScaler("cuda")
    run = Run(output, cfg, [r["source_index"] for r in rows], name, resume=args.resume)
    progress = run.restore({"model": model}, {"model": optimizer}, {"model": scheduler}, scaler)
    lam = train_cfg["lambda_l2sp"]
    penalty = None
    reference = None
    if full:
        # theta0 is the untouched pretrained checkpoint, reloaded so resumed runs anchor correctly.
        initial = load_model(cfg, torch.float32, "cpu")
        reference_params = [p for p in unique_parameters(initial)]
        if lam:
            penalty = L2SP(params, lam, train_cfg.get("l2sp_reference_device", "same"))
            with torch.no_grad():
                for ref, p0 in zip(penalty.reference, reference_params):
                    ref.copy_(p0)
            overhead.update(reference_bytes=penalty.reference_bytes, reference_setup_seconds=penalty.setup_seconds,
                            reference_device=penalty.device)
        reference = [p.detach().clone() for p in reference_params]
        del initial
    pad = tokenizer.pad_token_id
    microbatches = list(batches(rows, train_cfg["microbatch"], pad))
    update = progress["update"]
    skipped_total = sum(1 for r in _log_rows(output) if r.get("skipped"))
    displacements = dict(json.loads((output / "displacement.json").read_text())) if (output / "displacement.json").exists() else {}
    while update < updates:
        step_start = time.perf_counter()
        group = microbatches[update * train_cfg["grad_accum"]:(update + 1) * train_cfg["grad_accum"]]
        tokens = sum(int((b["labels"][:, 1:] != IGNORE).sum()) for b in group)
        loss_sum = 0.0
        for batch in group:
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                total, _ = response_loss_sum(model, batch, train_cfg["loss_chunk"])
            scaler.scale(total / tokens).backward()
            loss_sum += float(total.detach())
        if not math.isfinite(loss_sum):
            raise FloatingPointError(f"Non-finite SFT loss at update {update + 1}")
        record = {"update": update + 1, "examples_seen": (update + 1) * per_update, "sft_loss": loss_sum / tokens,
                  "tokens": tokens, "lr": scheduler.get_last_lr()[0], "scale": scaler.get_scale()}
        extra = (lambda: penalty.omega_and_grad(True)) if penalty else None
        norm, skipped = optimizer_step(params, optimizer, scaler, train_cfg["max_grad_norm"], extra)
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if penalty:
            record.update(omega=penalty.last_omega, lambda_omega=lam * penalty.last_omega,
                          total_loss=record["sft_loss"] + lam * penalty.last_omega)
        skipped_total += skipped
        record.update(grad_norm=norm, skipped=skipped, step_seconds=time.perf_counter() - step_start,
                      elapsed_seconds=run.seconds())
        update += 1
        run.log(record)
        if update % 10 == 0 or skipped:
            print(json.dumps(record), flush=True)
        if full and update in (train_cfg["midpoint_update"], updates):
            displacements[str(update)] = displacement(params, reference)
            atomic_json(output / "displacement.json", displacements)
        if update == train_cfg["midpoint_update"] and cfg.get("evaluate_midpoint"):
            save_full_snapshot(model, tokenizer, remote / "midpoint")
            save_down_proj(model, cfg, remote / "midpoint" / "down_proj.pt")
        if update in train_cfg["checkpoint_updates"] or update == updates:
            run.save({"model": model}, {"model": optimizer}, {"model": scheduler}, scaler,
                     {"update": update, "cursor": update * per_update})
    training_seconds = run.seconds()
    if full:
        export_seconds = save_full_snapshot(model, tokenizer, remote / "final")
        final_bytes = file_bytes(remote / "final" / "model.safetensors")
    else:
        from safetensors.torch import save_file
        started = time.perf_counter()
        state = adapters.adapter_state(model, effective=True)
        save_file(state, output / "adapter.safetensors")
        atomic_json(output / "adapter_config.json", {**cfg["adapter"], "scaling": cfg["adapter"]["alpha"] / cfg["adapter"]["rank"],
                                                   "exported_factors": "projected" if cfg["adapter"]["protect_k"] else "plain"})
        export_seconds = time.perf_counter() - started
        final_bytes = file_bytes(output / "adapter.safetensors")
        overhead["export_seconds"] = export_seconds
    save_down_proj(model, cfg, remote / "final" / "down_proj.pt")
    if penalty:
        overhead.update(penalty_seconds=penalty.penalty_seconds, transfer_seconds=penalty.transfer_seconds)
    run.finish(cfg["results_dir"], trainable_parameters=trainable, training_seconds=training_seconds,
               skipped_updates=skipped_total, updates=update, examples=update * per_update,
               final_artifact_bytes=final_bytes, final_export_seconds=export_seconds,
               displacement_l2=displacements, overhead=overhead, smoke=bool(args.max_updates))
    # The run is complete and its sizes are recorded; drop recovery state to stay within Kaggle's disk.
    (output / "training_state.pt").unlink(missing_ok=True)


def _log_rows(output):
    path = output / "train.jsonl"
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


if __name__ == "__main__":
    main()
