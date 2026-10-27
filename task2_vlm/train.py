"""Three-stage VLM curriculum: caption alignment, visual instruction tuning, natural VQA.

One model continues through all stages (4,500 updates, effective batch 16). Each draw yields two
examples: a complete question pair, or two caption examples for replay. A seeded sampler chooses
components by the stage probabilities without batch quotas. Stage optimizers start fresh with a
per-stage 5% warmup and cosine decay to 10% of the peak learning rate.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import set_seed
from common.runtime import Run, atomic_json, completed, optimizer_step
from task1_finetune import adapters
from task1_finetune.data import load_tokenizer
from task2_vlm.model import VLM, Connector, FeatureStore, Formatter, calibrate_scale, collate, pool_features


def load_data(cfg):
    d = repo_path(cfg["data"]["dir"])
    data = {name: read_jsonl(d / f"{name}.jsonl") for name in
            ("caption_train", "caption_dev", "presence_pool", "counting_pool", "vqa_train",
             "grounding_dev", "grounding_test", "vqa_dev", "vqa_test")}
    return data


def caption_example(row):
    return {"kind": "caption", "example_id": row["example_id"], "key": f"{row['split']}:{row['image_id']}",
            "target": row["caption"]}


def pair_examples(pair, image_override=None):
    """Two question examples of a pair; image_override maps side -> feature key."""
    out = []
    for side, item in enumerate(pair["items"]):
        key = f"{item['split']}:{item['image_id']}"
        if image_override is not None:
            key = image_override[side]
        out.append({"kind": pair["kind"], "example_id": item["example_id"], "pair_id": pair["pair_id"],
                    "question": pair["question"], "target": item["answer"], "key": key})
    return out


class Sampler:
    def __init__(self, data, seed):
        self.data = data
        self.rng = np.random.default_rng(seed)
        self.caption_order = None
        self.cursor = 0

    def state(self):
        return {"rng": self.rng.bit_generator.state, "cursor": self.cursor,
                "caption_order": None if self.caption_order is None else self.caption_order.tolist()}

    def load(self, state):
        self.rng.bit_generator.state = state["rng"]
        self.cursor = state["cursor"]
        self.caption_order = None if state["caption_order"] is None else np.array(state["caption_order"])

    def next_caption_pass(self):
        """Stage 1: two passes over all caption examples, each pass in a fresh seeded order."""
        if self.caption_order is None:
            n = len(self.data["caption_train"])
            self.caption_order = np.concatenate([self.rng.permutation(n), self.rng.permutation(n)])
        rows = [self.data["caption_train"][int(i)] for i in self.caption_order[self.cursor:self.cursor + 2]]
        self.cursor += 2
        return "caption", [caption_example(r) for r in rows]

    def draw(self, mixture):
        if list(mixture) == ["caption"]:
            return self.next_caption_pass()
        names = list(mixture)
        component = names[int(self.rng.choice(len(names), p=[mixture[n] for n in names]))]
        if component == "caption":
            idx = self.rng.integers(len(self.data["caption_train"]), size=2)
            return component, [caption_example(self.data["caption_train"][int(i)]) for i in idx]
        pool = {"presence": "presence_pool", "counting": "counting_pool", "vqa": "vqa_train"}[component]
        pair = self.data[pool][int(self.rng.integers(len(self.data[pool])))]
        return component, pair_examples(pair)


def stage_of(update, stages):
    start = 0
    for index, stage in enumerate(stages):
        if update < start + stage["updates"]:
            return index, start
        start += stage["updates"]
    return len(stages) - 1, start - stages[-1]["updates"]


def lr_lambda(warmup, total, final_fraction):
    def fn(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return final_fraction + (1 - final_fraction) * 0.5 * (1 + math.cos(math.pi * progress))
    return fn


def build_optimizer(vlm, stage, cfg):
    t = cfg["train"]
    lora_on = stage["lora_lr"] is not None
    for m in adapters.adapter_modules(vlm.lm).values():
        m.lora_A.requires_grad_(lora_on)
        m.lora_B.requires_grad_(lora_on)
    groups = []
    def add(params, lr):
        params = list(params)
        groups.append({"params": [p for p in params if p.ndim >= 2], "lr": lr, "weight_decay": t["weight_decay"]})
        groups.append({"params": [p for p in params if p.ndim < 2], "lr": lr, "weight_decay": 0.0})
    add(vlm.connector.parameters(), stage["connector_lr"])
    if lora_on:
        add([p for m in adapters.adapter_modules(vlm.lm).values() for p in (m.lora_A, m.lora_B)], stage["lora_lr"])
    groups = [g for g in groups if g["params"]]
    optimizer = torch.optim.AdamW(groups)
    warmup = max(1, round(t["warmup_fraction"] * stage["updates"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(warmup, stage["updates"], t["final_lr_fraction"]))
    return optimizer, scheduler


def build_vlm(cfg, tokenizer):
    from transformers import AutoModelForCausalLM
    lm = AutoModelForCausalLM.from_pretrained(cfg["models"]["qwen3"], revision=cfg["revisions"]["qwen3"],
                                              torch_dtype=torch.float16, attn_implementation="sdpa").cuda()
    for p in lm.parameters():
        p.requires_grad_(False)
    l = cfg["lora"]
    # Attached from the start with zero B (an exact no-op); trainable only in stages 2-3.
    adapters.attach(lm, l["targets"], l["rank"], l["alpha"], l["dropout"], 0, cfg["seed"])
    c = cfg["connector"]
    connector = Connector(c["input_dim"], c["hidden_dim"], c["output_dim"]).cuda()
    return VLM(lm, connector, Formatter(tokenizer, cfg, cfg["visual_tokens"]))


def features_for(store, keys, cfg):
    return pool_features(store.get(keys), cfg["visual_tokens"]).cuda(non_blocking=True)


def batch_loss(vlm, examples, store, cfg, tokenizer):
    encoded = [vlm.fmt.encode(e) for e in examples]
    batch = {k: v.cuda(non_blocking=True) for k, v in collate(encoded, tokenizer.pad_token_id).items()}
    with torch.autocast("cuda", dtype=torch.float16):
        return vlm.example_mean_loss(batch, features_for(store, [e["key"] for e in examples], cfg))


@torch.no_grad()
def dev_metrics(vlm, data, store, cfg, tokenizer):
    """Caption reference loss; reference-answer NLL under correct, complementary and no-image inputs."""
    vlm.eval()
    bs = cfg["eval"]["dev_batch_size"]
    result = {}
    captions = [caption_example(r) for r in data["caption_dev"]]
    losses = []
    for s in range(0, len(captions), bs):
        losses += batch_loss(vlm, captions[s:s + bs], store, cfg, tokenizer).tolist()
    result["caption_loss"] = float(np.mean(losses))
    for suite in ("grounding_dev", "vqa_dev"):
        per = {}
        for condition in ("correct", "complementary", "no_image"):
            examples = []
            for pair in data[suite]:
                keys = [f"{x['split']}:{x['image_id']}" for x in pair["items"]]
                override = keys[::-1] if condition == "complementary" else None
                examples += pair_examples(pair, override)
            per[condition] = answer_nll(vlm, examples, store, cfg, tokenizer, image=condition != "no_image")
        result[suite] = {f"nll_{c}": float(np.mean(v)) for c, v in per.items()}
        result[suite]["delta_nll_swap"] = float(np.mean(np.array(per["complementary"]) - np.array(per["correct"])))
    vlm.train()
    return result


@torch.no_grad()
def answer_nll(vlm, examples, store, cfg, tokenizer, image=True):
    values = []
    bs = cfg["eval"]["dev_batch_size"]
    for s in range(0, len(examples), bs):
        chunk = examples[s:s + bs]
        encoded = [vlm.fmt.encode(e, image=image) for e in chunk]
        batch = {k: v.cuda() for k, v in collate(encoded, tokenizer.pad_token_id).items()}
        features = features_for(store, [e["key"] for e in chunk], cfg) if image else None
        with torch.autocast("cuda", dtype=torch.float16):
            values += vlm.answer_nll(batch, features).tolist()
    return values


def save_stage(vlm, path):
    from safetensors.torch import save_file
    state = {f"connector.{k}": v.detach().float().cpu().contiguous() for k, v in vlm.connector.state_dict().items()}
    state.update({f"lm.{k}": v for k, v in adapters.adapter_state(vlm.lm).items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task2_vlm.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-updates", type=int, help="Smoke test: updates per stage")
    parser.add_argument("--name")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    if args.name:
        cfg["name"] = args.name
    if args.max_updates:
        for stage in cfg["train"]["stages"]:
            stage["updates"] = args.max_updates
        cfg["eval"]["dev_every"] = args.max_updates
        cfg["train"]["checkpoint_every"] = args.max_updates
    name, t = cfg["name"], cfg["train"]
    output = repo_path(cfg["outputs_dir"]) / name
    results = repo_path(cfg["results_dir"]) / name
    if completed(output):
        print("already complete", output)
        return
    set_seed(cfg["seed"])
    tokenizer = load_tokenizer(cfg)
    data = load_data(cfg)
    store = FeatureStore(cfg)
    vlm = build_vlm(cfg, tokenizer)
    if t["gradient_checkpointing"]:
        vlm.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    vlm.lm.config.use_cache = False
    stages = t["stages"]
    total = sum(s["updates"] for s in stages)
    per_update = t["microbatch"] * t["grad_accum"]
    example_ids = [r["example_id"] for r in data["caption_train"]] + [p["pair_id"] for k in ("presence_pool", "counting_pool", "vqa_train") for p in data[k]]
    run = Run(output, cfg, example_ids, name, resume=args.resume)
    progress = run.state["progress"] if run.state else {"update": 0, "cursor": 0}
    stage_index, _ = stage_of(progress["update"], stages)
    if progress["update"] == total:
        stage_index = len(stages) - 1
    if run.state is None:
        # Calibrate the connector output scale on training data only.
        rng = np.random.default_rng(cfg["seed"])
        rows = rng.choice(len(data["caption_train"]), cfg["connector"]["calibration_images"], replace=False)
        keys = sorted({caption_example(data["caption_train"][int(i)])["key"] for i in rows})
        prompts = [vlm.fmt.encode(caption_example(data["caption_train"][int(i)]))["input_ids"] for i in rows]
        token_ids = torch.tensor([x for p in prompts for x in p if x != vlm.fmt.visual_id])
        calibration = calibrate_scale(vlm.connector, vlm.lm, features_for(store, keys, cfg), token_ids)
        atomic_json(results / "connector_calibration.json", calibration)
    optimizer, scheduler = build_optimizer(vlm, stages[stage_index], cfg)
    scaler = torch.amp.GradScaler("cuda")
    sampler = Sampler(data, cfg["seed"])
    if run.state:
        run.restore({"connector": vlm.connector, "lm": vlm.lm}, {"stage": optimizer}, {"stage": scheduler}, scaler)
        vlm.connector.scale.fill_(run.state["progress"]["connector_scale"])
        sampler.load(run.state["progress"]["sampler"])
    dev_path = results / "dev.jsonl"
    done_dev = {r["update"] for r in read_jsonl(dev_path)} if dev_path.exists() else set()
    def maybe_dev(update):
        boundaries = np.cumsum([s["updates"] for s in stages]).tolist()
        if update in done_dev or not (update % cfg["eval"]["dev_every"] == 0 or update in boundaries):
            return
        started = time.perf_counter()
        metrics = dev_metrics(vlm, data, store, cfg, tokenizer)
        row = {"update": update, "stage": stages[stage_of(max(update - 1, 0), stages)[0]]["name"],
               "seconds": time.perf_counter() - started, **metrics}
        with dev_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        done_dev.add(update)
        print("dev", json.dumps(row), flush=True)
    results.mkdir(parents=True, exist_ok=True)
    vlm.train()
    update = progress["update"]
    maybe_dev(update)
    stage_stats = json.loads((results / "stage_stats.json").read_text()) if (results / "stage_stats.json").exists() else {}
    stage_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    while update < total:
        index, start = stage_of(update, stages)
        if index != stage_index:
            stage_index = index
            optimizer, scheduler = build_optimizer(vlm, stages[index], cfg)
            stage_started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
        stage = stages[index]
        params = [p for g in optimizer.param_groups for p in g["params"]]
        step_start = time.perf_counter()
        draws = [sampler.draw(stage["mixture"]) for _ in range(per_update // 2)]
        examples = [e for _, pair in draws for e in pair]
        loss_total = 0.0
        for s in range(0, per_update, t["microbatch"]):
            losses = batch_loss(vlm, examples[s:s + t["microbatch"]], store, cfg, tokenizer)
            scaler.scale(losses.sum() / per_update).backward()
            loss_total += float(losses.detach().sum())
        if not math.isfinite(loss_total):
            raise FloatingPointError(f"Non-finite training loss at update {update + 1}")
        lrs = [g["lr"] for g in optimizer.param_groups]
        norm, skipped = optimizer_step(params, optimizer, scaler, t["max_grad_norm"])
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        update += 1
        mixture = {}
        for component, _ in draws:
            mixture[component] = mixture.get(component, 0) + 1
        run.log({"update": update, "stage": stage["name"], "loss": loss_total / per_update, "grad_norm": norm,
                 "skipped": skipped, "connector_lr": lrs[0], "lora_lr": lrs[2] if len(lrs) > 2 else None,
                 "draws": mixture, "step_seconds": time.perf_counter() - step_start, "elapsed_seconds": run.seconds()})
        if update % 50 == 0:
            print(json.dumps({"update": update, "stage": stage["name"], "loss": loss_total / per_update}), flush=True)
        boundary = update == start + stage["updates"]
        if boundary:
            save_stage(vlm, output / f"stage{index + 1}.safetensors")
            trainable = sum(p.numel() for p in params)
            stage_stats[stage["name"]] = {"trainable_parameters": trainable, "updates": stage["updates"],
                                          "seconds": time.perf_counter() - stage_started,
                                          "peak_memory_bytes": torch.cuda.max_memory_allocated()}
            atomic_json(results / "stage_stats.json", stage_stats)
        if update % t["checkpoint_every"] == 0 or boundary:
            run.save({"connector": vlm.connector, "lm": vlm.lm}, {"stage": optimizer}, {"stage": scheduler}, scaler,
                     {"update": update, "cursor": update * per_update, "sampler": sampler.state(),
                      "connector_scale": float(vlm.connector.scale)})
        maybe_dev(update)
    draws_total = {}
    for row in read_jsonl(output / "train.jsonl"):
        for k, v in row["draws"].items():
            draws_total.setdefault(row["stage"], {}).setdefault(k, 0)
            draws_total[row["stage"]][k] += v
    run.finish(results, stage_stats=stage_stats, mixture_counts=draws_total, smoke=bool(args.max_updates),
               example_presentations=update * per_update)


if __name__ == "__main__":
    main()
