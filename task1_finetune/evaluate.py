"""Benchmark and qualitative evaluation for one Task 1 condition.

Conditions: the unchanged baseline, Full FT midpoint/final snapshots (FP16), and adapter runs
merged into a fresh FP16 backbone (OPLoRA uses its exported projected factors).
"""
from __future__ import annotations

import argparse
import json
import time

import torch
import yaml

from common.data import load_yaml, repo_path, write_jsonl
from common.logging_utils import set_seed
from common.runtime import atomic_json, cuda_peaks, package_versions, reset_peaks
from task1_finetune import adapters
from task1_finetune.data import load_tokenizer, prompt_text
from task1_finetune.train import load_model


def condition_name(cfg, stage):
    return "baseline" if stage == "baseline" else f"{cfg['name']}_{stage}"


def load_condition(cfg, stage, tokenizer):
    """Returns an FP16 eval-mode model and merge diagnostics."""
    info = {"stage": stage}
    if stage == "baseline":
        model = load_model(cfg, torch.float16)
    elif cfg["method"] == "full":
        from transformers import AutoModelForCausalLM
        path = repo_path(cfg["remote_dir"]) / cfg["name"] / stage
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, attn_implementation="sdpa").cuda()
    else:
        from safetensors.torch import load_file
        state = load_file(repo_path(cfg["outputs_dir"]) / cfg["name"] / "adapter.safetensors")
        a = cfg["adapter"]
        scaling = a["alpha"] / a["rank"]
        unmerged = adapters.load_exported(load_model(cfg, torch.float16), state, a["rank"], a["alpha"]).eval()
        model = adapters.merge_into(load_model(cfg, torch.float16), state, scaling).eval()
        info["merge_check"] = merge_check(unmerged, model, tokenizer)
        del unmerged
        torch.cuda.empty_cache()
    model.eval()
    model.config.use_cache = True
    return model, info


@torch.no_grad()
def merge_check(unmerged, merged, tokenizer):
    texts = ["Natalia sold clips to 48 of her friends in April. How many clips did she sell?",
             "The capital of France is", "Write a short poem about the sea."]
    tokenizer.padding_side = "left"
    batch = tokenizer(texts, return_tensors="pt", padding=True).to("cuda")
    a = unmerged(**batch).logits.float()
    b = merged(**batch).logits.float()
    mask = batch["attention_mask"].bool()
    diff = (a - b).abs()[mask]
    agree = (a.argmax(-1) == b.argmax(-1))[mask].float().mean()
    return {"max_abs_logit_diff": float(diff.max()), "mean_abs_logit_diff": float(diff.mean()),
            "top1_agreement": float(agree), "tokens": int(mask.sum())}


def run_gsm8k(model, tokenizer, cfg, out_dir, limit, seeds):
    """Zero-shot generation with the non-thinking chat template and greedy decoding."""
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    e = cfg["eval"]
    tokenizer.padding_side = "left"
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=e["gen_batch_size"], enable_thinking=False)
    started = time.perf_counter()
    gsm = simple_evaluate(model=lm, tasks=[e["gsm8k_task"]], num_fewshot=0, apply_chat_template=True,
                          gen_kwargs=f"max_gen_toks={e['max_gen_toks']},do_sample=False", log_samples=True,
                          limit=limit, bootstrap_iters=0, **seeds)
    seconds = time.perf_counter() - started
    task = e["gsm8k_task"]
    metrics = gsm["results"][task]
    by_doc = {}
    for sample in gsm["samples"][task]:  # lm-eval logs one row per (document, answer filter)
        row = by_doc.setdefault(sample["doc_id"], {"doc_id": sample["doc_id"], "target": sample["target"],
                                                   "response": sample["resps"][0][0]})
        key = sample["filter"].replace("-", "_")
        row[f"extracted_{key}"] = sample["filtered_resps"][0]
        row[f"exact_match_{key}"] = sample["exact_match"]
    rows = [by_doc[k] for k in sorted(by_doc)]
    for row in rows:
        length = len(tokenizer(row["response"], add_special_tokens=False)["input_ids"])
        row.update(response_tokens=length, truncated=length >= e["max_gen_toks"])
    write_jsonl(out_dir / "gsm8k_samples.jsonl", rows)
    n = len(rows)
    result = {"primary": metrics["exact_match,flexible-extract"], "strict": metrics["exact_match,strict-match"],
              "n": n, "truncation_rate": sum(r["truncated"] for r in rows) / n,
              "unextracted_rate": sum(r["extracted_flexible_extract"] in ("[invalid]", None, "") for r in rows) / n,
              "mean_response_tokens": sum(r["response_tokens"] for r in rows) / n}
    return result, seconds, gsm.get("configs", {})


def run_benchmarks(model, tokenizer, cfg, name, limit):
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    e = cfg["eval"]
    out_dir = repo_path(cfg["results_dir"]) / "eval" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = dict(random_seed=cfg["seed"], numpy_random_seed=cfg["seed"], torch_random_seed=cfg["seed"],
                 fewshot_random_seed=cfg["seed"])
    summary = {}
    timings = {}
    # GSM8K results are kept as soon as they exist, so a later failure does not repeat the generation pass.
    partial = out_dir / "gsm8k_partial.json"
    if partial.exists():
        saved = json.loads(partial.read_text())
        summary["gsm8k"], timings["gsm8k_seconds"], gsm_configs = saved["summary"], saved["seconds"], saved["configs"]
    else:
        summary["gsm8k"], timings["gsm8k_seconds"], gsm_configs = run_gsm8k(model, tokenizer, cfg, out_dir, limit, seeds)
        atomic_json(partial, {"summary": summary["gsm8k"], "seconds": timings["gsm8k_seconds"], "configs": gsm_configs})
    # Likelihood benchmarks: stock completion prompts without a chat template.
    tokenizer.padding_side = "right"
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=e["loglik_batch_size"])
    started = time.perf_counter()
    ll = simple_evaluate(model=lm, tasks=e["likelihood_tasks"], num_fewshot=0, apply_chat_template=False,
                         log_samples=True, limit=limit, bootstrap_iters=0, **seeds)
    timings["likelihood_seconds"] = time.perf_counter() - started
    for task in e["likelihood_tasks"]:
        metrics = ll["results"][task]
        primary = metrics.get("acc_norm,none", metrics["acc,none"])
        summary[task] = {"primary": primary, "acc": metrics["acc,none"], "acc_norm": metrics.get("acc_norm,none"),
                         "primary_metric": "acc_norm" if "acc_norm,none" in metrics else "acc",
                         "n": len(ll["samples"][task])}
        rows = [{"doc_id": s["doc_id"], "target": s["target"],
                 "loglik": [float(r[0]) for r in s["filtered_resps"]],
                 "acc": s.get("acc"), "acc_norm": s.get("acc_norm")} for s in ll["samples"][task]]
        write_jsonl(out_dir / f"{task}_samples.jsonl", rows)
    return summary, timings, {"lm_eval_configs": {**gsm_configs, **ll.get("configs", {})}}


@torch.no_grad()
def qualitative(model, tokenizer, cfg, name):
    path = repo_path(cfg["eval"]["qualitative_file"])
    questions = yaml.safe_load(path.read_text())
    if len(questions) != 24:
        raise ValueError("Expected 24 qualitative prompts")
    tokenizer.padding_side = "left"
    rows = []
    for q in questions:
        fmt = "json" if q["id"].endswith("_json") else "template" if q["id"].endswith("_template") else "plain"
        base_id = q["id"].removesuffix("_json").removesuffix("_template")
        text = prompt_text(tokenizer, q["paraphrases"][0], q.get("system"))
        batch = tokenizer([text], return_tensors="pt", add_special_tokens=False).to("cuda")
        output = model.generate(**batch, do_sample=False, max_new_tokens=cfg["eval"]["qualitative_max_new_tokens"],
                                pad_token_id=tokenizer.pad_token_id)
        new = output[0, batch["input_ids"].shape[1]:]
        eos = {tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id}
        finished = any(int(t) in eos for t in new)
        rows.append({"condition": name, "question_id": base_id, "format": fmt, "prompt_id": q["id"],
                     "system": q.get("system"), "prompt": q["paraphrases"][0],
                     "response": tokenizer.decode(new, skip_special_tokens=True),
                     "new_tokens": int(len(new)), "truncated": not finished})
    write_jsonl(repo_path(cfg["results_dir"]) / "qualitative" / f"{name}.jsonl", rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["baseline", "midpoint", "final"], required=True)
    parser.add_argument("--limit", type=int, help="Smoke test: examples per benchmark")
    parser.add_argument("--name")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    if args.name:
        cfg["name"] = args.name
    name = condition_name(cfg, args.stage) + (f"_limit{args.limit}" if args.limit else "")
    summary_path = repo_path(cfg["results_dir"]) / "eval" / name / "summary.json"
    if summary_path.exists():
        print("already evaluated", name)
        return
    set_seed(cfg["seed"])
    tokenizer = load_tokenizer(cfg)
    reset_peaks()
    model, info = load_condition(cfg, args.stage, tokenizer)
    started = time.perf_counter()
    scores, timings, configs = run_benchmarks(model, tokenizer, cfg, name, args.limit)
    qual_started = time.perf_counter()
    qualitative(model, tokenizer, cfg, name)
    timings["qualitative_seconds"] = time.perf_counter() - qual_started
    atomic_json(summary_path, {"condition": name, "scores": scores, "load": info, "timings": timings,
                               "eval_seconds": time.perf_counter() - started, "peak_memory": cuda_peaks(),
                               "versions": package_versions(), "limit": args.limit, **configs})
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
