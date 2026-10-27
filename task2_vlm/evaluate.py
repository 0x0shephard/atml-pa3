"""Final grounding evaluation of the Stage 3 model on both test suites under four image conditions.

The question and original references stay fixed: correct image, complementary swap (partner image),
mismatched (an image from another pair in the same suite, fixed before generation), and no image.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from common.data import load_yaml, repo_path, write_jsonl
from common.logging_utils import set_seed
from common.runtime import atomic_json, cuda_peaks, reset_peaks
from task1_finetune import adapters
from task1_finetune.data import load_tokenizer
from task2_vlm.metrics import normalize_answer, pair_bootstrap, soft_accuracy
from task2_vlm.model import FeatureStore, collate
from task2_vlm.train import answer_nll, build_vlm, features_for, load_data, pair_examples

CONDITIONS = ("correct", "complementary", "mismatched", "no_image")


def mismatched_mapping(pairs, seed):
    """A derangement of pairs: each pair borrows the same-side image of a different pair, so neither of
    its own two images is used. Fixed once and reused across comparisons."""
    rng = np.random.default_rng(seed)
    n = len(pairs)
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            break
    return {pairs[i]["pair_id"]: pairs[int(perm[i])]["pair_id"] for i in range(n)}


def condition_examples(pairs, condition, mapping):
    lookup = {p["pair_id"]: p for p in pairs}
    out = []
    for pair in pairs:
        keys = [f"{x['split']}:{x['image_id']}" for x in pair["items"]]
        if condition == "complementary":
            override = keys[::-1]
        elif condition == "mismatched":
            other = lookup[mapping[pair["pair_id"]]]
            override = [f"{x['split']}:{x['image_id']}" for x in other["items"]]
        else:
            override = None
        out += pair_examples(pair, override)
    return out


def load_final(cfg, tokenizer, stage_file):
    from safetensors.torch import load_file
    vlm = build_vlm(cfg, tokenizer)
    state = load_file(repo_path(cfg["outputs_dir"]) / cfg["name"] / stage_file)
    vlm.connector.load_state_dict({k[len("connector."):]: v for k, v in state.items() if k.startswith("connector.")})
    lora = {k[len("lm."):]: v for k, v in state.items() if k.startswith("lm.")}
    for name, module in adapters.adapter_modules(vlm.lm).items():
        module.lora_A.data.copy_(lora[f"{name}.lora_A"])
        module.lora_B.data.copy_(lora[f"{name}.lora_B"])
    vlm.lm.config.use_cache = True
    return vlm.eval()


@torch.no_grad()
def generate_all(vlm, examples, store, cfg, tokenizer, image):
    predictions, seconds = [], []
    bs, max_new = cfg["eval"]["generation_batch_size"], cfg["eval"]["max_new_tokens"]
    for s in range(0, len(examples), bs):
        chunk = examples[s:s + bs]
        encoded = [vlm.fmt.encode(e, image=image, with_target=False) for e in chunk]
        batch = {k: v.cuda() for k, v in collate(encoded, tokenizer.pad_token_id, left=True).items()}
        features = features_for(store, [e["key"] for e in chunk], cfg) if image else None
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            texts = vlm.generate(batch, features, max_new)
        torch.cuda.synchronize()
        seconds += [(time.perf_counter() - started) / len(chunk)] * len(chunk)
        predictions += texts
    return predictions, seconds


def score(suite_kind, pair, side, prediction):
    item = pair["items"][side]
    normalized = normalize_answer(prediction)
    if suite_kind == "vqa":
        return soft_accuracy(prediction, item["answers"]), normalized == item["canonical"]
    correct = float(normalized == normalize_answer(item["answer"]))
    return correct, bool(correct)


def summarize(rows, pairs, suite_kind, cfg):
    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["pair_id"], []).append(r)
    type_key = "answer_type" if suite_kind == "vqa" else "kind"
    out = {"n_questions": len(rows), "n_pairs": len(by_pair)}
    out["accuracy"] = float(np.mean([r["score"] for r in rows]))
    for t in sorted({p[type_key] for p in pairs}):
        out[f"accuracy_{t}"] = float(np.mean([r["score"] for r in rows if r["type"] == t]))
    out["pair_accuracy"] = float(np.mean([all(r["canonical_match"] for r in v) for v in by_pair.values()]))
    out["identical_answer_rate"] = float(np.mean([v[0]["normalized"] == v[1]["normalized"] for v in by_pair.values()]))
    out["empty_answer_rate"] = float(np.mean([r["normalized"] == "" for r in rows]))
    out["follows_partner_rate"] = float(np.mean([r["matches_partner"] for r in rows]))
    out["mean_reference_nll"] = float(np.mean([r["nll"] for r in rows]))
    out["generation_seconds_per_answer"] = float(np.mean([r["seconds"] for r in rows]))
    return out


def per_pair(rows, field):
    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["pair_id"], []).append(r[field])
    return {k: float(np.mean(v)) for k, v in sorted(by_pair.items())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task2_vlm.yaml")
    parser.add_argument("--name")
    parser.add_argument("--stage-file", default="stage3.safetensors")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    if args.name:
        cfg["name"] = args.name
    results = repo_path(cfg["results_dir"]) / cfg["name"]
    if (results / "final_summary.json").exists():
        print("already evaluated")
        return
    set_seed(cfg["seed"])
    tokenizer = load_tokenizer(cfg)
    data = load_data(cfg)
    store = FeatureStore(cfg)
    vlm = load_final(cfg, tokenizer, args.stage_file)
    reset_peaks()
    summary, mappings = {}, {}
    for suite, kind in (("grounding_test", "grounding"), ("vqa_test", "vqa")):
        pairs = data[suite]
        mapping = mismatched_mapping(pairs, cfg["seed"])
        mappings[suite] = mapping
        lookup = {p["pair_id"]: p for p in pairs}
        rows_by_condition = {}
        for condition in CONDITIONS:
            examples = condition_examples(pairs, condition, mapping)
            image = condition != "no_image"
            predictions, seconds = generate_all(vlm, examples, store, cfg, tokenizer, image)
            nll = answer_nll(vlm, examples, store, cfg, tokenizer, image=image)
            rows = []
            for e, prediction, sec, value in zip(examples, predictions, seconds, nll):
                pair, side = lookup[e["pair_id"]], int(e["example_id"].rsplit("_", 1)[1])
                s, canonical_match = score(kind, pair, side, prediction)
                partner = pair["items"][1 - side]
                partner_answer = partner["canonical"] if kind == "vqa" else normalize_answer(partner["answer"])
                rows.append({"suite": suite, "condition": condition, "pair_id": e["pair_id"], "example_id": e["example_id"],
                             "type": pair.get("answer_type", pair["kind"]), "question": e["question"],
                             "image_key": e["key"] if image else None, "reference": pair["items"][side]["answer"],
                             "prediction": prediction, "normalized": normalize_answer(prediction), "score": s,
                             "canonical_match": canonical_match, "matches_partner": normalize_answer(prediction) == partner_answer,
                             "nll": value, "seconds": sec})
            write_jsonl(results / "final" / f"{suite}_{condition}.jsonl", rows)
            rows_by_condition[condition] = rows
            summary.setdefault(suite, {})[condition] = summarize(rows, pairs, kind, cfg)
        # Correct-image advantages over complementary swaps, with matched pair resamples.
        for other in ("complementary", "mismatched", "no_image"):
            acc = per_pair(rows_by_condition["correct"], "score")
            acc_other = per_pair(rows_by_condition[other], "score")
            nll = per_pair(rows_by_condition["correct"], "nll")
            nll_other = per_pair(rows_by_condition[other], "nll")
            ids = sorted(acc)
            summary[suite][f"advantage_vs_{other}"] = {
                "accuracy": pair_bootstrap([acc[i] - acc_other[i] for i in ids], cfg["eval"]["bootstrap"], cfg["seed"]),
                "nll": pair_bootstrap([nll_other[i] - nll[i] for i in ids], cfg["eval"]["bootstrap"], cfg["seed"])}
        for condition in CONDITIONS:
            pa = per_pair([{**r, "both": float(r["canonical_match"])} for r in rows_by_condition[condition]], "both")
            summary[suite][condition]["pair_accuracy_ci"] = pair_bootstrap(
                [float(v == 1.0) for _, v in sorted(pa.items())], cfg["eval"]["bootstrap"], cfg["seed"])
            summary[suite][condition]["accuracy_ci"] = pair_bootstrap(
                list(per_pair(rows_by_condition[condition], "score").values()), cfg["eval"]["bootstrap"], cfg["seed"])
    atomic_json(results / "mismatched_mapping.json", mappings)
    atomic_json(results / "final_summary.json", {"suites": summary, "peak_memory": cuda_peaks(),
                                                "bootstrap_resamples": cfg["eval"]["bootstrap"]})
    print(json.dumps({s: {c: v.get("accuracy") for c, v in d.items() if isinstance(v, dict) and "accuracy" in v}
                      for s, d in summary.items()}, indent=1))


if __name__ == "__main__":
    main()
