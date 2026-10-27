"""MetaMathQA selection, chat formatting with response-only labels, and probe-set sampling."""
from __future__ import annotations

import argparse
import re
import unicodedata

import numpy as np
import torch

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.runtime import atomic_json, fingerprint

IGNORE = -100


def normalize_text(text):
    """Normalization used for GSM8K overlap: NFKC, lowercase, alphanumerics only, single spaces."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    return " ".join(re.sub(r"[^0-9a-z]+", " ", text).split())


def load_tokenizer(cfg):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["models"]["qwen3"], revision=cfg["revisions"]["qwen3"])
    tokenizer.padding_side = "right"
    return tokenizer


def prompt_text(tokenizer, question, system=None):
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def encode_example(tokenizer, question, response):
    """Prompt tokens (masked) followed by response tokens and the end-of-turn token (supervised).

    The prompt is the exact non-thinking generation prompt used at inference, so training and
    generation share one boundary.
    """
    prompt_ids = tokenizer(prompt_text(tokenizer, question), add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    eot = tokenizer.convert_tokens_to_ids("<|im_end|>")
    input_ids = prompt_ids + response_ids + [eot]
    labels = [IGNORE] * len(prompt_ids) + response_ids + [eot]
    return input_ids, labels


def gsm8k_test_questions(cfg):
    from datasets import load_dataset
    test = load_dataset("openai/gsm8k", "main", split="test", revision=cfg["revisions"]["gsm8k"])
    return {normalize_text(q) for q in test["question"]}


def prepare(cfg, limit=None):
    from datasets import load_dataset
    output = repo_path(cfg["data"]["output"])
    manifest = repo_path(cfg["data"]["manifest"])
    target = limit or cfg["data"]["train_examples"]
    if output.exists() and manifest.exists():
        rows = read_jsonl(output)
        if len(rows) == target:
            print("training selection exists", output)
            return rows
    tokenizer = load_tokenizer(cfg)
    banned = gsm8k_test_questions(cfg)
    source = load_dataset("meta-math/MetaMathQA", split="train", revision=cfg["revisions"]["metamathqa"])
    rows, excluded = [], {"empty": 0, "too_long": 0, "gsm8k_overlap": 0}
    for index, example in enumerate(source):
        query, response = (example.get("query") or "").strip(), (example.get("response") or "").strip()
        if not query or not response:
            excluded["empty"] += 1
            continue
        if normalize_text(query) in banned or normalize_text(example.get("original_question")) in banned:
            excluded["gsm8k_overlap"] += 1
            continue
        input_ids, labels = encode_example(tokenizer, example["query"], example["response"])
        if len(input_ids) > cfg["data"]["max_tokens"]:
            excluded["too_long"] += 1
            continue
        rows.append({"source_index": index, "type": example.get("type"), "input_ids": input_ids, "labels": labels})
        if len(rows) == target:
            break
    if len(rows) != target:
        raise ValueError(f"Only {len(rows)} eligible examples")
    write_jsonl(output, rows)
    lengths = np.array([len(r["input_ids"]) for r in rows])
    response_tokens = np.array([sum(x != IGNORE for x in r["labels"]) for r in rows])
    atomic_json(manifest, {
        "source": "meta-math/MetaMathQA", "revision": cfg["revisions"]["metamathqa"],
        "selected": len(rows), "scanned_through_source_index": rows[-1]["source_index"],
        "excluded": excluded, "normalization": "NFKC, lowercase, non-alphanumerics to spaces, collapse whitespace",
        "source_indices": [r["source_index"] for r in rows], "sha256": fingerprint([r["input_ids"] for r in rows]),
        "length_tokens": {"mean": float(lengths.mean()), "p50": float(np.median(lengths)),
                          "p99": float(np.percentile(lengths, 99)), "max": int(lengths.max())},
        "response_tokens_total": int(response_tokens.sum())})
    print("selected", len(rows), "excluded", excluded)
    return rows


def sample_probes(cfg):
    """Fixed activation-probe IDs drawn before any model results are inspected."""
    path = repo_path(cfg["data"]["probes"])
    if path.exists():
        return load_json(path)
    from datasets import load_dataset
    rng = np.random.default_rng(cfg["seed"])
    gsm = load_dataset("openai/gsm8k", "main", split="test", revision=cfg["revisions"]["gsm8k"])
    hella = load_dataset("Rowan/hellaswag", split="validation")
    n = cfg["data"]["probe_per_task"]
    probes = {"seed": cfg["seed"],
              "gsm8k_test": sorted(int(i) for i in rng.choice(len(gsm), n, replace=False)),
              "hellaswag_validation": sorted(int(i) for i in rng.choice(len(hella), n, replace=False))}
    atomic_json(path, probes)
    return probes


def load_json(path):
    import json
    return json.loads(repo_path(path).read_text())


def batches(rows, microbatch, pad_id):
    """Consecutive microbatches in source order, right-padded."""
    for start in range(0, len(rows), microbatch):
        chunk = rows[start:start + microbatch]
        width = max(len(r["input_ids"]) for r in chunk)
        ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        labels = torch.full((len(chunk), width), IGNORE, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for i, r in enumerate(chunk):
            n = len(r["input_ids"])
            ids[i, :n] = torch.tensor(r["input_ids"])
            labels[i, :n] = torch.tensor(r["labels"])
            mask[i, :n] = 1
        yield {"input_ids": ids, "labels": labels, "attention_mask": mask}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task1_base.yaml")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    if args.limit:
        cfg["data"]["output"] = cfg["data"]["output"].replace(".jsonl", f"_limit{args.limit}.jsonl")
        cfg["data"]["manifest"] = cfg["data"]["manifest"].replace(".json", f"_limit{args.limit}.json")
    prepare(cfg, args.limit)
    sample_probes(cfg)


if __name__ == "__main__":
    main()
