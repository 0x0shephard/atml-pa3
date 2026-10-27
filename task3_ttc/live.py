"""Live MATH comparison on 50 fixed held-out questions with a 3,072-token output allowance each:
one long response, three parallel responses with self-consistency, and sequential refinement
(the first parallel response, then one critique-and-revision turn). Batch size 1, FP16, no gold
feedback; every call is independently seeded and fully logged.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.runtime import atomic_json, package_versions, reset_peaks
from task3_ttc.answers import NO_ANSWER, group_math
from task3_ttc.load import load_dataset, splits
from task3_ttc.selection import vote_winner


def last_boxed(text):
    """Contents of the last \\boxed{...}, matching braces so nested groups survive."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i, depth, out = start + len("\\boxed{"), 1, []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip() or None
        out.append(ch)
        i += 1
    return None


class Generator:
    def __init__(self, cfg):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        live = cfg["live"]
        if not live.get("revision"):
            raise ValueError("Set live.revision in configs/task3.yaml to the revision listed in the supplied README.MD")
        self.cfg = live
        self.tok = AutoTokenizer.from_pretrained(live["model"], revision=live["revision"])
        self.model = AutoModelForCausalLM.from_pretrained(live["model"], revision=live["revision"],
                                                          torch_dtype=torch.float16).cuda().eval()

    @torch.no_grad()
    def __call__(self, messages, max_new_tokens, seed):
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = self.tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize()
        started = time.perf_counter()
        out = self.model.generate(**ids, do_sample=True, temperature=self.cfg["temperature"], top_p=self.cfg["top_p"],
                                  top_k=self.cfg["top_k"], repetition_penalty=self.cfg["repetition_penalty"],
                                  max_new_tokens=max_new_tokens, pad_token_id=self.tok.eos_token_id)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        new = out[0, ids["input_ids"].shape[1]:].tolist()
        stops = {self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>")}
        finished = any(t in stops for t in new)
        response = self.tok.decode(new, skip_special_tokens=True)
        return {"prompt": text, "seed": seed, "input_tokens": int(ids["input_ids"].shape[1]),
                "output_tokens": len([t for t in new if t not in stops]) + (1 if finished else 0),
                "truncated": not finished, "seconds": seconds, "response": response, "answer": last_boxed(response),
                "max_new_tokens": max_new_tokens}


def seed_for(base, question_index, call):
    return int(base * 1000 + question_index * 10 + call)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task3.yaml")
    parser.add_argument("--limit", type=int, help="Smoke test: number of questions")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    live = cfg["live"]
    questions, _ = load_dataset(cfg, "math", keep_text=False)
    ids = splits(cfg, list(questions))["eval"][: cfg["splits"]["live_questions"]]
    if args.limit:
        ids = ids[: args.limit]
    out = repo_path(cfg["results_dir"]) / "live"
    out.mkdir(parents=True, exist_ok=True)
    path = out / ("calls.jsonl" if not args.limit else f"calls_limit{args.limit}.jsonl")
    done = {(r["qid"], r["call"]) for r in read_jsonl(path)} if path.exists() else set()
    generator = Generator(cfg)
    reset_peaks()
    started = time.perf_counter()
    for index, qid in enumerate(ids):
        q = questions[qid]
        if q.question is None:
            raise ValueError("The supplied generations need a problem-text column for live generation")
        base = [{"role": "system", "content": live["system"]},
                {"role": "user", "content": f"{q.question}\n\n{live['instruction']}"}]  # two newlines, per README
        calls = [("single", base, live["single_tokens"])]
        calls += [(f"parallel{k}", base, live["parallel_tokens"]) for k in range(live["parallel_samples"])]
        records = {}
        for c, (call, messages, cap) in enumerate(calls):
            if (qid, call) in done:
                continue
            record = {"qid": qid, "call": call, **generator(messages, cap, seed_for(cfg["seed"], index, c))}
            records[call] = record
            with path.open("a") as f:
                f.write(json.dumps(record) + "\n")
        if (qid, "revision") not in done:
            first = records.get("parallel0") or next(r for r in read_jsonl(path) if r["qid"] == qid and r["call"] == "parallel0")
            messages = base + [{"role": "assistant", "content": first["response"]},
                               {"role": "user", "content": live["review"]}]
            record = {"qid": qid, "call": "revision", **generator(messages, live["revision_tokens"], seed_for(cfg["seed"], index, 9))}
            with path.open("a") as f:
                f.write(json.dumps(record) + "\n")
        print(f"live {index + 1}/{len(ids)} {qid}", flush=True)
    atomic_json(out / ("run.json" if not args.limit else f"run_limit{args.limit}.json"),
                {"questions": ids, "wall_seconds": time.perf_counter() - started, "versions": package_versions(),
                 "peak_memory_bytes": torch.cuda.max_memory_allocated(), "gpu": torch.cuda.get_device_name(0),
                 "model": live["model"], "revision": live["revision"]})
    summarize(cfg, questions, ids, path, out, suffix="" if not args.limit else f"_limit{args.limit}")


def summarize(cfg, questions, ids, path, out, suffix=""):
    calls = {}
    for r in read_jsonl(path):
        calls.setdefault(r["qid"], {})[r["call"]] = r
    rows = []
    for qid in ids:
        c, q = calls[qid], questions[qid]
        parallel = [c[f"parallel{k}"] for k in range(cfg["live"]["parallel_samples"])]
        answers = [c["single"]["answer"]] + [p["answer"] for p in parallel] + [c["revision"]["answer"]]
        groups, correct = group_math(answers, q.gold)
        is_correct = lambda i: bool(groups[i] >= 0 and correct[int(groups[i])])
        # Self-consistency over the three parallel answers; NO_ANSWER is its own (incorrect) group.
        pgroups = [int(g) if g >= 0 else -1 for g in groups[1:4]]
        winner, _ = vote_winner(pgroups, np.ones(3))
        sc_index = 1 + pgroups.index(winner)
        initial_ok, revised_ok = is_correct(1), is_correct(4)
        def cost(records):
            return {"input_tokens": sum(r["input_tokens"] for r in records), "output_tokens": sum(r["output_tokens"] for r in records),
                    "seconds": sum(r["seconds"] for r in records), "truncated_calls": sum(r["truncated"] for r in records),
                    "unextracted_calls": sum(r["answer"] is None for r in records)}
        rows.append({"qid": qid,
                     "single": {"correct": is_correct(0), "answer": c["single"]["answer"] or NO_ANSWER, **cost([c["single"]])},
                     "parallel": {"correct": bool(is_correct(sc_index) and winner >= 0), "answer": answers[sc_index] or NO_ANSWER,
                                  **cost(parallel)},
                     "refinement": {"correct": revised_ok, "initial_correct": initial_ok, "answer": c["revision"]["answer"] or NO_ANSWER,
                                    **cost([parallel[0], c["revision"]])},
                     "transition": f"{'C' if initial_ok else 'W'}->{'C' if revised_ok else 'W'}"})
    summary = {}
    for strategy in ("single", "parallel", "refinement"):
        values = [r[strategy] for r in rows]
        calls_per = {"single": 1, "parallel": 3, "refinement": 2}[strategy]
        summary[strategy] = {"accuracy": float(np.mean([v["correct"] for v in values])),
                             **{k: float(np.mean([v[k] for v in values])) for k in ("input_tokens", "output_tokens", "seconds")},
                             "truncation_rate": float(np.sum([v["truncated_calls"] for v in values]) / (calls_per * len(values))),
                             "extraction_rate": 1 - float(np.sum([v["unextracted_calls"] for v in values]) / (calls_per * len(values))),
                             "final_answer_missing_rate": float(np.mean([v["answer"] == NO_ANSWER for v in values]))}
    summary["refinement"]["transitions"] = {t: sum(r["transition"] == t for r in rows) for t in ("W->C", "C->W", "C->C", "W->W")}
    atomic_json(out / f"summary{suffix}.json", {"strategies": summary, "per_question": rows})
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
