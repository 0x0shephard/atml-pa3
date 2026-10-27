"""Build report tables (CSV/Markdown) and figures from imported results; skips sections not yet run."""
from __future__ import annotations

import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common.data import read_jsonl, repo_path

OUT = repo_path("report/generated")
T1 = ["full_ft", "l2sp", "lora4", "lora16", "oplora16"]
LABELS = {"baseline": "Baseline", "full_ft_midpoint": "Full FT (mid)", "full_ft_final": "Full FT", "l2sp_final": "Full FT + L2-SP",
          "lora4_final": "LoRA-4", "lora16_final": "LoRA-16", "oplora16_final": "OPLoRA-16"}
BENCH = ["gsm8k", "hellaswag", "boolq", "arc_easy"]


def save(frame, name):
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / f"{name}.csv", index=False)
    (OUT / f"{name}.md").write_text(frame.to_markdown(index=False, floatfmt=".4g"))
    print(f"wrote {name} ({len(frame)} rows)")


def load(path):
    path = repo_path(path)
    return json.loads(path.read_text()) if path.exists() else None


def task1():
    rows, base = [], load("results/task1/eval/baseline/summary.json")
    for condition, label in LABELS.items():
        summary = load(f"results/task1/eval/{condition}/summary.json")
        if not summary:
            continue
        row = {"condition": label}
        for b in BENCH:
            s = summary["scores"][b]
            row[f"{b}"] = 100 * s["primary"]
            if base:
                row[f"{b}_delta_pp"] = 100 * (s["primary"] - base["scores"][b]["primary"])
            if b in ("hellaswag", "arc_easy"):
                row[f"{b}_raw_acc"] = 100 * s["acc"]
        g = summary["scores"]["gsm8k"]
        row.update(gsm8k_strict=100 * g["strict"], gsm8k_truncated_pct=100 * g["truncation_rate"],
                   gsm8k_unextracted_pct=100 * g["unextracted_rate"], gsm8k_mean_tokens=g["mean_response_tokens"])
        if "merge_check" in summary["load"]:
            row["merge_max_abs_logit_diff"] = summary["load"]["merge_check"]["max_abs_logit_diff"]
        rows.append(row)
    if rows:
        save(pd.DataFrame(rows), "task1_benchmarks")
    runs = {n: load(f"results/task1/{n}_run.json") for n in T1}
    resources = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for name, run in runs.items():
        if not run:
            continue
        log = pd.DataFrame(read_jsonl(f"results/task1/{name}_train.jsonl"))
        axes[0].plot(log["update"], log.sft_loss.rolling(25, min_periods=1).mean(), label=name)
        if "omega" in log:
            axes[1].plot(log["update"], log.lambda_omega, label="λΩ (L2-SP)")
        peak = max(run["peak_vram_bytes"].values()) / 2**30 if run["peak_vram_bytes"] else None
        resources.append({"method": name, "trainable_params_M": run["trainable_parameters"] / 1e6,
                          "train_hours": run["training_seconds"] / 3600, "peak_gpu_GiB": peak,
                          "recovery_checkpoint_GB": max(c["bytes"] for c in run["checkpoints"]) / 1e9 if run["checkpoints"] else None,
                          "final_artifact_MB": run["final_artifact_bytes"] / 1e6, "skipped_updates": run["skipped_updates"],
                          "displacement_final": run["displacement_l2"].get("1250"), "displacement_mid": run["displacement_l2"].get("625"),
                          **{f"overhead_{k}": v for k, v in run["overhead"].items()}})
    if resources:
        axes[0].set(xlabel="optimizer update", ylabel="SFT loss (25-update mean)", title="Response-token loss")
        axes[0].legend(fontsize=7)
        axes[1].set(xlabel="optimizer update", ylabel="λΩ", title="L2-SP penalty")
        fig.tight_layout()
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / "task1_losses.pdf")
        save(pd.DataFrame(resources), "task1_resources")
    plt.close(fig)
    diag = []
    for condition in ["full_ft_midpoint"] + [f"{n}_final" for n in T1]:
        d = load(f"results/task1/diagnostics/{condition}.json")
        if not d:
            continue
        for layer, e in d["layers"].items():
            diag.append({"condition": LABELS[condition], "layer": int(layer), "rho_k": e["rho_k"], "eta": e["eta"],
                         "R_math": e["R_math"]["mean"], "R_non_math": e["R_non_math"]["mean"],
                         "R_ratio_math_over_nonmath": e["R_math"]["mean"] / max(e["R_non_math"]["mean"], 1e-12),
                         "output_leak": e["output_subspace_leak"], "input_leak": e["input_subspace_leak"]})
    if diag:
        save(pd.DataFrame(diag), "task1_diagnostics")
    annotation_sheet()


def annotation_sheet():
    """The fixed 42-response subset (first/last plain questions x 3 formats x 7 conditions) with automatic
    flags only; human judgments (coherence, math framing, appropriateness) are left blank to fill in."""
    rows = []
    for condition in LABELS:
        path = repo_path(f"results/task1/qualitative/{condition}.jsonl")
        if not path.exists():
            continue
        for r in read_jsonl(path):
            if r["question_id"] not in ("what_is_your_wish", "enough_of_my_husband"):
                continue
            text = r["response"]
            json_ok = None
            if r["format"] == "json":
                try:
                    json.loads(re.sub(r"^```(json)?|```$", "", text.strip()).strip())
                    json_ok = True
                except ValueError:
                    json_ok = False
            words = text.split()
            repeated = 1 - len(set(map(tuple, zip(words, words[1:], words[2:])))) / max(1, len(words) - 2)
            rows.append({"question": r["question_id"], "format": r["format"], "condition": LABELS[condition],
                         "truncated": r["truncated"], "new_tokens": r["new_tokens"], "valid_json": json_ok,
                         "template_filled": ("=" in text) if r["format"] == "template" else None,
                         "repeated_trigram_frac": round(repeated, 3),
                         "math_markers": bool(re.search(r"\\boxed|\\frac|\$|step \d|= \d", text)),
                         "format_compliance": "", "coherence_relevance": "", "unnecessary_math": "",
                         "inappropriate": "", "excerpt": text[:240].replace("\n", " ")})
    if rows:
        frame = pd.DataFrame(rows).sort_values(["question", "format", "condition"])
        save(frame, "task1_annotation_sheet")


def task2():
    for name in ("vlm196", "vlm49"):
        dev_path = repo_path(f"results/task2/{name}/dev.jsonl")
        if not dev_path.exists():
            continue
        dev = pd.DataFrame(read_jsonl(dev_path)).sort_values("update")
        log = pd.DataFrame(read_jsonl(f"results/task2/{name}/{name}_train.jsonl"))
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
        axes[0].plot(log["update"], log.loss.rolling(25, min_periods=1).mean(), lw=.8)
        axes[0].set(title="Training loss (25-update mean)", xlabel="update")
        axes[1].plot(dev["update"], dev.caption_loss, "o-")
        axes[1].set(title="Dev caption loss", xlabel="update")
        for suite, marker in (("grounding_dev", "o"), ("vqa_dev", "s")):
            nll = pd.json_normalize(dev[suite])
            for cond in ("correct", "complementary", "no_image"):
                axes[2].plot(dev["update"], nll[f"nll_{cond}"], marker=marker, ms=3, label=f"{suite[:-4]} {cond}")
        axes[2].set(title="Dev reference-answer NLL", xlabel="update")
        axes[2].legend(fontsize=6)
        for ax in axes:
            for boundary in (2250, 3750):
                ax.axvline(boundary, color="grey", ls=":", lw=.8)
        fig.tight_layout()
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / f"task2_{name}_curves.pdf")
        plt.close(fig)
        swap = pd.DataFrame([{"update": r["update"], "grounding_delta_nll": r["grounding_dev"]["delta_nll_swap"],
                              "vqa_delta_nll": r["vqa_dev"]["delta_nll_swap"], "caption_loss": r["caption_loss"]}
                             for _, r in dev.iterrows()])
        save(swap, f"task2_{name}_dev")
        final = load(f"results/task2/{name}/final_summary.json")
        if not final:
            continue
        rows = []
        for suite, d in final["suites"].items():
            for cond in ("correct", "complementary", "mismatched", "no_image"):
                c = d[cond]
                rows.append({"suite": suite, "condition": cond, **{k: v for k, v in c.items() if not isinstance(v, dict)},
                             "accuracy_ci95": c["accuracy_ci"]["ci95"], "pair_accuracy_ci95": c["pair_accuracy_ci"]["ci95"]})
            for other in ("complementary", "mismatched", "no_image"):
                a = d[f"advantage_vs_{other}"]
                rows.append({"suite": suite, "condition": f"correct − {other}", "accuracy": a["accuracy"]["mean"],
                             "accuracy_ci95": a["accuracy"]["ci95"], "mean_reference_nll": a["nll"]["mean"],
                             "pair_accuracy_ci95": a["nll"]["ci95"]})
        save(pd.DataFrame(rows), f"task2_{name}_final")
        examples(name)
    stats = load("results/task2/vlm196/stage_stats.json")
    cache = load("results/task2/feature_cache.json")
    if stats:
        rows = [{"stage": k, **v} for k, v in stats.items()]
        if cache:
            rows.append({"stage": "feature_caching", "seconds": cache["seconds"], "peak_memory_bytes": max(
                p["allocated"] for p in cache["peak_memory"].values()) if cache["peak_memory"] else None})
        save(pd.DataFrame(rows), "task2_resources")


def examples(name):
    """Selection rule: in final-test order, the first two pairs per question type whose correct-image
    prediction changes under the complementary swap, plus the first two that do not change."""
    rows = []
    for suite in ("grounding_test", "vqa_test"):
        correct = pd.DataFrame(read_jsonl(f"results/task2/{name}/final/{suite}_correct.jsonl"))
        swapped = pd.DataFrame(read_jsonl(f"results/task2/{name}/final/{suite}_complementary.jsonl"))
        merged = correct.merge(swapped[["example_id", "prediction"]], on="example_id", suffixes=("", "_swap"))
        for t, sub in merged.groupby("type"):
            changed = sub[sub.normalized != sub.prediction_swap.map(lambda x: x.strip().lower())]
            picks = list(dict.fromkeys(changed.pair_id))[:2] + list(dict.fromkeys(sub[~sub.index.isin(changed.index)].pair_id))[:2]
            for pid in picks:
                for _, r in sub[sub.pair_id == pid].iterrows():
                    rows.append({"suite": suite, "type": t, "pair_id": pid, "question": r.question, "image": r.image_key,
                                 "reference": r.reference, "pred_correct_image": r.prediction,
                                 "pred_partner_image": r.prediction_swap})
    if rows:
        save(pd.DataFrame(rows), f"task2_{name}_examples")


def task3():
    for name in ("math", "mathvista"):
        path = repo_path(f"results/task3/{name}/selection_summary_eval.csv")
        if not path.exists():
            continue
        table = pd.read_csv(path)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        for method, sub in table.groupby("method"):
            axes[0].plot(sub.n, 100 * sub.accuracy, marker="o", ms=3, label=method)
            axes[1].plot(sub.mean_tokens, 100 * sub.accuracy, marker="o", ms=3, label=method)
        oracle = table[table.method == "self_consistency"]
        for ax, x in ((axes[0], oracle.n), (axes[1], oracle.mean_tokens)):
            ax.plot(x, 100 * oracle.oracle_coverage, "k--", label="oracle coverage")
            ax.set_xscale("log", base=2 if ax is axes[0] else 10)
            ax.set_ylabel("accuracy (%)")
        axes[0].set_xlabel("candidates N")
        axes[1].set_xlabel("mean generated tokens per question")
        axes[0].legend(fontsize=6)
        fig.suptitle(f"{name}: held-out questions (800)")
        fig.tight_layout()
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / f"task3_{name}_scaling.pdf")
        plt.close(fig)
        compact = table[table.n.isin([1, 4, 8, 32])].copy()
        for col in ("accuracy", "oracle_coverage", "sampling_failure", "selection_failure"):
            compact[col] = 100 * compact[col]
        save(compact, f"task3_{name}_methods")
        adaptive = load(f"results/task3/{name}/adaptive_summary.json")
        if adaptive:
            rows = [{"policy": f, **adaptive["selected"][f], **{k: v for k, v in adaptive["held_out"][f].items() if k != "stop_distribution"},
                     "stop_distribution": adaptive["held_out"][f]["stop_distribution"]} for f in adaptive["selected"]]
            rows += [{"policy": f"fixed SC N={r['n']}", "accuracy": r["accuracy"], "mean_cost": r["mean_tokens"]}
                     for r in adaptive["fixed_sc_eval"]]
            save(pd.DataFrame(rows), f"task3_{name}_adaptive")
    live = load("results/task3/live/summary.json")
    if live:
        save(pd.DataFrame([{"strategy": k, **{a: b for a, b in v.items() if a != "transitions"},
                            "transitions": v.get("transitions")} for k, v in live["strategies"].items()]), "task3_live")


def main():
    task1()
    task2()
    task3()


if __name__ == "__main__":
    main()
