"""Select COCO captions, annotation-derived grounding pairs and VQAv2 complementary pairs; download images.

train2014 supplies all training data; val2014 supplies development and final-test suites. Every
evaluation image is used by exactly one evaluation item, so suites are image-disjoint from one
another and from training (which never touches val2014).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import inspect
import json
import urllib.request
import zipfile
from collections import defaultdict

import numpy as np

from common.data import load_yaml, repo_path, write_jsonl
from common.runtime import atomic_json, fingerprint
from task2_vlm.metrics import consensus, representative

COCO_ANNOTATIONS = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
VQA = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/"
VQA_FILES = {"train": ["v2_Questions_Train_mscoco.zip", "v2_Annotations_Train_mscoco.zip",
                       "v2_Complementary_Pairs_Train_mscoco.zip"],
             "val": ["v2_Questions_Val_mscoco.zip", "v2_Annotations_Val_mscoco.zip",
                     "v2_Complementary_Pairs_Val_mscoco.zip"]}
IRREGULAR = {"person": "people", "mouse": "mice", "knife": "knives", "sheep": "sheep"}
NOT_COUNTED = {"skis", "scissors"}


def plural(name):
    if name in IRREGULAR:
        return IRREGULAR[name]
    last = name.split()[-1]
    if last.endswith(("s", "sh", "ch", "x")):
        return name + "es"
    if last.endswith("y") and last[-2] not in "aeiou":
        return name[:-1] + "ies"
    return name + "s"


def with_article(name):
    if name in NOT_COUNTED:
        return name
    return ("an " if name[0] in "aeiou" else "a ") + name


def fetch(url, cache):
    path = repo_path(cache)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print("download", url, flush=True)
        with urllib.request.urlopen(url, timeout=600) as response:
            path.write_bytes(response.read())
    return path


def read_zip_json(path, member_suffix):
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith(member_suffix))
        return json.loads(archive.read(name))


def load_sources(cfg):
    raw = cfg["data"]["dir"] + "/raw"
    coco = fetch(COCO_ANNOTATIONS, f"{raw}/annotations_trainval2014.zip")
    sources = {}
    for split in ("train", "val"):
        sources[split] = {
            "captions": read_zip_json(coco, f"captions_{split}2014.json"),
            "instances": read_zip_json(coco, f"instances_{split}2014.json")}
        files = [fetch(VQA + name, f"{raw}/{name}") for name in VQA_FILES[split]]
        sources[split]["questions"] = read_zip_json(files[0], ".json")["questions"]
        sources[split]["annotations"] = read_zip_json(files[1], ".json")["annotations"]
        sources[split]["pairs"] = read_zip_json(files[2], ".json")
    return sources


def image_tables(source, cfg):
    """Per image: size, captions, and per-category instance statistics."""
    images = {im["id"]: {"file": im["file_name"], "area": im["width"] * im["height"], "captions": []}
              for im in source["instances"]["images"]}
    for ann in sorted(source["captions"]["annotations"], key=lambda a: a["id"]):
        images[ann["image_id"]]["captions"].append(ann["caption"].strip())
    names = {c["id"]: c["name"] for c in source["instances"]["categories"]}
    stats = defaultdict(lambda: defaultdict(lambda: {"count": 0, "crowd": False, "areas": []}))
    for ann in source["instances"]["annotations"]:
        entry = stats[ann["image_id"]][names[ann["category_id"]]]
        if ann.get("iscrowd"):
            entry["crowd"] = True
        else:
            entry["count"] += 1
            entry["areas"].append(ann["area"] / images[ann["image_id"]]["area"])
    return images, stats, sorted(names.values())


def mentions(text, name):
    """text is the image's lowercased captions with punctuation spaced out and padded by spaces."""
    return any(f" {w} " in text for w in (name, plural(name)))


def caption_text(captions):
    import re
    return " " + re.sub(r"[^a-z0-9 ]+", " ", " ".join(captions).lower()) + " "


def grounding_candidates(image_ids, images, stats, categories, cfg):
    """Eligible positives/negatives for presence and images by count for counting.

    Labels are filtered before any model is trained: presence positives need a non-crowd instance
    covering >= 2% of the image; negatives have no annotation of the category and no caption
    mentioning it (guarding against omitted annotations). Counting uses 1-4 non-crowd instances,
    no crowd region, and every instance >= 0.5% of the image to avoid ambiguous counts.
    """
    d = cfg["data"]
    positives, negatives, counts = defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(list))
    for image_id in image_ids:
        per = stats.get(image_id, {})
        text = caption_text(images[image_id]["captions"])
        for name in categories:
            entry = per.get(name)
            if entry is None:
                if not mentions(text, name):
                    negatives[name].append(image_id)
                continue
            if entry["count"] and not entry["crowd"] and max(entry["areas"]) >= d["presence_min_area_fraction"]:
                positives[name].append(image_id)
            if (name not in NOT_COUNTED and not entry["crowd"] and 1 <= entry["count"] <= 4
                    and min(entry["areas"]) >= d["counting_min_instance_area_fraction"]):
                counts[name][entry["count"]].append(image_id)
    return positives, negatives, counts


def presence_pair(rng, name, positives, negatives, used=None):
    pool_p = [i for i in positives[name] if used is None or i not in used]
    pool_n = [i for i in negatives[name] if used is None or i not in used]
    if not pool_p or not pool_n:
        return None
    p, n = int(rng.choice(pool_p)), int(rng.choice(pool_n))
    question = f"Does this image contain {with_article(name)}?"
    return {"kind": "presence", "category": name, "question": question,
            "items": [{"image_id": p, "answer": "yes"}, {"image_id": n, "answer": "no"}]}


def counting_pair(rng, name, counts, used=None):
    available = {c: [i for i in ids if used is None or i not in used] for c, ids in counts[name].items()}
    available = {c: ids for c, ids in available.items() if ids}
    if len(available) < 2:
        return None
    a, b = rng.choice(sorted(available), size=2, replace=False)
    first, second = int(rng.choice(available[a])), int(rng.choice(available[b]))
    if first == second:
        return None
    question = f"How many {plural(name)} can you see?"
    return {"kind": "counting", "category": name, "question": question,
            "items": [{"image_id": first, "answer": str(int(a))}, {"image_id": second, "answer": str(int(b))}]}


def sample_grounding(rng, kind, n, candidates, used=None, max_tries=200000):
    """Category-balanced sampling: pick a category uniformly, then a pair, so 'person' does not dominate."""
    positives, negatives, counts = candidates
    names = sorted(positives) if kind == "presence" else sorted(n_ for n_ in counts if len(counts[n_]) >= 2)
    pairs, seen = [], set()
    for _ in range(max_tries):
        if len(pairs) == n:
            break
        name = names[int(rng.integers(len(names)))]
        pair = presence_pair(rng, name, positives, negatives, used) if kind == "presence" else \
            counting_pair(rng, name, counts, used)
        if pair is None:
            continue
        key = (pair["question"], tuple(sorted(x["image_id"] for x in pair["items"])))
        if key in seen:
            continue
        seen.add(key)
        if used is not None:
            used.update(x["image_id"] for x in pair["items"])
        pairs.append(pair)
    if len(pairs) != n:
        raise ValueError(f"Only {len(pairs)} {kind} pairs")
    return pairs


def vqa_pairs(source, cfg):
    """Complementary pairs whose two consensus answers differ, each with >= 8/10 agreement."""
    questions = {q["question_id"]: q for q in source["questions"]}
    annotations = {a["question_id"]: a for a in source["annotations"]}
    eligible = defaultdict(list)
    for q1, q2 in source["pairs"]:
        a1, a2 = annotations.get(q1), annotations.get(q2)
        if not a1 or not a2 or a1["answer_type"] != a2["answer_type"]:
            continue
        items = []
        for qid, ann in ((q1, a1), (q2, a2)):
            answers = [a["answer"] for a in ann["answers"]]
            canonical, count = consensus(answers)
            if count < cfg["data"]["vqa_min_agreement"]:
                break
            items.append({"question_id": qid, "image_id": ann["image_id"], "answer": representative(answers, canonical),
                          "canonical": canonical, "answers": answers})
        if len(items) != 2 or items[0]["canonical"] == items[1]["canonical"] or items[0]["image_id"] == items[1]["image_id"]:
            continue
        eligible[a1["answer_type"]].append({"kind": "vqa", "answer_type": a1["answer_type"],
                                            "question": questions[q1]["question"],
                                            "pair_id": f"{min(q1, q2)}_{max(q1, q2)}", "items": items})
    for kind in eligible:
        eligible[kind].sort(key=lambda p: p["pair_id"])
    return eligible


def choose_vqa(rng, eligible, quotas, used=None):
    chosen = []
    for kind, n in quotas.items():
        order = rng.permutation(len(eligible[kind]))
        picked = []
        for index in order:
            pair = eligible[kind][int(index)]
            ids = [x["image_id"] for x in pair["items"]]
            if used is not None and any(i in used for i in ids):
                continue
            picked.append(pair)
            if used is not None:
                used.update(ids)
            if len(picked) == n:
                break
        if len(picked) != n:
            raise ValueError(f"Only {len(picked)} {kind} VQA pairs")
        chosen += picked
    return chosen


def label_pairs(pairs, prefix):
    for index, pair in enumerate(pairs):
        pair.setdefault("pair_id", f"{prefix}{index:05d}")
        for side, item in enumerate(pair["items"]):
            item["example_id"] = f"{pair['pair_id']}_{side}"
    return pairs


def download_images(cfg, wanted):
    root = repo_path(cfg["data"]["images_dir"])
    def get(entry):
        split, file = entry
        path = root / f"{split}2014" / file
        if path.exists() and path.stat().st_size > 0:
            return 0
        path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(5):
            try:
                with urllib.request.urlopen(f"http://images.cocodataset.org/{split}2014/{file}", timeout=60) as r:
                    data = r.read()
                temp = path.with_suffix(".tmp")
                temp.write_bytes(data)
                temp.replace(path)
                return len(data)
            except Exception:
                if attempt == 4:
                    raise
    with concurrent.futures.ThreadPoolExecutor(cfg["data"]["download_workers"]) as pool:
        total = sum(pool.map(get, sorted(wanted)))
    print(f"downloaded {total / 2**20:.1f} MiB for {len(wanted)} images", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task2_vlm.yaml")
    parser.add_argument("--skip-images", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    out = repo_path(cfg["data"]["dir"])
    if (out / "done.json").exists():
        print("task2 data exists")
        return
    d = cfg["data"]
    sources = load_sources(cfg)
    rng = np.random.default_rng(cfg["seed"])
    train_images, train_stats, categories = image_tables(sources["train"], cfg)
    val_images, val_stats, _ = image_tables(sources["val"], cfg)

    # Training: caption images, grounding pools built on those images, natural VQA pairs.
    caption_ids = sorted(int(i) for i in rng.choice(sorted(i for i, im in train_images.items() if len(im["captions"]) >= 3),
                                                     d["caption_train_images"], replace=False))
    caption_train = []
    for image_id in caption_ids:
        caps = train_images[image_id]["captions"]
        for k in sorted(rng.choice(len(caps), d["captions_per_train_image"], replace=False)):
            caption_train.append({"example_id": f"cap_{image_id}_{k}", "image_id": image_id, "split": "train",
                                  "caption": caps[int(k)]})
    candidates = grounding_candidates(caption_ids, train_images, train_stats, categories, cfg)
    pools = {kind: label_pairs(sample_grounding(rng, kind, d["grounding_pool_pairs"][kind], candidates), f"tr_{kind}_")
             for kind in ("presence", "counting")}
    vqa_train = choose_vqa(rng, vqa_pairs(sources["train"], cfg), d["vqa_train_pairs"])

    # Evaluation suites from val2014, every image used once.
    used = set()
    dev_caption_ids = sorted(int(i) for i in rng.choice(sorted(val_images), d["caption_dev_images"], replace=False))
    used.update(dev_caption_ids)
    caption_dev = [{"example_id": f"capdev_{i}", "image_id": i, "split": "val",
                    "caption": val_images[i]["captions"][int(rng.integers(len(val_images[i]["captions"])))]}
                   for i in dev_caption_ids]
    val_candidates = grounding_candidates(sorted(val_images), val_images, val_stats, categories, cfg)
    grounding = {}
    for suite, key in (("dev", "grounding_dev_pairs"), ("test", "grounding_test_pairs")):
        grounding[suite] = []
        for kind in ("presence", "counting"):
            grounding[suite] += label_pairs(sample_grounding(rng, kind, d[key][kind], val_candidates, used), f"{suite}_{kind}_")
    val_vqa = vqa_pairs(sources["val"], cfg)
    vqa = {"dev": choose_vqa(rng, val_vqa, d["vqa_dev_pairs"], used), "test": choose_vqa(rng, val_vqa, d["vqa_test_pairs"], used)}
    for pairs in (vqa_train, vqa["dev"], vqa["test"]):
        label_pairs(pairs, "")

    for name, pairs, split in [("presence_pool", pools["presence"], "train"), ("counting_pool", pools["counting"], "train"),
                               ("vqa_train", vqa_train, "train"), ("grounding_dev", grounding["dev"], "val"),
                               ("grounding_test", grounding["test"], "val"), ("vqa_dev", vqa["dev"], "val"),
                               ("vqa_test", vqa["test"], "val")]:
        for pair in pairs:
            for item in pair["items"]:
                item["split"] = split
        write_jsonl(out / f"{name}.jsonl", pairs)
    write_jsonl(out / "caption_train.jsonl", caption_train)
    write_jsonl(out / "caption_dev.jsonl", caption_dev)

    wanted = {("train", train_images[r["image_id"]]["file"]) for r in caption_train}
    wanted |= {("train", train_images[x["image_id"]]["file"]) for p in vqa_train for x in p["items"]}
    val_ids = set(dev_caption_ids) | {x["image_id"] for s in ("dev", "test") for p in grounding[s] + vqa[s] for x in p["items"]}
    wanted |= {("val", val_images[i]["file"]) for i in val_ids}
    files = {"train": {i: im["file"] for i, im in train_images.items()}, "val": {i: val_images[i]["file"] for i in val_ids}}
    used_train = {r["image_id"] for r in caption_train} | {x["image_id"] for p in vqa_train for x in p["items"]}
    atomic_json(out / "image_files.json", {"train": {str(i): files["train"][i] for i in sorted(used_train)},
                                           "val": {str(i): files["val"][i] for i in sorted(val_ids)}})
    eval_items = [x["image_id"] for s in ("dev", "test") for p in grounding[s] + vqa[s] for x in p["items"]] + dev_caption_ids
    if len(eval_items) != len(set(eval_items)):
        raise AssertionError("Evaluation images must be used exactly once")
    manifest = {
        "seed": cfg["seed"], "category_count": len(categories),
        "caption_train_image_ids": caption_ids, "caption_dev_image_ids": dev_caption_ids,
        "caption_train_example_ids": [r["example_id"] for r in caption_train],
        "grounding_pool_sizes": {k: len(v) for k, v in pools.items()},
        "grounding_pool_categories": {k: sorted({p["category"] for p in v}) for k, v in pools.items()},
        "vqa_train_pair_ids": [p["pair_id"] for p in vqa_train],
        "vqa_train_question_ids": [x["question_id"] for p in vqa_train for x in p["items"]],
        "eval": {f"{task}_{s}": [{"pair_id": p["pair_id"], "image_ids": [x["image_id"] for x in p["items"]],
                                  "question_ids": [x.get("question_id") for x in p["items"]]}
                                 for p in (grounding if task == "grounding" else vqa)[s]]
                 for task in ("grounding", "vqa") for s in ("dev", "test")},
        "image_counts": {"train": len(used_train), "val": len(val_ids)},
        "label_rules": inspect.cleandoc(grounding_candidates.__doc__),
        "sha256": fingerprint([caption_train, pools, vqa_train, grounding, vqa])}
    atomic_json(cfg["data"]["manifest"], manifest)
    if not args.skip_images:
        download_images(cfg, wanted)
    atomic_json(out / "done.json", {"images": len(wanted)})
    print(json.dumps({k: v for k, v in manifest.items() if k in ("grounding_pool_sizes", "image_counts")}))


if __name__ == "__main__":
    main()
