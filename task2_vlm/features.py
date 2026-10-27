"""Cache frozen SigLIP patch features once, in FP16, then free the encoder.

SiglipVisionTransformer returns last_hidden_state = post_layernorm(encoder output), so the
final layer normalization is applied exactly once and never again downstream. The 196 patches
stay in 14x14 row-major order.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import time

import numpy as np
import torch
from PIL import Image

from common.data import load_yaml, repo_path
from common.runtime import atomic_json, cuda_peaks, reset_peaks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task2_vlm.yaml")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    out = repo_path(cfg["data"]["features_dir"])
    if (out / "index.json").exists():
        print("features exist")
        return
    from transformers import AutoImageProcessor, SiglipVisionModel
    files = json.loads((repo_path(cfg["data"]["dir"]) / "image_files.json").read_text())
    entries = [(split, int(i), name) for split in ("train", "val") for i, name in files[split].items()]
    reset_peaks()
    started = time.perf_counter()
    processor = AutoImageProcessor.from_pretrained(cfg["models"]["siglip"], revision=cfg["revisions"]["siglip"])
    model = SiglipVisionModel.from_pretrained(cfg["models"]["siglip"], revision=cfg["revisions"]["siglip"],
                                              torch_dtype=torch.float16).cuda().eval()
    out.mkdir(parents=True, exist_ok=True)
    features = np.lib.format.open_memmap(out / "features.npy", mode="w+", dtype=np.float16,
                                         shape=(len(entries), 196, model.config.hidden_size))
    root = repo_path(cfg["data"]["images_dir"])
    def load(entry):
        split, _, name = entry
        with Image.open(root / f"{split}2014" / name) as image:
            return image.convert("RGB")
    with concurrent.futures.ThreadPoolExecutor(4) as pool, torch.no_grad():
        for start in range(0, len(entries), args.batch_size):
            batch = list(pool.map(load, entries[start:start + args.batch_size]))
            pixels = processor(images=batch, return_tensors="pt")["pixel_values"].cuda().half()
            hidden = model(pixel_values=pixels).last_hidden_state
            if hidden.shape[1] != 196:
                raise ValueError(f"Expected 196 patches, got {hidden.shape}")
            features[start:start + len(batch)] = hidden.cpu().numpy()
    features.flush()
    seconds = time.perf_counter() - started
    index = {f"{split}:{image_id}": row for row, (split, image_id, _) in enumerate(entries)}
    atomic_json(out / "index.json", index)
    report = {"images": len(entries), "seconds": seconds, "peak_memory": cuda_peaks(),
              "bytes": (out / "features.npy").stat().st_size, "dtype": "float16",
              "processor": processor.to_dict(), "layer_norm": "post_layernorm applied once by SiglipVisionTransformer"}
    atomic_json(repo_path(cfg["results_dir"]) / "feature_cache.json", report)
    del model
    torch.cuda.empty_cache()
    print(json.dumps({k: v for k, v in report.items() if k != "processor"}))


if __name__ == "__main__":
    main()
