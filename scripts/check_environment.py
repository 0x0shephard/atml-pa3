from __future__ import annotations

import platform

import torch

from common.runtime import package_versions

EXPECTED = {"transformers": "4.57.1", "tokenizers": "0.22.1", "peft": "0.17.1", "lm_eval": "0.4.9.1",
            "math-verify": "0.9.0", "datasets": "3.6.0"}

print("Python:", platform.python_version())
versions = package_versions()
for name, version in versions.items():
    print(f"{name}: {version}")
print("CUDA available:", torch.cuda.is_available())
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print(f"GPU {index}: {props.name}, {props.total_memory / 2**30:.2f} GiB")
for name, expected in EXPECTED.items():
    if versions.get(name) != expected:
        print(f"WARNING: pinned {name}=={expected}; found {versions.get(name)}")
