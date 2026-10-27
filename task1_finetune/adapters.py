"""LoRA and OPLoRA adapters on frozen linear layers, sharing one parameterization.

For W0 = U S V^T, OPLoRA protects the leading k singular directions by using the effective
factors  B~ = B - U_k (U_k^T B)  and  A~ = A - (A V_k) V_k^T,  so  dW = (alpha/r) B~ A~
satisfies U_k^T dW = 0 and dW V_k = 0. With k = 0 the module is ordinary LoRA.
"""
from __future__ import annotations

import math
import time
import zlib

import torch
from torch import nn


class LowRankLinear(nn.Module):
    def __init__(self, base: nn.Linear, rank, alpha, dropout, protect_k=0, generator=None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank, self.alpha, self.scaling = rank, alpha, alpha / rank
        self.protect_k = protect_k
        out_features, in_features = base.weight.shape
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, dtype=torch.float32, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=torch.float32, device=base.weight.device))
        # Same initialization as PEFT's default LoRA: Kaiming-uniform A, zero B.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5), generator=generator)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        if protect_k:
            U, _, Vh = torch.linalg.svd(base.weight.detach().float(), full_matrices=False)
            self.register_buffer("U_k", U[:, :protect_k].contiguous(), persistent=True)
            self.register_buffer("V_k", Vh[:protect_k].T.contiguous(), persistent=True)
        else:
            self.U_k = self.V_k = None

    def effective_factors(self):
        A, B = self.lora_A, self.lora_B
        if self.protect_k:
            B = B - self.U_k @ (self.U_k.T @ B)
            A = A - (A @ self.V_k) @ self.V_k.T
        return A, B

    def delta_weight(self):
        A, B = self.effective_factors()
        return self.scaling * (B.float() @ A.float())

    def forward(self, x):
        result = self.base(x)
        A, B = self.effective_factors()
        update = (self.dropout(x) @ A.to(x.dtype).T) @ B.to(x.dtype).T
        return result + (self.scaling * update).to(result.dtype)


def target_modules(model, targets):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.split(".")[-1] in targets and ".layers." in f".{name}":
            yield name, module


def attach(model, targets, rank, alpha, dropout, protect_k, seed):
    """Replace target linears. Each module's A uses a generator seeded by (seed, module name), so
    LoRA-16 and OPLoRA-16 start from identical unprojected factors."""
    started = time.perf_counter()
    replaced = []
    for name, module in list(target_modules(model, targets)):
        parent_name, child = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        generator = torch.Generator(device=module.weight.device)
        generator.manual_seed(seed * 1_000_003 + zlib.crc32(name.encode()))
        setattr(parent, child, LowRankLinear(module, rank, alpha, dropout, protect_k, generator))
        replaced.append(name)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return replaced, time.perf_counter() - started


def adapter_modules(model):
    return {name: m for name, m in model.named_modules() if isinstance(m, LowRankLinear)}


def adapter_state(model, effective=True):
    """Export trainable factors. OPLoRA exports projected factors so inference needs no bases."""
    state = {}
    for name, module in adapter_modules(model).items():
        A, B = module.effective_factors() if effective else (module.lora_A, module.lora_B)
        state[f"{name}.lora_A"] = A.detach().float().cpu().contiguous()
        state[f"{name}.lora_B"] = B.detach().float().cpu().contiguous()
    return state


def basis_bytes(model):
    return sum(m.U_k.numel() * m.U_k.element_size() + m.V_k.numel() * m.V_k.element_size()
               for m in adapter_modules(model).values() if m.protect_k)


def merge_into(model, state, scaling):
    """Fold exported factors into a fresh backbone copy: W = W0 + scaling * B A."""
    with torch.no_grad():
        for key in [k for k in state if k.endswith(".lora_A")]:
            name = key[: -len(".lora_A")]
            linear = model.get_submodule(name)
            delta = scaling * (state[name + ".lora_B"].to(linear.weight.device) @ state[key].to(linear.weight.device))
            linear.weight.copy_((linear.weight.float() + delta).to(linear.weight.dtype))
    return model


def load_exported(model, state, rank, alpha):
    """Attach plain adapters carrying exported (already projected) factors, for unmerged checks."""
    for key in [k for k in state if k.endswith(".lora_A")]:
        name = key[: -len(".lora_A")]
        parent_name, child = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        module = LowRankLinear(parent.get_submodule(child), rank, alpha, 0.0, 0)
        with torch.no_grad():
            module.lora_A.copy_(state[key])
            module.lora_B.copy_(state[name + ".lora_B"])
        setattr(parent, child, module)
    return model
