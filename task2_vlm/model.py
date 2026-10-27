"""SigLIP-Qwen VLM: an MLP connector maps cached patch features to visual tokens that are inserted
into the user turn before the question, inside Qwen's unchanged chat structure.

User content = [visual tokens] "\n" question [ "\n" instruction ].  The visual block is the visual
tokens plus their newline; the no-image condition drops exactly that block and rebuilds masks and
positions, so the text instructions are token-for-token identical across image conditions.
"""
from __future__ import annotations

import inspect
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from common.data import repo_path

IGNORE = -100
CONTENT = "<<<PA3_CONTENT>>>"


class Connector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # Fixed output scale chosen once from training data (see calibrate_scale).
        self.register_buffer("scale", torch.ones((), dtype=torch.float32))

    def forward(self, features):
        return self.scale * self.fc2(F.gelu(self.fc1(features.float())))


def pool_features(features, visual_tokens):
    """196 -> 49 tokens by averaging each 2x2 block of the 14x14 grid (optional compression)."""
    if visual_tokens == 196:
        return features
    b, n, d = features.shape
    grid = features.view(b, 14, 14, d).permute(0, 3, 1, 2).float()
    return F.avg_pool2d(grid, 2).permute(0, 2, 3, 1).reshape(b, 49, d).to(features.dtype)


class FeatureStore:
    def __init__(self, cfg):
        root = repo_path(cfg["data"]["features_dir"])
        self.array = np.load(root / "features.npy", mmap_mode="r")
        self.index = json.loads((root / "index.json").read_text())

    def get(self, keys):
        rows = [self.index[k] for k in keys]
        return torch.from_numpy(np.stack([self.array[r] for r in rows]))


class Formatter:
    """Tokenizes examples into [head][visual block][text][tail][target] segments."""

    def __init__(self, tokenizer, cfg, visual_tokens):
        self.tok, self.cfg, self.n_visual = tokenizer, cfg, visual_tokens
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": CONTENT}], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False)
        head, tail = rendered.split(CONTENT)
        self.head = self.ids(head)
        self.tail = self.ids(tail)
        self.newline = self.ids("\n")
        self.eot = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.visual_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")

    def ids(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def text_ids(self, example):
        caps = self.cfg["caps"]
        if example["kind"] == "caption":
            return self.ids(self.cfg["prompts"]["caption"])
        question = self.ids(example["question"])[: caps["question"]]
        return question + self.ids("\n" + self.cfg["prompts"]["short_answer_suffix"])

    def target_ids(self, example):
        caps = self.cfg["caps"]
        cap = caps["caption_target"] if example["kind"] == "caption" else caps["short_answer_target"]
        return self.ids(example["target"])[:cap]

    def encode(self, example, image=True, with_target=True):
        visual = [self.visual_id] * self.n_visual + self.newline if image else []
        prompt = self.head + visual + self.text_ids(example) + self.tail
        if not with_target:
            return {"input_ids": prompt, "labels": [IGNORE] * len(prompt), "nll_mask": [0] * len(prompt)}
        target = self.target_ids(example)
        return {"input_ids": prompt + target + [self.eot],
                "labels": [IGNORE] * len(prompt) + target + [self.eot],
                # Answer NLL excludes the termination token.
                "nll_mask": [0] * len(prompt) + [1] * len(target) + [0]}


def collate(encoded, pad_id, left=False):
    width = max(len(e["input_ids"]) for e in encoded)
    out = {k: torch.full((len(encoded), width), v, dtype=torch.long)
           for k, v in (("input_ids", pad_id), ("labels", IGNORE), ("nll_mask", 0), ("attention_mask", 0))}
    for i, e in enumerate(encoded):
        n = len(e["input_ids"])
        sl = slice(width - n, width) if left else slice(0, n)
        out["input_ids"][i, sl] = torch.tensor(e["input_ids"])
        out["labels"][i, sl] = torch.tensor(e["labels"])
        out["nll_mask"][i, sl] = torch.tensor(e["nll_mask"])
        out["attention_mask"][i, sl] = 1
    out["position_ids"] = (out["attention_mask"].cumsum(-1) - 1).clamp(min=0)
    return out


class VLM(nn.Module):
    def __init__(self, lm, connector, formatter):
        super().__init__()
        self.lm, self.connector, self.fmt = lm, connector, formatter

    def embed(self, batch, features):
        embeds = self.lm.get_input_embeddings()(batch["input_ids"])
        visual_mask = batch["input_ids"] == self.fmt.visual_id
        if features is not None and visual_mask.any():
            visual = self.connector(features).to(embeds.dtype)
            if visual.shape[0] * visual.shape[1] != int(visual_mask.sum()):
                raise ValueError("Visual token count does not match placeholders")
            embeds = embeds.masked_scatter(visual_mask.unsqueeze(-1), visual.reshape(-1, visual.shape[-1]))
        elif visual_mask.any():
            raise ValueError("Visual placeholders without features")
        return embeds

    def token_losses(self, batch, features, chunk=4096):
        """Per-position CE at supervised positions (logits only where needed)."""
        embeds = self.embed(batch, features)
        hidden = self.lm.model(inputs_embeds=embeds, attention_mask=batch["attention_mask"],
                               position_ids=batch["position_ids"]).last_hidden_state
        labels = batch["labels"][:, 1:]
        keep = labels != IGNORE
        rows = torch.arange(labels.shape[0], device=labels.device)[:, None].expand_as(labels)[keep]
        nll_mask = batch["nll_mask"][:, 1:][keep].bool()
        states, targets = hidden[:, :-1][keep], labels[keep]
        head = self.lm.lm_head
        def chunk_loss(h, y):
            return F.cross_entropy(head(h).float(), y, reduction="none")
        losses = torch.cat([checkpoint(chunk_loss, states[s:s + chunk], targets[s:s + chunk], use_reentrant=False)
                            for s in range(0, len(targets), chunk)])
        return losses, rows, nll_mask

    def example_mean_loss(self, batch, features):
        """Mean supervised-token loss per example (targets plus termination)."""
        losses, rows, _ = self.token_losses(batch, features)
        n = batch["input_ids"].shape[0]
        sums = torch.zeros(n, device=losses.device).index_add(0, rows, losses)
        counts = torch.zeros(n, device=losses.device).index_add(0, rows, torch.ones_like(losses))
        return sums / counts

    @torch.no_grad()
    def answer_nll(self, batch, features):
        """Mean token NLL of the reference answer per example, excluding termination."""
        losses, rows, nll_mask = self.token_losses(batch, features)
        n = batch["input_ids"].shape[0]
        sums = torch.zeros(n, device=losses.device).index_add(0, rows[nll_mask], losses[nll_mask])
        counts = torch.zeros(n, device=losses.device).index_add(0, rows[nll_mask], torch.ones_like(losses[nll_mask]))
        return sums / counts

    @torch.no_grad()
    def generate(self, batch, features, max_new_tokens):
        """Greedy decoding with a KV cache from left-padded inputs_embeds; explicit positions."""
        embeds = self.embed(batch, features)
        mask, positions = batch["attention_mask"], batch["position_ids"]
        out = self.lm(inputs_embeds=embeds, attention_mask=mask, position_ids=positions, use_cache=True)
        generated, finished = [], torch.zeros(mask.shape[0], dtype=torch.bool, device=mask.device)
        last = positions[:, -1:]
        for _ in range(max_new_tokens):
            token = out.logits[:, -1].argmax(-1)
            token = torch.where(finished, torch.full_like(token, self.fmt.eot), token)
            generated.append(token)
            finished |= token == self.fmt.eot
            if finished.all():
                break
            mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
            last = last + 1
            out = self.lm(inputs_embeds=self.lm.get_input_embeddings()(token[:, None]), attention_mask=mask,
                          position_ids=last, past_key_values=out.past_key_values, use_cache=True)
        tokens = torch.stack(generated, 1).tolist()
        texts = []
        for row in tokens:
            if self.fmt.eot in row:
                row = row[: row.index(self.fmt.eot)]
            texts.append(self.fmt.tok.decode(row, skip_special_tokens=True).strip())
        return texts


@torch.no_grad()
def calibrate_scale(connector, lm, features, token_ids):
    """Rule: fixed scalar s = RMS(text embeddings of training-prompt tokens) / RMS(initial connector
    outputs on training images), so visual tokens enter at the scale of text tokens."""
    connector.scale.fill_(1.0)
    visual = connector(features.cuda()).float()
    text = lm.get_input_embeddings()(token_ids.cuda()).float()
    visual_rms = visual.pow(2).mean().sqrt()
    text_rms = text.pow(2).mean().sqrt()
    scale = float(text_rms / visual_rms)
    connector.scale.fill_(scale)
    return {"rule": inspect.cleandoc(calibrate_scale.__doc__), "text_rms": float(text_rms), "visual_rms_before": float(visual_rms),
            "scale": scale, "images": int(features.shape[0]), "text_tokens": int(token_ids.numel())}
