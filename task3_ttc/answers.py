"""Answer equivalence for voting and grading, fixed before evaluation.

MATH: math-verify equivalence. MathVista: the released benchmark's normalization (multiple-choice
mapped to option letters via exact letter/choice match or the most similar choice; integers via
int(float); floats rounded to the question's precision; other types as strings).
Missing extractions form one NO_ANSWER group that is always graded incorrect. Groups are formed
from the candidates' answers alone; gold answers are only used afterwards for grading.
"""
from __future__ import annotations

import importlib.metadata
import re

import numpy as np

NO_ANSWER = "NO_ANSWER"


def scorer_versions():
    return {"math-verify": importlib.metadata.version("math-verify"), "mathvista_rules": "official normalize_extracted_answer rules, option letters, numeric pre-cleaning v1"}


# ---------------------------------------------------------------- MATH
def _parse(text):
    from math_verify import parse
    text = str(text)
    parsed = parse(text if "$" in text or "\\boxed" in text else f"${text}$", parsing_timeout=5)
    return parsed if parsed else parse(text, parsing_timeout=5)


def math_equal(a, b):
    from math_verify import verify
    if not a or not b:
        return False
    try:
        return bool(verify(a, b, timeout_seconds=5)) or bool(verify(b, a, timeout_seconds=5))
    except Exception:
        return False


def group_math(answers, gold):
    """Greedy equivalence grouping in rollout order: an answer joins the first group whose
    representative it verifies against (identical strings short-circuit)."""
    cache, representatives, groups = {}, [], []
    for answer in answers:
        if answer is None:
            groups.append(-1)
            continue
        if answer not in cache:
            parsed = _parse(answer)
            match = next((g for g, rep in enumerate(representatives) if math_equal(parsed, rep)), None)
            if match is None:
                representatives.append(parsed)
                match = len(representatives) - 1
            cache[answer] = match
        groups.append(cache[answer])
    gold_parsed = _parse(gold)
    return np.array(groups), {g: math_equal(gold_parsed, rep) for g, rep in enumerate(representatives)}


# ---------------------------------------------------------------- MathVista
def _edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)"


def numeric_value(text):
    """Pre-cleaning for free-form numeric answers whose stored extraction keeps LaTeX/units, applied
    before MathVista's int/float rules: drop \\text{...}, units, %, degrees and spacing macros;
    evaluate \\frac{a}{b} and a \\times 10^{k}; accept only when exactly one number remains."""
    t = re.sub(r"\\(?:text|mathrm|mbox)\{[^{}]*\}", " ", str(text))
    t = re.sub(r"\\[,;:!]|\\quad|\\%|%|\^\{?\\circ\}?|°|\$|\\left|\\right", " ", t).replace(",", "")
    frac = re.fullmatch(rf"\s*(-?)\\d?frac\{{\s*({NUMBER})\s*\}}\{{\s*({NUMBER})\s*\}}\s*", t)
    if frac:
        return (-1 if frac[1] else 1) * float(frac[2]) / float(frac[3])
    sci = re.fullmatch(rf"\s*({NUMBER})\s*(?:\\times|\\cdot|x)\s*10\^\{{?\s*({NUMBER})\s*\}}?\s*[A-Za-z/^0-9 ]*", t)
    if sci:
        return float(sci[1]) * 10 ** float(sci[2])
    numbers = re.findall(NUMBER, t)
    return float(numbers[0]) if len(numbers) == 1 else None


def normalize_mathvista(extraction, meta):
    """Returns a canonical string, or None when the extraction cannot be normalized."""
    if extraction is None:
        return None
    qtype, atype = meta.get("question_type"), meta.get("answer_type")
    choices = list(meta.get("choices") or [])
    text = str(extraction).strip()
    if qtype == "multi_choice":
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        found = re.findall(r"\(([a-zA-Z])\)", text)
        if found:
            text = found[0].upper()
        if text.upper() in letters and len(text) == 1:
            return text.upper()
        if text in choices:
            return letters[choices.index(text)]
        if not choices:
            return None
        distances = [_edit_distance(text.lower(), str(c).lower()) for c in choices]
        return letters[int(np.argmin(distances))]
    if atype in ("integer", "float"):
        try:
            value = float(text)
        except ValueError:
            value = numeric_value(text)
        if value is None:
            return None
        return str(int(value)) if atype == "integer" else str(round(value, int(meta.get("precision") or 0)))
    return text


def gold_mathvista(meta):
    answer = meta["answer"]
    if meta.get("question_type") == "multi_choice":
        choices = list(meta["choices"])
        return chr(ord("A") + choices.index(answer)) if answer in choices else normalize_mathvista(answer, meta)
    return normalize_mathvista(answer, meta)


def group_mathvista(answers, meta):
    # Non-null extractions that cannot be normalized keep their own (incorrect) groups by raw text;
    # only missing extractions form NO_ANSWER.
    normalized = [None if a is None else (normalize_mathvista(a, meta) or f"UNPARSED:{str(a).strip()}") for a in answers]
    keys, groups = {}, []
    for value in normalized:
        if value is None:
            groups.append(-1)
            continue
        keys.setdefault(value, len(keys))
        groups.append(keys[value])
    gold = gold_mathvista(meta)
    return np.array(groups), {g: value == gold for value, g in keys.items()}


def load_mathvista_metadata(cfg):
    from datasets import load_dataset
    m = cfg["mathvista_metadata"]
    data = load_dataset(m["hf_dataset"], split=m["split"])
    keep = ["pid", "question_type", "answer_type", "precision", "choices", "answer", "question"]
    return {str(row["pid"]): {k: row.get(k) for k in keep} for row in data.select_columns(keep)}


def label(questions, kind, metadata=None):
    """Attach group ids (NO_ANSWER = its own group) and per-rollout correctness."""
    for q in questions.values():
        if kind == "math":
            groups, group_correct = group_math(q.answers, q.gold)
        else:
            groups, group_correct = group_mathvista(q.answers, metadata[q.qid])
        no_answer = groups < 0
        groups = groups.copy()
        groups[no_answer] = groups.max(initial=-1) + 1 if no_answer.any() else 0
        if no_answer.any():
            group_correct = {**group_correct, int(groups[no_answer][0]): False}
        q.groups, q.no_answer = groups, no_answer
        q.correct = np.array([bool(group_correct[int(g)]) for g in groups])
    return questions
