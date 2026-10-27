"""VQA answer normalization, soft consensus accuracy, and pair-level bootstrap."""
from __future__ import annotations

import re

import numpy as np

# Normalization follows the official VQA evaluation rules (punctuation, digits, articles, contractions).
CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't",
    "dont": "don't", "hadnt": "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
    "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's", "howd": "how'd",
    "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm", "Ive": "I've",
    "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've",
    "mightve": "might've", "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
    "oclock": "o'clock", "oughtnt": "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at",
    "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
    "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd",
    "somethingd've": "something'd've", "something'dve": "something'd've", "somethingll": "something'll",
    "thats": "that's", "thered": "there'd", "thered've": "there'd've", "there'dve": "there'd've",
    "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've",
    "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't",
    "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll",
    "whatre": "what're", "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd",
    "wheres": "where's", "whereve": "where've", "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've",
    "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", "whyre": "why're", "whys": "why's",
    "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
    "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd",
    "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll", "youre": "you're", "youve": "you've"}
NUMBERS = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
           "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
ARTICLES = {"a", "an", "the"}
PUNCTUATION = [";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",",
               "?", "!"]
PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA_STRIP = re.compile(r"(\d)(,)(\d)")


def _punctuation(text):
    out = text
    for p in PUNCTUATION:
        if (p + " " in text or " " + p in text) or re.search(COMMA_STRIP, text) is not None:
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    return PERIOD_STRIP.sub("", out, re.UNICODE)


def _digits_articles(text):
    words = []
    for word in text.lower().split():
        word = NUMBERS.get(word, word)
        if word not in ARTICLES:
            words.append(CONTRACTIONS.get(word, word))
    return " ".join(words)


def normalize_answer(text):
    text = (text or "").replace("\n", " ").replace("\t", " ").strip()
    return _digits_articles(_punctuation(text)).strip()


def soft_accuracy(prediction, answers):
    """Official leave-one-annotator-out VQA accuracy against all ten normalized human answers."""
    prediction = normalize_answer(prediction)
    answers = [normalize_answer(a) for a in answers]
    scores = []
    for i in range(len(answers)):
        matches = sum(1 for j, a in enumerate(answers) if j != i and a == prediction)
        scores.append(min(1.0, matches / 3))
    return float(np.mean(scores))


def consensus(answers):
    """Most frequent normalized answer and its count; ties resolved by first occurrence."""
    normalized = [normalize_answer(a) for a in answers]
    counts = {}
    for a in normalized:
        counts[a] = counts.get(a, 0) + 1
    best = max(counts, key=lambda a: (counts[a], -normalized.index(a)))
    return best, counts[best]


def representative(answers, target):
    """An original answer string from the consensus group (most frequent raw form)."""
    raw = [a for a in answers if normalize_answer(a) == target]
    return max(raw, key=lambda a: (raw.count(a), -raw.index(a)))


def pair_bootstrap(values_by_pair, iterations, seed):
    """95% interval for the mean of a per-pair statistic; resampled pair indices are shared."""
    values = np.asarray(values_by_pair, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(iterations, len(values)))
    means = values[idx].mean(axis=1)
    return {"mean": float(values.mean()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}
