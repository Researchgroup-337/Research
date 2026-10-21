"""Adversarial-robust text normalisation.

Attackers evade keyword and ML filters with leetspeak (k1ll), spaced letters
(k i l l), zero-width characters, homoglyphs (Cyrillic 'а' for Latin 'a') and
character stretching (diiiie). normalize() undoes these before scoring, and
perturb() produces the same attacks for robustness testing.
"""
from __future__ import annotations

import random
import re
import unicodedata

ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)

# Common Cyrillic/Greek look-alikes -> Latin.
HOMOGLYPHS = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i",
    "ј": "j", "ѕ": "s", "к": "k", "м": "m", "т": "t", "в": "b", "н": "h",
    "α": "a", "ο": "o", "ε": "e", "ι": "i", "κ": "k", "ν": "v", "τ": "t", "υ": "u",
}
LEET = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i", "|": "l"}

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.I)
MENTION_RE = re.compile(r"@\w+")
HTML_ENT_RE = re.compile(r"&#?\w+;")
SPACED_RE = re.compile(r"\b(?:[a-z][\s.\-_*]){2,}[a-z]\b")
REPEAT_RE = re.compile(r"(.)\1{2,}")
WORD_LEET_RE = re.compile(r"\b\w*[a-z]\w*\b")


def _deleet_token(tok: str) -> str:
    # Only translate digits/symbols inside tokens that also contain letters,
    # so plain numbers ("2024", "100") are left alone.
    if any(c.isalpha() for c in tok) and any(c in LEET for c in tok):
        return "".join(LEET.get(c, c) for c in tok)
    return tok


def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFKC", text).translate(ZERO_WIDTH)
    t = "".join(HOMOGLYPHS.get(c, c) for c in t)
    t = HTML_ENT_RE.sub(" ", t)
    t = URL_RE.sub(" xxurl ", t)
    t = MENTION_RE.sub(" xxuser ", t)
    t = t.lower()
    t = " ".join(_deleet_token(tok) for tok in re.split(r"(\s+)", t) if tok.strip())
    t = SPACED_RE.sub(lambda m: re.sub(r"[\s.\-_*]", "", m.group(0)), t)
    t = REPEAT_RE.sub(r"\1\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ---------------------------------------------------------------- attacks
_INV_LEET = {"o": "0", "i": "1", "e": "3", "a": "4", "s": "5", "t": "7"}
_INV_HOMO = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "i": "і"}


def perturb(text: str, kind: str, rng: random.Random | None = None, rate: float = 0.5) -> str:
    """Apply an evasion attack. kind in {leet, spaced, zero_width, homoglyph, stretch, mixed}."""
    rng = rng or random.Random(0)
    words = text.split()

    def pick(w):
        return len(w) > 3 and rng.random() < rate

    if kind == "leet":
        words = ["".join(_INV_LEET.get(c, c) for c in w) if pick(w) else w for w in words]
    elif kind == "spaced":
        words = [" ".join(w) if pick(w) else w for w in words]
    elif kind == "zero_width":
        words = ["​".join(w) if pick(w) else w for w in words]
    elif kind == "homoglyph":
        words = ["".join(_INV_HOMO.get(c, c) for c in w) if pick(w) else w for w in words]
    elif kind == "stretch":
        words = [w[:-1] + w[-1] * 4 if pick(w) else w for w in words]
    elif kind == "mixed":
        kinds = ["leet", "spaced", "zero_width", "homoglyph", "stretch"]
        return " ".join(perturb(w, rng.choice(kinds), rng, rate=1.0) if pick(w) else w for w in words)
    else:
        raise ValueError(kind)
    return " ".join(words)
