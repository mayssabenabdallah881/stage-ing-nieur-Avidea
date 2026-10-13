"""Comparaison de noms arabe / latin via un squelette consonantique (clé de comparaison uniquement)."""
import itertools
import re
from typing import List

_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_ARABIC_MAP = {
    "ا": "", "أ": "", "إ": "", "آ": "", "ى": "", "ء": "", "ئ": "", "ؤ": "", "ع": "", "غ": "g",
    "ة": "", "و": "", "ي": "",
    "ب": "b", "ت": "t", "ث": "t", "ج": "j", "ح": "h", "خ": "x", "د": "d", "ذ": "d",
    "ر": "r", "ز": "z", "س": "s", "ش": "s", "ص": "s", "ض": "d", "ط": "t", "ظ": "z",
    "ف": "f", "ق": "k", "ك": "k", "ل": "l", "م": "m", "ن": "n", "ه": "h", "ڨ": "g", "پ": "b",
}
_LATIN_DIGRAPHS = [("kh", "x"), ("sh", "s"), ("ch", "s"), ("th", "t"), ("dh", "d"),
                   ("gh", "g"), ("ph", "f"), ("dj", "j"), ("ou", ""), ("ee", ""), ("oo", "")]

def has_arabic(text: str) -> bool:
    return bool(_ARABIC_RE.search(text))

def _collapse(s: str) -> str:
    return re.sub(r"(.)\1+", r"\1", s)

def skeleton(token: str) -> str:
    token = token.casefold()
    if has_arabic(token):
        token = re.sub(r"[\u0647]$", "", token)
        out = "".join(_ARABIC_MAP.get(c, "") for c in token)
    else:
        for src, dst in _LATIN_DIGRAPHS:
            token = token.replace(src, dst)
        token = token.replace("q", "k").replace("c", "k").replace("v", "f").replace("p", "b")
        out = re.sub(r"[^a-z]", "", re.sub(r"[aeiouyhw]", lambda m: "h" if m.group() == "h" else "", token))
    return _collapse(out)

def _joined_variants(tokens):
    sks = [skeleton(t) for t in tokens]
    return {"".join(p) for p in itertools.permutations(sks)}


def cross_script_equal(tokens_a, tokens_b):
    a_ar = any(has_arabic(t) for t in tokens_a)
    b_ar = any(has_arabic(t) for t in tokens_b)
    if a_ar == b_ar or not tokens_a or not tokens_b or len(tokens_a) > 4 or len(tokens_b) > 4:
        return False
    va, vb = _joined_variants(tokens_a), _joined_variants(tokens_b)
    if min(len(x) for x in va | vb) < 2:
        return False
    if va & vb:
        return True
    return bool({x.replace("l", "") for x in va} & {x.replace("l", "") for x in vb})
