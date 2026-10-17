"""Typo correction for chat questions, before anything else reads them: "Any fire incdent" -> "Any fire incident",
"acident on mg road" -> "accident ...", "figths today" -> "fights today". Rule-based and instant (no model call).

A word is corrected only when it is unknown and close to exactly one word the chat understands:
- unknown: not a word the chat already understands, and not a word in any stored summary (so a real but rare word
  that the cameras described, like "flight", is never "fixed")
- long enough: 5 letters or more (4-letter words have too many real neighbours: mall/fall, last/list)
- close: 1 edit, or 2 for words of 7+ letters that start with the same letter; swapped neighbours count as one
- the target vocabulary: category names and their keywords, plus the question and time words of the parser
Place names have their own fuzzy matching in the parser (parser._close) and are left alone here."""

from __future__ import annotations

import re

from aequitas.taxonomy import KEYWORDS, label

# Words the parser reads that people often mistype (time, status and question words).
EXTRA = """incident incidents happened anything anyone someone yesterday today tonight morning afternoon evening
week hours minutes confirmed unconfirmed disputed rejected between before after since everywhere camera cameras
show list where what which many violence violent fight fights fighting fires accident accidents crash crashes
stampede robbery robberies snatching knife knives pistol smoke flames collision loitering abandoned unattended
luggage arrests arrested police harassment vandalism weapon weapons theft thefts chase chasing riots rioting crowd
crowds suspicious tailgating person down fallen""".split()


def targets(categories) -> set[str]:
    """Every word the chat understands that a typo may be corrected to."""
    words = set(EXTRA)
    for c in categories:
        words.update(w for w in re.findall(r"[a-z]+", label(c)) if len(w) >= 4)
        # Whole words from the keyword patterns: "fires?" gives fire and fires; stems such as "thie\w*" are skipped
        # (they are not words, and "thief" would be "corrected" to "thie").
        for w, rest in re.findall(r"(?<![a-z\\])([a-z]+)(\\w\*|\?|)", KEYWORDS.get(c, "")):
            if rest == "\\w*":
                continue
            forms = [w, w[:-1]] if rest == "?" else [w]
            words.update(f for f in forms if len(f) >= 4)
    return words


def distance(a: str, b: str, cap: int = 3) -> int:
    """Optimal string alignment distance (Levenshtein + adjacent swaps), stopping early above `cap`."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > cap:
            return cap + 1
        prev2, prev = prev, cur
    return prev[-1]


# Real words one letter from a target ("light" -> fight, "fine" -> fire): never corrected.
COMMON = set("""light lights right night nights sight tight might eight bright flight forest chair share crown brown
cloud clown first""".split())


def best_match(word: str, vocab: set[str]) -> str | None:
    allowed = 1 if len(word) <= 5 else 2
    found: dict[int, set[str]] = {}
    for t in vocab:
        if abs(len(t) - len(word)) <= allowed:
            d = distance(word, t, allowed)
            # two edits only for longer words that start with the same letter ("forest" is not "arrest")
            if d <= 1 or (d == 2 and len(word) >= 7 and word[0] == t[0]):
                found.setdefault(d, set()).add(t)
    if not found:
        return None
    nearest = found[min(found)]
    return next(iter(nearest)) if len(nearest) == 1 else None  # two equally close words: don't guess


def correct(question: str, vocab: set[str], known: set[str]) -> tuple[str, list[tuple[str, str]]]:
    """The question with typos fixed, and the fixes made [(typed, corrected)]."""
    fixes: list[tuple[str, str]] = []

    def fix(m: re.Match) -> str:
        w = m.group(0)
        low = w.lower()
        if len(low) < 5 or low in vocab or low in known or low in COMMON:  # 4-letter words: too many real neighbours
            return w
        t = best_match(low, vocab)
        if t is None:
            return w
        fixes.append((w, t))
        return t.capitalize() if w[0].isupper() else t

    return re.sub(r"[A-Za-z]+", fix, question), fixes
