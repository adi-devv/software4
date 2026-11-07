"""Deterministic graders. Code only, never a model.

Rules (strict on purpose; a false pass is worse than a false fail):
* If the response has one or more lines starting "Final answer:" (also "####" lines), only those
  lines are graded; if they disagree with each other the response fails.
* With no final-answer line the response must be a single line.
* Numbers: thousands commas, "+", leading zeros and trailing ".0" are normalised. The graded
  text must contain exactly one distinct number, and it must equal the expected one.
* Text: case, whitespace, outer punctuation and a leading article are normalised, and a short
  wrapper like "the answer is" is stripped. Then it must EQUAL the expected text. Several
  candidates ("A or B", "A, B") therefore fail, and so does text that merely contains the answer.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from .tasks import AnswerLocked, Item, grading_active

_NUM = re.compile(r"(?<![\w.])[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?![\w])")
_FINAL = re.compile(r"^[\s>*_`#-]*(?:final\s+answer\s*[:=]|####)\s*(.*?)\s*$", re.I | re.M)
_WRAPPERS = ("the answer is", "my answer is", "answer is", "answer:", "it is", "it's", "that is",
             "this is", "is")
_EDGE = " \t\r\n.,;:!?\"'`*_()[]{}<>“”‘’"


@dataclass(frozen=True)
class GradeResult:
    correct: bool
    extracted: str
    reason: str
    stale: bool = False
    absorbed: bool = False


def norm_number(s: str):
    """Canonical string for a number, or None if s is not a number."""
    t = s.strip().replace(",", "")
    if t.startswith("+"):
        t = t[1:]
    try:
        d = Decimal(t)
    except InvalidOperation:
        return None
    if not d.is_finite():
        return None
    if d == d.to_integral():
        return str(int(d))
    return format(d.normalize(), "f")


def norm_text(s: str) -> str:
    t = unicodedata.normalize("NFKC", s).casefold()
    t = t.replace("**", "")
    t = re.sub(r"\s+", " ", t).strip(_EDGE)
    for art in ("the ", "a ", "an "):
        if t.startswith(art):
            t = t[len(art):]
            break
    return t


def _strip_wrapper(t: str) -> str:
    for w in _WRAPPERS:
        if t.startswith(w + " "):
            return norm_text(t[len(w):])
    return t


def _is_number(s: str) -> bool:
    t = s.strip()
    return bool(_NUM.fullmatch(t)) and norm_number(t) is not None


def _extract(response):
    """Return (text_to_grade, reason_if_unusable)."""
    r = response if isinstance(response, str) else str(response)
    finals = [m.group(1) for m in _FINAL.finditer(r)]
    if finals:
        normed = {norm_text(f) for f in finals}
        if len(normed) > 1:
            return None, "conflicting final answers"
        return finals[-1], ""
    stripped = r.strip()
    if "\n" in stripped:
        return None, "multi-line response without a final-answer line"
    return stripped, ""


def match(expected: str, text: str) -> bool:
    """Strict comparison of already-extracted text with the expected answer."""
    if _is_number(expected):
        want = norm_number(expected)
        nums = {norm_number(m.group(0)) for m in _NUM.finditer(text)}
        nums.discard(None)
        return len(nums) == 1 and next(iter(nums)) == want
    return _strip_wrapper(norm_text(text)) == norm_text(expected)


def grade_response(expected: str, response: str) -> GradeResult:
    text, why = _extract(response)
    if text is None:
        return GradeResult(False, "", why)
    ok = match(expected, text)
    return GradeResult(ok, text.strip(), "match" if ok else "no match")


def grade(item: Item, response: str, cycle: int) -> GradeResult:
    """Grade a response to an item at a given cycle. Needs a grading scope (the runner's)."""
    if not grading_active():
        raise AnswerLocked("grade() called outside a grading scope")
    text, why = _extract(response)
    if text is None:
        return GradeResult(False, "", why)
    ok = match(item.expected_at(cycle), text)
    stale = absorbed = False
    if not ok:
        stale = any(match(v, text) for v in item.stale_at(cycle))
        fv = item.false_value()
        absorbed = fv is not None and match(fv, text)
    return GradeResult(ok, text.strip(), "match" if ok else "no match", stale, absorbed)
