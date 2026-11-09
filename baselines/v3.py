"""Pure format helpers for the opt-in v3 arms and offline validity check."""
import re

from .arms import parse_answer, render_entry


FALLBACK_TRIGGER_K = 2


def shape_signature(text):
    """Keep length, punctuation and whitespace; erase case and letter/digit values."""
    return ''.join('a' if c.isalpha() else
                   '0' if c.isdigit() else c for c in text)


def demo_answer(text):
    """Use verified/correct targets, never the unverified response of a failure."""
    rendered = render_entry(text)
    verified = re.findall(r'^Verified answer:\s*(.+)$', rendered, re.M)
    if verified:
        return verified[-1].strip()
    correct = re.search(r'^(?:Correction: )?(Correct answer: .+)$', rendered, re.M)
    return parse_answer(correct.group(1), correction=True) if correct else None
