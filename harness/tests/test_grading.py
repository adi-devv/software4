from concurrent.futures import ThreadPoolExecutor

import pytest

from harness.grading import grade, grade_response, norm_number, norm_text
from harness.sealed import SealedStore
from harness.tasks import (AnswerLocked, Item, SealedAccessError, _gate_grading_scope, arm_scope,
                           grading_scope)


def ok(exp, resp):
    return grade_response(exp, resp).correct


def test_number_normalisation():
    assert norm_number("1,234") == norm_number("1234.0") == norm_number("+1234")
    assert ok("42", "Final answer: 42")
    assert ok("1234", "work\nFinal answer: 1,234")
    assert ok("7", "Final answer: 007")


@pytest.mark.parametrize("resp", [
    "Final answer: 42/42",
    "Final answer: not 42",
    "Final answer: 42%",
    "Final answer: 42 - 42",
    "Final answer: 42 * 42",
    "Final answer: the answer is not 42",
])
def test_numeric_expressions_denials_and_percentages_fail(resp):
    assert not ok("42", resp)


@pytest.mark.parametrize("expected,resp", [
    ("42", "Final answer: The answer is **42**."),
    ("-2.5", "Final answer: -2.500."),
    ("0.5", "Final answer: .5"),
    ("0.5", "Final answer: The answer is .5."),
    ("1234", "Final answer: +1,234.0"),
])
def test_numeric_wrappers_preserve_number_normalisation(expected, resp):
    assert ok(expected, resp)


def test_text_normalisation():
    assert norm_text("  The Red  Fox. ") == "red fox"
    assert ok("Red Fox", "Final answer: the red fox.")
    assert ok("Red Fox", "Final answer: The answer is red fox")


@pytest.mark.parametrize("resp", [
    "Final answer: 41 or 42",
    "Final answer: 41, 42",
    "Final answer: 42 and 43",
    "Final answer: it could be 42 but maybe 43",
    "Final answer: 41\nFinal answer: 42",
    "Maybe 42.\nOr 41.",
    "Final answer: wrong",
    "",
])
def test_multi_candidate_or_ambiguous_fails_number(resp):
    assert not ok("42", resp)


@pytest.mark.parametrize("resp", [
    "Final answer: Red Fox or Blue Fox",
    "Final answer: red fox, blue fox",
    "Final answer: probably the red fox, not sure",
    "Final answer: Red Fox\nFinal answer: Blue Fox",
])
def test_multi_candidate_fails_text(resp):
    assert not ok("Red Fox", resp)


def test_repeated_same_final_is_fine():
    assert ok("42", "Final answer: 42\nFinal answer: 42")


def test_grade_needs_scope_and_flags_stale_and_absorbed():
    it = Item(id="x", skill="s", family="revised", kind="probe", key="k", input_key="k",
              prompt="p", split="hidden", expected="old", history=[(3, "new")], prior=("older",),
              false_value="fake")
    with pytest.raises(AnswerLocked):
        grade(it, "Final answer: new", 5)
    with pytest.raises(AnswerLocked):
        it.expected_at(5)
    with grading_scope():
        assert grade(it, "Final answer: new", 5).correct
        g = grade(it, "Final answer: old", 5)
        assert not g.correct and g.stale
        assert grade(it, "Final answer: older", 5).stale
        assert grade(it, "Final answer: old", 1).correct
        g = grade(it, "Final answer: fake", 5)
        assert not g.correct and g.absorbed


@pytest.mark.parametrize("fail", [False, True])
def test_gate_grading_scope_keeps_other_threads_locked(fail):
    it = Item(id="x", skill="s", family="rule", kind="probe", key="k", input_key="k",
              prompt="p", split="gate", expected="42")
    sealed = SealedStore([it])

    def peek():
        with pytest.raises(AnswerLocked):
            it.expected_at(0)
        with pytest.raises(AnswerLocked):
            with grading_scope():
                it.expected_at(0)
        with pytest.raises(SealedAccessError):
            sealed.read(milestone="peek", reason="read during gate scoring")

    with arm_scope():
        try:
            with _gate_grading_scope():
                assert grade(it, "42", 0).correct
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(peek).result(timeout=5)
                if fail:
                    raise ValueError("scoring failed")
        except ValueError:
            assert fail
        with pytest.raises(AnswerLocked):
            it.expected_at(0)
        with pytest.raises(AnswerLocked):
            with grading_scope():
                pass
    assert sealed.read_count == 0
    with grading_scope():
        assert it.expected_at(0) == "42"
