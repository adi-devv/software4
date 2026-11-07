import pytest

from harness.grading import grade, grade_response, norm_number, norm_text
from harness.tasks import AnswerLocked, Item, grading_scope


def ok(exp, resp):
    return grade_response(exp, resp).correct


def test_number_normalisation():
    assert norm_number("1,234") == norm_number("1234.0") == norm_number("+1234")
    assert ok("42", "Final answer: 42")
    assert ok("1234", "work\nFinal answer: 1,234")
    assert ok("7", "Final answer: 007")


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
