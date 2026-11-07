from harness.canaries import JsonlCanarySet
from harness.canary_data import arc_rows, gsm8k_rows, write_jsonl
from harness.grading import grade_response
from harness.stats import canaries_needed
from harness.tasks import grading_scope
from harness.worlds import SUFFIX


GSM = [
    {"question": "How many?", "answer": "Work. #### 1,234"},
    {"question": "Another?", "answer": "#### -2.5"},
    {"question": "How many?", "answer": "#### 1,234"},
    {"question": "Bad", "answer": "no answer"},
    {"question": "Bad number", "answer": "#### 12,34"},
]
ARC = [
    {"id": "one", "question": "Which?", "choices": {"text": ["first", "second"], "label": ["1", "2"]}, "answerKey": "2"},
    {"id": "two", "question": "What?", "choices": {"text": ["yes", "no"], "label": ["A", "B"]}, "answerKey": "A"},
    {"id": "one", "question": "Which?", "choices": {"text": ["first", "second"], "label": ["1", "2"]}, "answerKey": "2"},
    {"id": "bad", "question": "Bad?", "choices": {"text": ["a"], "label": ["1", "2"]}, "answerKey": "1"},
]


def test_conversion_and_determinism():
    gsm = gsm8k_rows(GSM, 10, 0)
    arc = arc_rows(ARC, 10, 0)
    assert gsm == gsm8k_rows(GSM, 10, 0)
    assert arc == arc_rows(ARC, 10, 0)
    assert len(gsm) == len(arc) == 2
    assert {r[2] for r in gsm} == {"1234", "-2.5"}
    digit = next(r for r in arc if r[0] == "arc-one")
    assert "A. first\nB. second" in digit[1] and digit[2] == "B"
    for rows in (gsm, arc):
        assert len({r[0] for r in rows}) == len(rows)
        assert all(r[1].endswith(SUFFIX) for r in rows)
        assert len(rows[:1]) == 1
    assert len(gsm8k_rows(GSM, 1, 0)) == 1


def test_jsonl_and_grading(tmp_path):
    for name, rows in (("gsm", gsm8k_rows(GSM, 10, 0)), ("arc", arc_rows(ARC, 10, 0))):
        path = tmp_path / name / "rows.jsonl"
        write_jsonl(rows, path)
        items = JsonlCanarySet(name, path).items()
        assert len(items) == len(rows)
        for item, (_, prompt, expected) in zip(items, rows):
            with grading_scope():
                assert item.expected_at(0) == expected
            assert item.prompt == prompt
            assert grade_response(expected, "Final answer: " + expected).correct


def test_two_point_drop_sample_size():
    assert canaries_needed(0.02) >= 308
    assert canaries_needed(0.02, discordance=0.02) >= 308
