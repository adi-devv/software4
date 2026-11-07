import json

import pytest

from harness.arms_fake import Noisy, build_key
from harness.results import Results
from harness.runner import run_arm, run_sealed_milestone
from harness.streams import synthetic_stream
from harness.tasks import SealedAccessError, arm_scope


def fresh(**kw):
    return synthetic_stream(11, **kw)


def test_ordinary_access_raises():
    s = fresh().sealed
    for f in (lambda: list(s), lambda: s[0], lambda: len(s), lambda: 1 in s, lambda: s.items):
        with pytest.raises(SealedAccessError):
            f()
    assert s.read_count == 0


def test_read_needs_milestone_and_reason():
    s = fresh().sealed
    with pytest.raises(SealedAccessError):
        s.read(milestone="m", reason="")
    with pytest.raises(SealedAccessError):
        s.read(milestone="m", reason="short")
    with pytest.raises(SealedAccessError):
        s.read(milestone="", reason="a perfectly good reason")
    assert s.read_count == 0


def test_read_is_logged_and_counted(tmp_path):
    log = tmp_path / "sealed.log"
    st = fresh(sealed_log_path=log)
    items = st.sealed.read(milestone="m1", reason="final check before writing up")
    assert items and st.sealed.read_count == 1
    st.sealed.read(milestone="m2", reason="second look at one skill", skill=items[0].skill)
    assert st.sealed.read_count == 2
    lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    assert [x["milestone"] for x in lines] == ["m1", "m2"]
    assert st.sealed.access_log[0]["reason"].startswith("final check")


def test_read_refused_while_arm_runs():
    s = fresh().sealed
    with arm_scope():
        with pytest.raises(SealedAccessError):
            s.read(milestone="m", reason="trying while an arm is running")
    assert s.read_count == 0


def test_milestone_run_scored_and_reported(tmp_path, canaries):
    st = fresh()
    key = build_key(st, canaries)
    path = tmp_path / "r.jsonl"
    arm = Noisy(key, 1.0, name="x", canary_p=1.0)
    run_arm(arm, st, canaries, path)
    assert st.sealed.read_count == 0
    n = run_sealed_milestone(arm, st, path, milestone="m1", reason="end-of-project check")
    assert n > 0 and st.sealed.read_count == 1
    res = Results.from_file(path)
    assert res.sealed_reads == 1
    assert res.sealed_events[0]["reason"] == "end-of-project check"
    sealed = [r for r in res.records if r["type"] == "score" and r["phase"] == "sealed"]
    assert len(sealed) == n and all(r["correct"] for r in sealed)
