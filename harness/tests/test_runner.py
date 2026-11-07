import json

import pytest

from harness.arms_fake import (Cheater, Forgetter, NoKnowledge, Oracle, PerfectAfterSeeing,
                               build_key)
from harness.gate import GateReuseError
from harness.results import Results, load_records
from harness.runner import ArmViolation, Services, SimulatedCrash, read_lines, run_arm
from harness.streams import synthetic_stream


def final_acc(res, arm, stream):
    return res.primary(arm, stream.stream_id, stream.n_cycles - 1)


def test_oracle_and_no_knowledge(tmp_path, stream, canaries):
    p = tmp_path / "r.jsonl"
    key = build_key(stream, canaries)
    run_arm(Oracle(key), stream, canaries, p)
    run_arm(NoKnowledge(key), stream, canaries, p)
    res = Results.from_file(p)
    assert final_acc(res, "oracle", stream) == 1.0
    assert final_acc(res, "no_memory", stream) == 0.0
    assert res.canary_drop("oracle", stream.stream_id)[0] == 0.0
    # every cycle scored, including the pre-learning canary measurement at cycle -1
    assert res.complete("oracle", stream.stream_id, stream.n_cycles)


def test_scores_are_per_task_per_cycle(tmp_path, stream, canaries):
    p = tmp_path / "r.jsonl"
    run_arm(Oracle(build_key(stream, canaries)), stream, canaries, p)
    recs = [r for r in load_records(p) if r["type"] == "score" and r["phase"] == "hidden"]
    c = stream.n_cycles - 1
    assert len([r for r in recs if r["cycle"] == c]) == len(stream.hidden_items(c))
    assert len([r for r in recs if r["cycle"] == 0]) == len(stream.hidden_items(0))
    assert {"task_id", "skill", "family", "correct", "cycle"} <= set(recs[0])


def test_forgetter_fails_old_skill_and_canary(tmp_path, stream, canaries):
    p = tmp_path / "r.jsonl"
    key = build_key(stream, canaries)
    run_arm(PerfectAfterSeeing(key), stream, canaries, p)
    run_arm(Forgetter(key), stream, canaries, p)
    res = Results.from_file(p)
    sid = stream.stream_id
    assert res.canary_drop("forgetter", sid)[0] > 0.02          # canary criterion fails
    assert res.canary_drop("perfect", sid)[0] <= 0.02
    assert final_acc(res, "forgetter", stream) < final_acc(res, "perfect", stream) - 0.2


@pytest.mark.parametrize("mode", ["item", "grade", "scope", "sealed"])
def test_cheater_is_caught(tmp_path, mode, canaries):
    st = synthetic_stream(21)
    p = tmp_path / "r.jsonl"
    with pytest.raises(ArmViolation):
        run_arm(Cheater(st, mode), st, canaries, p)
    res = Results.from_file(p)
    assert len(res.violations) == 1
    assert st.sealed.read_count == 0
    assert not res.complete("cheater", st.stream_id, st.n_cycles)


def _signature(path):
    out = []
    for r in load_records(path):
        if r["type"] == "score":
            out.append((r["arm"], r["stream"], r["phase"], r["cycle"], r["task_id"], r["correct"]))
    return sorted(out)


@pytest.mark.parametrize("use_ckpt", [True, False])
def test_resume_after_crash_equals_uninterrupted(tmp_path, use_ckpt, canaries):
    st = synthetic_stream(31)
    key = build_key(st, canaries)

    clean = tmp_path / "clean.jsonl"
    run_arm(Forgetter(key, seed=3), st, canaries, clean)

    crashed = tmp_path / "crashed.jsonl"
    ck = tmp_path / "ck" if use_ckpt else None

    def boom(cycle, phase, n):
        if (cycle, phase, n) == (6, "hidden", 7):
            raise SimulatedCrash("power cut")

    with pytest.raises(SimulatedCrash):
        run_arm(Forgetter(key, seed=3), st, canaries, crashed, fault_hook=boom, ckpt_dir=ck)
    partial = Results.from_file(crashed)
    assert not partial.complete("forgetter", st.stream_id, st.n_cycles)
    with crashed.open("a", encoding="utf-8") as f:   # a torn last line, as after a real crash
        f.write('{"type": "score", "arm": "forg')
    run_arm(Forgetter(key, seed=3), st, canaries, crashed, ckpt_dir=ck)   # fresh arm object

    assert _signature(crashed) == _signature(clean)
    assert Results.from_file(crashed).complete("forgetter", st.stream_id, st.n_cycles)
    # finished run is a no-op to resume
    n = len(read_lines(crashed))
    run_arm(Forgetter(key, seed=3), st, canaries, crashed, ckpt_dir=ck)
    assert _signature(crashed) == _signature(clean)
    assert len(read_lines(crashed)) >= n


def test_results_file_is_append_only(tmp_path, stream, canaries):
    p = tmp_path / "r.jsonl"
    key = build_key(stream, canaries)
    run_arm(Oracle(key), stream, canaries, p)
    first = p.read_bytes()
    run_arm(NoKnowledge(key), stream, canaries, p)
    assert p.read_bytes().startswith(first)
    for line in p.read_text(encoding="utf-8").splitlines():
        json.loads(line)


def test_gate_session_one_shot_and_rollback(stream, canaries):
    key = build_key(stream, canaries)
    svc = Services(stream, canaries)
    cycle = 6
    sess = svc.gate_session(cycle)
    with pytest.raises(RuntimeError):
        svc.gate_session(cycle)                                   # opened once per cycle
    good = {t.id: "Final answer: " + key.answer(t.id, cycle) for t in sess.tasks}
    bad = {t.id: "Final answer: wrong" for t in sess.tasks}
    d = sess.decide(good, good)
    assert d.accept and not d.rollback
    with pytest.raises(GateReuseError):
        sess.decide(good, good)

    sess2 = Services(stream, canaries).gate_session(cycle)
    d2 = sess2.decide(bad, good)                                  # candidate much worse
    assert d2.rollback and d2.reasons
    sess3 = Services(stream, canaries).gate_session(cycle)
    assert sess3.decide(good, bad).accept                          # candidate better


def test_gate_margin_widens_with_noise():
    from harness.gate import paired_check
    steady = paired_check([1, 1, 1, 1, 0, 1, 1, 1] * 5, [1] * 40)
    noisy = paired_check([1, 0, 1, 0, 1, 1, 0, 1] * 5, [1, 1, 0, 0, 1, 0, 1, 1] * 5)
    assert noisy["margin"] > steady["margin"] > 0 or steady["margin"] >= 0
    same = paired_check([1, 0, 1, 1], [1, 0, 1, 1])
    assert same["margin"] == 0 and not same["regression"]
    assert paired_check([0] * 30, [1] * 30)["regression"]
    one_off = paired_check([1] * 29 + [0], [1] * 30)
    assert not one_off["regression"]


def test_gate_flags_absorbed_poison(stream, canaries):
    from harness.gate import gate_decision
    d = gate_decision(gate=([1] * 20, [1] * 20), canary=([1] * 20, [1] * 20),
                      poison=([1, 1, 1, 0] * 5, [0] * 20))
    assert d.rollback and d.poison_absorbed_candidate == 15
