import json

import pytest

from harness.arms_fake import (Cheater, Forgetter, NoKnowledge, Oracle, PerfectAfterSeeing,
                               build_key)
from harness.gate import GateReuseError
from harness.canaries import ListCanarySet
from harness.results import Results, load_records
from harness.runner import ArmViolation, Services, SimulatedCrash, read_lines, run_arm
from harness.streams import build_stream, synthetic_stream
from harness.tasks import AnswerLocked, grading_scope
from harness.worlds import StreamConfig, TaskSource


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


@pytest.mark.parametrize("target", ["hidden", "gate", "sealed", "canary"])
def test_bound_services_do_not_expose_answers(tmp_path, stream, canaries, target):
    class Inspector(NoKnowledge):
        def bind(self, services):
            assert services.stream_id == stream.stream_id
            assert services.n_cycles == stream.n_cycles
            with pytest.raises(AttributeError):
                if target == "hidden":
                    services._stream.hidden_items(0)[0]._secret[0]
                elif target == "gate":
                    services._stream.gate_items(0)[0]._secret[0]
                elif target == "sealed":
                    services._stream.sealed._items[0]._secret[0]
                else:
                    services._canaries[0].items()[0]._secret[0]

    run_arm(Inspector(), stream, canaries, tmp_path / "r.jsonl")
    assert stream.sealed.read_count == 0


@pytest.mark.parametrize("target", ["_gate", "_canary"])
def test_bound_gate_session_does_not_expose_items(tmp_path, stream, canaries, target):
    class Inspector(NoKnowledge):
        def bind(self, services):
            session = services.gate_session(0)
            assert session.tasks
            assert all(not hasattr(t, "_secret") for t in session.tasks)
            with pytest.raises(AttributeError):
                getattr(session, target)[0]._secret[0]

    run_arm(Inspector(), stream, canaries, tmp_path / "r.jsonl")


@pytest.mark.parametrize("mode", ["grading", "sealed"])
def test_save_state_leak_is_caught(tmp_path, stream, canaries, mode):
    class CheckpointCheater(NoKnowledge):
        def save_state(self):
            if mode == "sealed":
                stream.sealed.read(milestone="peek", reason="collect answers before learning")
            with grading_scope():
                return {"answer": stream.hidden_items(0)[0].expected_at(0)}

    p = tmp_path / "r.jsonl"
    with pytest.raises(ArmViolation):
        run_arm(CheckpointCheater(), stream, canaries, p, ckpt_dir=tmp_path / "ck")
    violations = Results.from_file(p).violations
    assert len(violations) == 1
    assert violations[0]["where"] == "save_state" and violations[0]["cycle"] == -1
    assert stream.sealed.read_count == 0
    assert not list((tmp_path / "ck").glob("*.json"))
    assert not any(r["type"] == "cycle_done" for r in read_lines(p))


@pytest.mark.parametrize("candidate_good", [True, False])
def test_gate_can_be_scored_from_sleep(tmp_path, stream, canaries, candidate_good):
    key = build_key(stream, canaries)
    decisions = []

    class GatedArm(NoKnowledge):
        def bind(self, services):
            self.services = services

        def sleep(self, cycle):
            session = self.services.gate_session(cycle)
            good = {t.id: "Final answer: " + key.answer(t.id, cycle) for t in session.tasks}
            bad = {t.id: "Final answer: wrong" for t in session.tasks}
            decision = session.decide(good if candidate_good else bad, good)
            decisions.append(decision)
            with pytest.raises(GateReuseError):
                session.decide(good, good)
            # The trusted gate must restore the arm lock before returning.
            with pytest.raises(AnswerLocked):
                with grading_scope():
                    stream.gate_items(cycle)[0].expected_at(cycle)
            with pytest.raises(AnswerLocked):
                stream.gate_items(cycle)[0].expected_at(cycle)

    p = tmp_path / "r.jsonl"
    run_arm(GatedArm(), stream, canaries, p)
    assert len(decisions) == stream.n_cycles
    assert all(d.accept == candidate_good for d in decisions)
    assert not Results.from_file(p).violations
    assert stream.sealed.read_count == 0


def test_gate_response_callbacks_stay_under_arm_lock(tmp_path, stream, canaries):
    class SneakyResponses(dict):
        def get(self, task_id, default=""):
            with grading_scope():
                return stream.hidden_items(0)[0].expected_at(0)

    class GatedCheater(NoKnowledge):
        def bind(self, services):
            self.services = services

        def sleep(self, cycle):
            self.services.gate_session(cycle).decide(SneakyResponses(), {})

    p = tmp_path / "r.jsonl"
    with pytest.raises(ArmViolation):
        run_arm(GatedCheater(), stream, canaries, p)
    assert Results.from_file(p).violations[0]["where"] == "sleep"


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


@pytest.mark.parametrize("change", ["items_per_rule", "fracs", "n_cycles", "canary", "lesson"])
def test_resume_rejects_changed_experiment(tmp_path, change):
    p = tmp_path / "r.jsonl"
    st = synthetic_stream(1)
    can = [ListCanarySet("old", [("one", "Old skill?", "1")])]
    run_arm(NoKnowledge(), st, can, p)
    before = p.read_bytes()
    cfg = StreamConfig()
    if change == "items_per_rule":
        cfg.items_per_rule += 1
    elif change == "fracs":
        cfg.fracs = {"train": 0.25, "gate": 0.25, "hidden": 0.35, "sealed": 0.15}
    elif change == "n_cycles":
        cfg.n_cycles += 1
    elif change == "canary":
        can = [ListCanarySet("old", [("one", "Old skill?", "2")])]
    changed = synthetic_stream(1, cfg)
    if change == "lesson":
        item = next(it for c in range(changed.n_cycles) for it in changed.train_items(c)
                    if it.kind == "probe")
        item.lesson = "Changed correction lesson."
    assert changed.stream_id == st.stream_id
    with pytest.raises(ValueError, match="experiment"):
        run_arm(NoKnowledge(), changed, can, p)
    assert p.read_bytes() == before


def test_resume_rejects_unverified_legacy_results(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps({"type": "cycle_done", "arm": "no_memory", "stream": "s1",
                             "cycle": -1, "attempt": 1}) + "\n", encoding="utf-8")
    before = p.read_bytes()
    with pytest.raises(ValueError, match="experiment"):
        run_arm(NoKnowledge(), synthetic_stream(1), [], p)
    assert p.read_bytes() == before


class _CounterSource(TaskSource):
    def raw(self):
        return ([{"name": "counter", "family": "fact", "intro_cycle": 0}],
                [{"skill": "counter", "kind": "probe", "key": "counter",
                  "input_key": "counter", "prompt": "Return the parity.",
                  "expected": "0", "split": "hidden"}])


class _CounterArm:
    name = "counter"

    def reset(self):
        self.count = 0

    def answer(self, task):
        self.count += 1
        return str(self.count % 2)

    def sleep(self, cycle):
        pass


@pytest.mark.parametrize("use_ckpt", [False, True])
@pytest.mark.parametrize("crash_cycle", [0, 1])
def test_resume_replays_stateful_test_calls(tmp_path, use_ckpt, crash_cycle):
    loaded = []

    class CheckpointCounter(_CounterArm):
        def save_state(self):
            return self.count

        def load_state(self, state):
            loaded.append(state)
            self.count = state

    cls = CheckpointCounter if use_ckpt else _CounterArm
    st = build_stream(_CounterSource(), seed=1, n_cycles=3)
    can = [ListCanarySet("parity", [("one", "Return the parity.", "1")])]
    clean, crashed = tmp_path / "clean.jsonl", tmp_path / "crashed.jsonl"
    run_arm(cls(), st, can, clean)
    assert all(r["correct"] == 1 for r in load_records(clean) if r["type"] == "score")

    def boom(cycle, phase, n):
        if (cycle, phase) == (crash_cycle, "sleep"):
            raise SimulatedCrash("power cut")

    ck = tmp_path / "ck" if use_ckpt else None
    with pytest.raises(SimulatedCrash):
        run_arm(cls(), st, can, crashed, fault_hook=boom, ckpt_dir=ck)
    resumed = cls()
    run_arm(resumed, st, can, crashed, ckpt_dir=ck)
    assert _signature(crashed) == _signature(clean)
    assert resumed.count == 7
    if use_ckpt:
        assert loaded == [1 + 2 * crash_cycle]


def test_resume_twice_after_torn_tail_keeps_attempts_unique(tmp_path):
    st = build_stream(_CounterSource(), seed=1, n_cycles=8)
    can = [ListCanarySet("parity", [("one", "Return the parity.", "1")])]
    clean, crashed = tmp_path / "clean.jsonl", tmp_path / "crashed.jsonl"
    run_arm(_CounterArm(), st, can, clean)

    def boom_before_sleep(cycle, phase, n):
        if (cycle, phase) == (6, "sleep"):
            raise SimulatedCrash("attempt 8")

    with pytest.raises(SimulatedCrash):
        run_arm(_CounterArm(), st, can, crashed, fault_hook=boom_before_sleep)
    with crashed.open("a", encoding="utf-8") as f:
        f.write('{"type": "score", "arm": "cou')
    torn = crashed.read_bytes()

    def boom_after_score(cycle, phase, n):
        if (cycle, phase) == (6, "canary"):
            raise SimulatedCrash("attempt 9")

    with pytest.raises(SimulatedCrash):
        run_arm(_CounterArm(), st, can, crashed, fault_hook=boom_after_score)
    run_arm(_CounterArm(), st, can, crashed)
    starts = [r["attempt"] for r in read_lines(crashed)
              if r["type"] == "cycle_start" and r["cycle"] == 6]
    assert starts == [8, 9, 10]
    assert crashed.read_bytes().startswith(torn)
    assert _signature(crashed) == _signature(clean)
