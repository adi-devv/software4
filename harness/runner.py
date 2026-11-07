"""Runner: plays a stream through an arm, writes crash-safe JSONL, resumes without redoing work.

Per (arm, stream):
  cycle -1: canaries measured before anything is learned (the baseline for drops)
  cycle c = 0..n-1: day tasks (answer -> grade -> observe), then arm.sleep(c), then frozen hidden
                    tests and canaries (same canary items every cycle)
Every record is appended as one JSON line when it happens. Each cycle writes `cycle_start` (with an
attempt number), score records, then `cycle_done`. The loader keeps only records of an attempt that
reached `cycle_done`, so a crash mid-cycle leaves harmless debris, and a rerun redoes only that cycle.

Resuming a stateful arm: if it has save_state/load_state, the checkpoint of the last finished cycle
is loaded. Otherwise the finished cycles' day tasks and sleeps are replayed (not re-scored or
re-logged), which only reproduces the arm's state if the arm is deterministic.

Leak guard: arm calls run inside `arm_scope()`; the arm gets `Task` objects with no answers;
`observe` is only ever called for train items; a grading scope or sealed read attempted while arm
code runs raises, the attempt is logged as a `violation` record, and the run stops with ArmViolation.
"""
from __future__ import annotations

import json
from pathlib import Path

from .gate import GateSession
from .grading import grade
from .tasks import AnswerLocked, SealedAccessError, arm_scope, grading_scope


class ArmViolation(RuntimeError):
    """An arm tried to reach answers it must not have."""


class SimulatedCrash(RuntimeError):
    """Raised by test fault hooks."""


class Services:
    """Handed to arms that define bind(). Gives gate sessions; never answers."""

    def __init__(self, stream, canaries):
        self.stream_id = stream.stream_id
        self.n_cycles = stream.n_cycles
        self._stream = stream
        self._canaries = list(canaries)
        self._opened = set()

    def gate_session(self, cycle: int) -> GateSession:
        """Fresh gate batch + canaries for this cycle. Can be opened once per cycle."""
        if cycle in self._opened:
            raise RuntimeError(f"gate session for cycle {cycle} already opened")
        self._opened.add(cycle)
        can = [i for cs in self._canaries for i in cs.items()]
        return GateSession(cycle, self._stream.gate_items(cycle), can)


class _Writer:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("a", encoding="utf-8", newline="\n")

    def write(self, rec):
        self.f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")

    def flush(self):
        self.f.flush()

    def close(self):
        self.f.close()


def read_lines(path):
    """Raw records from a results file; a truncated last line (crash mid-write) is ignored."""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _scan(path, arm, stream_id):
    done, attempts = {}, 0
    for r in read_lines(path):
        if r.get("arm") != arm or r.get("stream") != stream_id:
            continue
        if r["type"] == "cycle_start":
            attempts = max(attempts, r["attempt"])
        elif r["type"] == "cycle_done":
            done[r["cycle"]] = r["attempt"]
    return done, attempts


def _call(writer, arm_name, stream_id, cycle, what, fn, *args):
    """Run arm code inside the arm scope; convert leak attempts into ArmViolation."""
    try:
        with arm_scope():
            return fn(*args)
    except (AnswerLocked, SealedAccessError) as e:
        writer.write({"type": "violation", "arm": arm_name, "stream": stream_id, "cycle": cycle,
                      "where": what, "error": f"{type(e).__name__}: {e}"})
        writer.flush()
        raise ArmViolation(f"arm {arm_name!r} tried to reach locked data in {what}: {e}") from e


def _ckpt_path(ckpt_dir, name, stream_id, cycle):
    return Path(ckpt_dir) / f"{name}__{stream_id}__{cycle}.json"


def run_arm(arm, stream, canaries, out_path, *, arm_name=None, fault_hook=None, ckpt_dir=None):
    """Play `stream` through `arm`, appending to `out_path`. Safe to call again after a crash."""
    name = arm_name or arm.name
    sid = stream.stream_id
    done, attempts = _scan(out_path, name, sid)
    nxt = -1
    while nxt in done:
        nxt += 1
    w = _Writer(out_path)
    try:
        if hasattr(arm, "reset"):
            _call(w, name, sid, nxt, "reset", arm.reset)
        if hasattr(arm, "bind"):
            _call(w, name, sid, nxt, "bind", arm.bind, Services(stream, canaries))
        if nxt > -1:
            _restore(arm, stream, name, nxt - 1, ckpt_dir, w)
        canary_items = [(cs, cs.items()) for cs in canaries]

        for cycle in range(nxt, stream.n_cycles):
            attempts += 1
            at = attempts
            w.write({"type": "cycle_start", "arm": name, "stream": sid, "cycle": cycle,
                     "attempt": at})
            info = {}
            if cycle >= 0:
                n = 0
                for item in stream.train_items(cycle):
                    if fault_hook:
                        fault_hook(cycle, "day", n)
                    n += 1
                    _train_one(arm, stream, item, cycle, w, name, sid, at, record=True)
                if fault_hook:
                    fault_hook(cycle, "sleep", 0)
                info = _call(w, name, sid, cycle, "sleep", arm.sleep, cycle) or {}
                n = 0
                for item in stream.hidden_items(cycle):
                    if fault_hook:
                        fault_hook(cycle, "hidden", n)
                    n += 1
                    _test_one(arm, item, cycle, "hidden", w, name, sid, at)
            n = 0
            for cs, items in canary_items:
                for item in items:
                    if fault_hook:
                        fault_hook(cycle, "canary", n)
                    n += 1
                    _test_one(arm, item, cycle, "canary", w, name, sid, at)
            if ckpt_dir is not None and hasattr(arm, "save_state"):
                state = arm.save_state()
                if state is not None:
                    p = _ckpt_path(ckpt_dir, name, sid, cycle)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    tmp = p.with_suffix(".tmp")
                    tmp.write_text(json.dumps(state), encoding="utf-8")
                    tmp.replace(p)
            w.write({"type": "cycle_done", "arm": name, "stream": sid, "cycle": cycle,
                     "attempt": at, "info": info})
            w.flush()
        w.write({"type": "run_summary", "arm": name, "stream": sid,
                 "sealed_reads": stream.sealed.read_count})
        w.flush()
    finally:
        w.close()


def _train_one(arm, stream, item, cycle, w, name, sid, attempt, record):
    task = item.task("day", cycle)
    if item.kind == "teach":
        _call(w, name, sid, cycle, "observe", arm.observe, task, "", None, None)
        return
    assert item.split == "train", "only train items may reach observe()"
    ans = _call(w, name, sid, cycle, "answer", arm.answer, task)
    ans = ans if isinstance(ans, str) else str(ans)
    with grading_scope():
        g = grade(item, ans, cycle)
    if record:
        w.write({"type": "score", "arm": name, "stream": sid, "cycle": cycle, "attempt": attempt,
                 "phase": "day", "skill": item.skill, "family": item.family, "task_id": item.id,
                 "correct": int(g.correct), "stale": int(g.stale), "absorbed": int(g.absorbed)})
    corr = stream.correction_for(item)
    _call(w, name, sid, cycle, "observe", arm.observe, task, ans, bool(g.correct), corr)


def _test_one(arm, item, cycle, phase, w, name, sid, attempt, record=True):
    task = item.task("test", cycle)
    ans = _call(w, name, sid, cycle, "answer", arm.answer, task)
    ans = ans if isinstance(ans, str) else str(ans)
    with grading_scope():
        g = grade(item, ans, cycle)
    if record:
        w.write({"type": "score", "arm": name, "stream": sid, "cycle": cycle, "attempt": attempt,
                 "phase": phase, "skill": item.skill, "family": item.family, "task_id": item.id,
                 "correct": int(g.correct), "stale": int(g.stale), "absorbed": int(g.absorbed)})
    return g


def _restore(arm, stream, name, last_done, ckpt_dir, w):
    sid = stream.stream_id
    if last_done < 0:
        return
    if ckpt_dir is not None and hasattr(arm, "load_state"):
        p = _ckpt_path(ckpt_dir, name, sid, last_done)
        if p.exists():
            _call(w, name, sid, last_done, "load_state", arm.load_state,
                  json.loads(p.read_text(encoding="utf-8")))
            return
    for c in range(0, last_done + 1):  # replay, no scoring, no logging
        for item in stream.train_items(c):
            _train_one(arm, stream, item, c, w, name, sid, 0, record=False)
        _call(w, name, sid, c, "sleep", arm.sleep, c)


def run_sealed_milestone(arm, stream, out_path, *, milestone: str, reason: str, arm_name=None):
    """The only place the sealed set is read. Logged (stream.sealed.access_log, and the results file)."""
    name = arm_name or arm.name
    sid = stream.stream_id
    items = stream.sealed.read(milestone=milestone, reason=reason)
    w = _Writer(out_path)
    try:
        w.write({"type": "sealed_read", "arm": name, "stream": sid, "milestone": milestone,
                 "reason": reason, "n_items": len(items)})
        cycle = stream.n_cycles - 1
        for item in items:
            _test_one(arm, item, cycle, "sealed", w, name, sid, 0)
        w.flush()
    finally:
        w.close()
    return len(items)
