"""The arm interface and the eleven PLAN.md arm names.

An arm is any object with:

    name: str
    answer(task: Task) -> str                      # called for day tasks AND frozen test tasks
    observe(task, answer, result, correction)      # day tasks only (never hidden/gate/sealed)
    sleep(cycle: int) -> dict                      # called once at the end of each cycle (JSON-able info)

and optionally:

    reset()                       # called at the start of every stream
    bind(services)                # receives `Services` (gate sessions); see runner.py
    save_state() -> dict / load_state(dict)   # JSON-able checkpoint, used to resume cheaply

`task.mode` is "day" (you may learn from it afterwards) or "test" (frozen: do not log it as an
episode, do not train on it). `task.kind == "teach"` tasks are statements; the runner calls only
`observe(task, "", None, None)` for them and does not call `answer`.
`result` is True/False (the arm's answer was graded by code) or None for teach tasks.
`correction` is a string for train probes ("Correct answer: ..." plus a rule note) else None.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .tasks import Task

ARM_NAMES = {
    1: "no_memory", 2: "rag", 3: "long_context", 4: "text_notes", 5: "ours",
    6: "ours_sparse_memory", 7: "ours_no_dream", 8: "ours_generic_data", 9: "oracle",
    10: "hippo_only", 11: "ours_merge",
}
# Baselines for the success rule (PLAN.md): arms 1-4 and 10. The oracle (9) is an upper bound and
# arms 5-8, 11 are variants of ours.
PLAN_BASELINES = ("no_memory", "rag", "long_context", "text_notes", "hippo_only")
OURS = "ours"


@runtime_checkable
class Arm(Protocol):
    name: str

    def answer(self, task: Task) -> str: ...

    def observe(self, task: Task, answer: str, result, correction) -> None: ...

    def sleep(self, cycle: int) -> dict: ...
