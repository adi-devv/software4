# Evaluation harness

Plain Python 3.13 plus numpy. No other dependencies. Run the tests from the repo root:

    python -m pytest harness/tests

## What it does

It builds invented worlds (made-up villages, guilds, rules), turns them into a stream of
16 daily cycles, plays an arm through the stream, grades every answer with code, and writes
one line per answer to a JSONL file. A report function reads that file.

Each cycle is one day (train tasks, the arm learns from each) and one night (`sleep`),
then the hidden tasks and the canaries are asked.

## Write an arm

An arm is any object with these methods:

    class MyArm:
        name = "ours"
        def answer(self, task):            # task.prompt is the question; return text
            return "reasoning...\nFinal answer: 42"
        def observe(self, task, answer, result, correction):
            ...                            # day tasks only: you may learn from these
        def sleep(self, cycle):            # once per cycle; return a small dict
            return {}
        # optional: reset(), bind(services), save_state() / load_state(state)

Rules the runner enforces: you only ever get `Task` objects (no answers); `observe` is called
on train tasks only; touching a locked answer, the grader, or the sealed set while your code
runs raises, the run stops, and a `violation` line is written. Answers must end with a
`Final answer: ...` line. Several candidates ("A or B") count as wrong.

`save_state()` / `load_state()` are optional. With them a resumed run restores a checkpoint;
without them the runner replays your `observe`/`sleep` calls, which needs a deterministic arm.

## Run a stream

    from harness.streams import synthetic_stream
    from harness.canaries import SyntheticCanarySet
    from harness.runner import run_arm

    stream = synthetic_stream(seed=1)
    canaries = [SyntheticCanarySet(seed=0, n=300)]     # or JsonlCanarySet("gsm8k", "gsm8k.jsonl")
    run_arm(MyArm(), stream, canaries, "results.jsonl", ckpt_dir="ckpt")

Use at least 6 streams (different seeds) per arm; the success check cannot pass with fewer than 3.
Calling `run_arm` again after a crash picks up at the first unfinished cycle.
Your own task files load with `JsonlSource` (see `harness/streams.py` for the line format).

The sealed set is read only through `run_sealed_milestone(arm, stream, path, milestone=...,
reason=...)`. Any other access raises. The report prints how many times it was read.

## Read the report

    from harness.report import format_report
    print(format_report("results.jsonl"))

- Hidden score by cycle: percent right on held-out tasks of every skill seen so far.
  A rising line is learning; a falling line for old skills is forgetting.
- Canary curve: the same old-skill questions every cycle. Column -1 is before learning.
  The canary drop is baseline minus the worst later value.
- Poisoned claims absorbed: how often the arm repeated a planted false claim.
- Success check (PLAN.md): ours beats the best baseline (arm 10 must be included) by at
  least 3 points, the 95% interval over streams has a lower end above 0, and the canary drop
  is at most 2 points. The interval is built from per-stream differences. The naive
  per-task interval printed beside it is too narrow, because tasks in a stream are not
  independent.

`harness/stats.py` has the power helpers (`plan_figure_check()`: about 1.7k paired tasks for
3 points holds at roughly 20% disagreement between arms; far more if samples are unpaired or
clustered). `harness/gate.py` gives arms a one-shot gate (`Services.gate_session`) with a
rollback margin measured from paired noise.

`harness/arms_fake.py` holds test-only arms (oracle, forgetter, noisy, cheater).
