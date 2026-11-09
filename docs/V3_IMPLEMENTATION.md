# V3 implementation notes (unfrozen)

All changes are opt-in. No benchmark/control run, freeze, commit or push is part
of this patch. `V3_PREREG_DRAFT.md` and `step4/v2_analyze.py` remain untouched.

| Config field (`SleepConfig` / runner config) | Runner environment | Effect |
|---|---|---|
| `v3_fact_filter` | `S4_V3_FACT_FILTER=1` | On `ours` / `expert_only`, filter the existing top-k to facts only when its first hit is a fact. No overfetch. |
| `v3_fallback` | `S4_V3_FALLBACK=1` | On `ours` / `expert_only`, compare expert output shape with answer targets in raw v2 recall (`FALLBACK_K = 2`); retry once on mismatch. |
| `v3_wrong_rule` | `S4_V3_WRONG_RULE=1` | Make `wrong_rule` available for explicit arm selection. Does not add it to default arm lists or schedule it. |

Flags default to false. Disabled flags and fallback statistics are omitted from
checkpoints; normal answer prompts, traces and logging rows retain v2's shape.
Runner source digests necessarily change with source edits.

Fallback retries use the same backend **and the same active expert**. The trace
stores both prompts and the answer log charges both context blocks, including
the first empty-context block, to the existing `context_tokens` field. The final
prompt/retrieved IDs describe the returned answer. `result.json` exposes
`stats.fallback_fired`; the counter survives checkpoints and resets with the arm.
Sleep training and gate probes keep their existing direct generation flow.

Implementation choices for details not fixed in the draft:

- Raw recall is called once with k=2, then entries without usable verified/correct
  answers are ignored. No usable demos means keep the original expert answer.
  One usable demo means compare/retry with one. The normal context budget still
  limits which whole demos fit the retry prompt.
- `wrong_rule` uses a fixed seed (0), sorted unique answer targets, and a shuffled
  cyclic derangement within each shape-signature group of the selected training
  batch. Thus no target maps to itself and answer formats stay intact. A singleton
  group changes its first letter/digit within the same case/digit class. Facts
  keep their true targets; prompts are unchanged. This is a batch target shuffle,
  not a newly defined benchmark rule or a global permutation across nights.
- The control uses `ExpertOnly`'s empty-context prompts and `ForcedSleepArm`'s
  diagnostic gates, since true-target checks would reject intentionally wrong
  adapters. Runner construction disables fact filtering/fallback for controls.
- `control_validity()` evaluates rule rows for `wrong_rule` by default. It accepts
  target text in `correct_answer`, `expected` or a textual `correct`, or a separate
  expected-target mapping keyed by task ID / (day, task ID). Raw `answers.jsonl`
  does not contain targets. Missing targets raise; missing answers count as
  malformed. Empty evidence yields `valid=False`, rate `None`. A boolean
  correctness field never establishes format validity.

CPU coverage is in `baselines/tests/test_v3.py` and
`step4/tests/test_v3_wiring.py`. These inspect the control's training examples
without training or evaluating a `wrong_rule` adapter. No GPU, model download or
confirmation-seed experiment is required.

V3 confirmation-seed orchestration and claim-report integration are outside
changes 1–3 in this patch: the runner's existing v1/v2 seed dispatch and the
frozen v2 analyzer's seed validation still need a separate v3 integration before
confirmation runs. The validity helper does not alter Claim 3's frozen analyzer.

Validation uses the requested Python 3.12 executable. Its global Transformers
4.56.2 lacks `Qwen3_5TextConfig`, so the unchanged hybrid CPU test cannot run in
that environment. An ignored, workspace-local overlay at `.test-runs/v3-deps`
contains the repository's pinned Transformers 5.3.0 and Hugging Face Hub 1.33.0.
Setting `PYTHONPATH` to that directory selects it without changing global
packages. Test runs also set `CUDA_VISIBLE_DEVICES` empty, hub/model access
offline, and temp/cache directories under `.test-runs` inside this worktree.

Final CPU suite (`python -m pytest baselines step4 harness -q -x`, with the
workspace overlay): **581 passed, 1 skipped**, in 305.49 seconds. All **33 new
v3 tests** pass. The existing skip is `test_v1_capture.py:398`, which requires
the original registered v1 model, versioned stacks and emitted-token reference
logs. No existing test was edited.

## Review fixes

The integration limitation above is resolved by `step4/v3_report.py`. A v3 flag
or step5 seed 53-57 selects it; flags-off v1/v2 paths retain their behavior.
The wrapper grades the real seed's registered world, then translates only seed
registration/control labels to reuse the frozen v2 verdict and evidence tables.
Claims 1, 2, 4 and 5 are unchanged. Claim 3 compares `expert_only` with
`wrong_rule`, keeping the 30-point margin, strict CI low > 0, matched-training
checks, and the frozen 10,000 paired draws with `default_rng(0)`.

Control validity now requires an explicit full 360-rule-item target mapping and
exactly one row per item at cycle 2. Missing/duplicate/unknown rows fail closed;
missing answers and conflicting final-answer lines count as malformed. Each
valid seed must meet 90% well-formed answers or Claim 3 is UNINFORMATIVE.
Fallback checks also use strict generated-answer extraction; memory targets
retain their existing parser. The wrong-rule training test first requires three
selected examples, including two rule examples.

Reports enforce no subset verdicts, stop on invalid seeds, request void-seed
replacements 56 then 57, and annotate verdicts with fewer than three valid seeds.
They include per-seed/arm row audits for all hidden cycles (including zeros),
`fallback_fired`, and context cost, also appended to each v3 run's `RESULTS.md`.
Aggregate reports use `python -m step4.v3_report --runs <all run directories>
--out <report.md>`. Per-run reports remain pending until the full seed set is
supplied. Resume pruning now covers v3 seeds and flags while retaining genuine
duplicates in completed cycles. The report is included in packaging/digests.

Review validation (Python 3.12, Transformers 5.3.0 overlay on `PYTHONPATH`,
CUDA disabled, offline): `python -m pytest baselines step4 harness -q` completed
with **630 passed, 1 skipped** in 386.95 seconds. The existing reference-stack
skip remains. After the final report/failed-worker checks, the focused v3 suite
(`baselines/tests/test_v3.py`, `step4/tests/test_v3_wiring.py`,
`step4/tests/test_v3_report.py`) passed **83 tests** in 40.77 seconds. No Kaggle,
GPU, model download, commit or push was performed; frozen files were not edited.


## Design fixes 2

These decisions supersede the fallback and signature details above. The fallback
flag applies only to `ours`; `expert_only` and `wrong_rule` never retry, and their
retriever metadata remains disabled. Fact filtering remains available for `ours`
and `expert_only`.

The trigger still compares the strictly extracted expert answer with usable
targets from raw v2 recall using `FALLBACK_TRIGGER_K = 2`. On mismatch, the retry
uses the frozen core without an expert and the exact HippoOnly prompt path: raw
v2 recall with the arm's k, no fact filter, and the same context budget. The
trace and token accounting include both prompts; retrieved IDs and pending
feedback correspond to the retry.

Shape signatures now erase letter case, matching case-folded scoring. Wrong-rule
targets are grouped by that signature and deranged across distinct case-folded
values. If a group has only one such value, its first alphanumeric character is
changed within its class, ensuring every target differs after case folding.

CPU validation with the requested Python 3.12 dependency overlay, CUDA disabled
and offline: `python -m pytest baselines step4 harness -q` completed with **631
passed, 1 skipped** in 411.86 seconds. One additional retry-context test added
after suite collection passed separately. The skip is the existing v1 reference
stack test.

## Design fixes 3

Before invoking the frozen analyzer, every supplied seed must contain every
registered V3 arm with an explicit complete status. Missing or unfinished
workers stop the Addendum-1 procedure without claim verdicts, including for
seeds whose validity gates otherwise pass. Rule-answer miss classification now
case-folds extracted and expected text before comparing shapes, letters and
suffixes, consistent with scoring.

CPU validation with the requested Python 3.12 dependency overlay, CUDA disabled
and Hugging Face offline: `python -m pytest baselines step4 harness -q` completed
with **634 passed, 1 skipped** in 346.42 seconds. The skip is the existing v1
reference stack test.
