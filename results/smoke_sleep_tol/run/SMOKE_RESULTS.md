# Local sleep smoke (full)

Command: `python scripts/smoke_sleep.py --steps 64 --lr 0.0004`

Model: Qwen/Qwen3-0.6B; seed 0; fp16 base, fp32 rank-4 q/v LoRA; 64 optimizer steps, batch 4, lr 0.0004. Cycle 3: one step then finite random damage to the new adapter.

| Arm | Cycle | A correct / 8 | B correct / 8 | Facts correct / 6 | Canaries correct / 8 |
|---|---:|---:|---:|---:|---:|
| no_memory | 0 | - | - | - | 8 |
| no_memory | 1 | 0 | - | 0 | 8 |
| no_memory | 2 | 0 | 2 | 0 | 8 |
| no_memory | 3 | 0 | 2 | 0 | 8 |
| hippo_only | 0 | - | - | - | 8 |
| hippo_only | 1 | 0 | - | 6 | 8 |
| hippo_only | 2 | 0 | 2 | 6 | 8 |
| hippo_only | 3 | 0 | 2 | 6 | 8 |
| ours | 0 | - | - | - | 8 |
| ours | 1 | 5 | - | 6 | 8 |
| ours | 2 | 5 | 8 | 6 | 8 |
| ours | 3 | 5 | 8 | 6 | 7 |
| oracle | 0 | - | - | - | 8 |
| oracle | 1 | 0 | - | 6 | 8 |
| oracle | 2 | 0 | 6 | 6 | 8 |
| oracle | 3 | 0 | 6 | 6 | 8 |

| Cycle | Ablation | A correct / 8 | B correct / 8 |
|---:|---|---:|---:|
| 1 | on_no_hippo | 5 | - |
| 1 | off_no_hippo | 0 | - |
| 1 | off_with_hippo | 0 | - |
| 2 | on_no_hippo | 5 | 8 |
| 2 | off_no_hippo | 0 | 2 |
| 2 | off_with_hippo | 0 | 2 |
| 3 | on_no_hippo | 5 | 8 |
| 3 | off_no_hippo | 0 | 2 |
| 3 | off_with_hippo | 0 | 2 |

| Sleep | Gate | Wall seconds | Peak allocated MiB | Peak reserved MiB |
|---:|---|---:|---:|---:|
| 1 | accept | 191.7 | 1327.8 | 1380.0 |
| 2 | accept | 196.8 | 1336.6 | 1476.0 |
| 3 | reject | 2.9 | 1320.2 | 1476.0 |

Checklist: a: pass, b: pass, c: pass, d: pass, e: pass, f: pass, g: pass.

Total wall time: 669.5 seconds.

The arm first caps the effective adapter delta at the frozen projection norm and requires zero canary output changes, no per-example regression on forced trusted train/replay validation, and independent gate/canary route coverage. The norm cap alone does not establish behavioral safety. Gate uses the unchanged harness paired-noise test (alpha 0.05). No learning-gain threshold is used for accepting candidates. Eight items cannot establish broad retention or statistical significance. All scores use harness exact-answer grading.

Assumptions: hashing router and its default 0.2 threshold; 64-token context budget; 112 input tokens plus at most 16 target tokens. The same world is used in both runs. A and B are opaque rule names; instructions are supplied only as train corrections or oracle context. Cycle 3 has eight fresh arithmetic control examples. Trusted fact update notices use hippocampus versioning; the harness gate, grading, router threshold, and rule definitions are unchanged. Failed checks and complete evidence are in CHECKLIST.json and evidence.json.

- a: **pass**. A without hippocampus: 0.625; experts off/base: 0.000/0.000.
- b: **pass**. A without hippocampus after sleeps 1/2: 0.625/0.625; requires nonzero learned accuracy and no drop.
- c: **pass**. 0 routing errors across 82 test prompts.
- d: **pass**. Current revision recalled: True; obsolete original also recalled: False; quarantined poison recalled: 0.
- e: **pass**. Broken candidate accepted: False; post-sleep outputs identical: True. The arm checks scale, forced trusted validation, route coverage and exact canary preservation before the unchanged paired-noise gate.
- f: **pass**. Frozen fingerprint equal: True; uncached base outputs identical: True.
- g: **pass**. Nonfinite/skipped training steps: 0; split-audit violations: 0; forbidden training prompt overlap: 0.
