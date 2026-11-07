# Local sleep smoke (small)

Model: Qwen/Qwen3-0.6B; seed 0; fp16 base, fp32 rank-4 q/v LoRA; 2 optimizer steps, batch 4, lr 0.0002. Cycle 3: one step then finite random damage to the new adapter.

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
| ours | 1 | 0 | - | 6 | 8 |
| ours | 2 | 0 | 5 | 6 | 8 |
| ours | 3 | 0 | 5 | 0 | 7 |
| oracle | 0 | - | - | - | 8 |
| oracle | 1 | 0 | - | 6 | 8 |
| oracle | 2 | 0 | 6 | 6 | 8 |
| oracle | 3 | 0 | 6 | 6 | 8 |

| Cycle | Ablation | A correct / 8 | B correct / 8 |
|---:|---|---:|---:|
| 1 | on_no_hippo | 0 | - |
| 1 | off_no_hippo | 0 | - |
| 1 | off_with_hippo | 0 | - |
| 2 | on_no_hippo | 0 | 5 |
| 2 | off_no_hippo | 0 | 2 |
| 2 | off_with_hippo | 0 | 2 |
| 3 | on_no_hippo | 0 | 5 |
| 3 | off_no_hippo | 0 | 2 |
| 3 | off_with_hippo | 0 | 2 |

| Sleep | Gate | Wall seconds | Peak allocated MiB | Peak reserved MiB |
|---:|---|---:|---:|---:|
| 1 | accept | 10.7 | 1326.0 | 1378.0 |
| 2 | accept | 13.6 | 1327.8 | 1378.0 |
| 3 | accept | 8.3 | 1320.2 | 1378.0 |

Checklist: a: fail, b: fail, c: fail, d: fail, e: fail, f: pass, g: pass.

Total wall time: 310.3 seconds.

Gate uses the unchanged harness paired-noise test (alpha 0.05). No learning-gain threshold is used for accepting candidates. Eight items cannot establish broad retention or statistical significance. All scores use harness exact-answer grading.

Assumptions: hashing router and its default 0.2 threshold; 64-token context budget; 112 input tokens plus at most 16 target tokens. The same world is used in both runs. A and B are opaque rule names; instructions are supplied only as train corrections or oracle context. Cycle 3 has eight fresh arithmetic control examples. No hippocampus, existing arm, gate, or grading code was changed. Failed checks and complete evidence are in CHECKLIST.json and evidence.json.

- a: **fail**. A without hippocampus: 0.000; experts off/base: 0.000/0.000.
- b: **fail**. A without hippocampus after sleeps 1/2: 0.000/0.000; requires nonzero learned accuracy and no drop.
- c: **fail**. 6 routing errors across 82 test prompts.
- d: **fail**. Current revision recalled: True; obsolete original also recalled: True; quarantined poison recalled: 0.
- e: **fail**. Broken candidate accepted: True; post-sleep outputs identical: False. The unchanged gate rejects regressions beyond paired noise, not arbitrary bad weights.
- f: **pass**. Frozen fingerprint equal: True; uncached base outputs identical: True.
- g: **pass**. Nonfinite/skipped training steps: 0; split-audit violations: 0; forbidden training prompt overlap: 0.
