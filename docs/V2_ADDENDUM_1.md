# V2 pre-registration, Addendum 1 (procedure only, stricter-only)

Written: 2026-11-09 10:47 UTC, before any v2 seed (48, 49, 50) has finished.
Applies to: `docs/V2_PREREG_DRAFT.md` (SHA-256 `4420f00642a5452e3dc96eff752a245e9c54fa1c864d7c61828d44f347f8fa90`) and source digest `98eb1f1f945dc19beb4e53c954f6c76228a131d11cb30023405efb0155c65d42`.
This file lives in `docs/`, outside the source digest. It changes no claim, threshold, arm, seed list or analyzer code. It only adds hand-checks the analyzer does not enforce.

1. **No subset verdicts.** Run `step4/v2_analyze.py` only after seeds 48, 49 and 50 have all finished and all are supplied. No verdict from a subset, and no peeking at partial outputs.
2. **Void seed (gate-failed).** Run seed 51, then 52 if still needed, before any verdict. If both are used up and fewer than 2 valid seeds remain, report that plainly.
3. **Invalid seed, or a kernel that dies with no usable results.** The frozen plan gives no replacement or rerun. Stop and tell the strategy session and the user before reading any verdict. A rerun would be a declared deviation and is the user's call.
4. **Duplicate or missing answer rows outside cycle 2.** Report them per seed and per arm in `step4/RESULTS.md`, even if zero. (The analyzer only treats cycle 2 as an evidence error.)
5. **Fewer than 3 valid seeds.** Any verdict resting on fewer than 3 valid seeds must say so next to PASS/FAIL, naming the excluded seed and why.

Known analyzer gaps, accepted and not patched: a corrupt `results.json` crashes with a raw error (acceptable, fails loudly); an invalid or never-supplied seed is not flagged by the analyzer, so items 1 and 3 above are checked by hand.
