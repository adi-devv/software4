# v1 verdict

Valid seeds: [44, 45, 46]; excluded: [].

| Rule | ours | expert_only | rag | hippo_only | text_notes | no_memory | gold_context | ours_generic_only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rule:A | 97.2% | 97.2% | 71.1% | 71.1% | 0.0% | 0.0% | 93.9% | 71.1% |
| rule:B | 91.1% | 91.1% | 93.6% | 93.6% | 80.0% | 0.0% | 93.1% | 93.6% |
| rule:C | 91.9% | 91.9% | 96.4% | 96.4% | 94.7% | 0.0% | 94.4% | 96.4% |

Rule C is the composition test.
Claim 1 comparator: rag.
Context cost is measured on every hidden prompt of every cycle (the context block only, header included).

- ours - rag: 6.4% [4.1%, 8.8%], n=1080.
- ours - hippo_only: 6.4% [4.1%, 8.8%], n=1080.
- ours - no_memory: 93.4% [91.9%, 94.9%], n=1080.
- ours - ours_generic_only: 6.4% [4.1%, 8.8%], n=1080.

Mean context tokens: {'ours': 17.79268292682927, 'rag': 195.70957542908764}

Claim 1 (non-inferiority): PASS
Claim 2 (rule learning): PASS
Claim 3 (not format learning): FAIL
Claim 4 (cost): PASS
Claim 5 (retention and safety): FAIL

At least two valid seeds: PASS
Overall verdict: FAIL
