"""Turn a results JSONL into a markdown report."""
from __future__ import annotations

import numpy as np

from .arms import OURS, PLAN_BASELINES
from .results import Results, common_streams, success_check


def _fmt(x):
    return "  - " if x != x else f"{100 * x:5.1f}"


def format_report(path, *, ours: str = OURS, baselines=None, require=PLAN_BASELINES,
                  n_cycles=15, last_k: int = 1) -> str:
    res = Results.from_file(path)
    arms = res.arms()
    out = ["# Evaluation report", ""]
    if not arms:
        return "# Evaluation report\n\nNo completed cycles in the results file.\n"
    n_streams = {a: len(res.streams(a)) for a in arms}
    out.append("Streams per arm: " + ", ".join(f"{a}={n}" for a, n in n_streams.items()))
    out.append("")
    max_cycle = max((c for k, cs in res.done.items() for c in cs), default=-1)
    cycles = list(range(0, max_cycle + 1))

    out.append("## Hidden-test score by cycle (percent, mean over streams)")
    out.append("")
    out.append("| arm | " + " | ".join(str(c) for c in cycles) + " |")
    out.append("|---|" + "---|" * len(cycles))
    for a in arms:
        row = []
        for c in cycles:
            v = [res.hidden_acc(a, s, c) for s in res.streams(a)]
            v = [x for x in v if x == x]
            row.append(_fmt(float(np.mean(v)) if v else float("nan")))
        out.append(f"| {a} | " + " | ".join(row) + " |")
    out.append("")

    out.append("## Canary curve (percent, mean over streams; column -1 is before learning)")
    out.append("")
    cc = [-1] + cycles
    out.append("| arm | " + " | ".join(str(c) for c in cc) + " |")
    out.append("|---|" + "---|" * len(cc))
    for a in arms:
        row = []
        for c in cc:
            v = [res.canary_acc(a, s, c) for s in res.streams(a)]
            v = [x for x in v if x == x]
            row.append(_fmt(float(np.mean(v)) if v else float("nan")))
        out.append(f"| {a} | " + " | ".join(row) + " |")
    out.append("")

    out.append("## Poisoned claims absorbed (final cycle: absorbed / probes, summed over streams)")
    out.append("")
    for a in arms:
        ab = tot = 0
        for s in res.streams(a):
            lc = res.last_cycle(a, s)
            x, y = res.poison(a, s, lc)
            ab, tot = ab + x, tot + y
        out.append(f"- {a}: {ab} / {tot}")
    out.append("")

    out.append("## Success check (PLAN.md)")
    out.append("")
    if ours not in arms:
        out.append(f"Arm {ours!r} not in results; no check.")
    else:
        sr = success_check(res, ours=ours, baselines=baselines, require=require,
                           n_cycles=n_cycles, last_k=last_k)
        out.append("```")
        out.append(sr.summary())
        if sr.ci is not None and sr.ci.ok:
            out.append(f"  stream-level Bonferroni-adjusted CI of ours - best baseline: {sr.ci}")
        elif sr.ci is not None:
            out.append(f"  stream-level interval: not available (n={sr.ci.n})")
        if sr.naive_ci is not None and sr.naive_ci.ok:
            out.append(f"  naive task-level CI (NOT valid, tasks are clustered): {sr.naive_ci}")
        out.append("```")
    out.append("")
    out.append("## Sealed set")
    out.append("")
    out.append(f"Sealed reads: {res.sealed_reads}")
    for e in res.sealed_events:
        out.append(f"- {e['arm']} / {e['stream']} / {e['milestone']}: {e['reason']}")
    if res.violations:
        out.append("")
        out.append("## Violations")
        for v in res.violations:
            out.append(f"- {v['arm']} / {v['stream']} cycle {v['cycle']} in {v['where']}: {v['error']}")
    out.append("")
    return "\n".join(out)
