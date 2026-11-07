"""Offline retrieval comparison. Run: python scripts/retrieval_check.py --seed 0 1 2."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

# Bound CPU threading before numpy is imported; no GPU/model libraries are used.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from baselines import Rag, TextNotes
from harness.worlds import NameFactory
from hippocampus import HashingEmbedder, Hippocampus

SYSTEMS = ("hippocampus", "rag", "text_notes")
STRUCTURAL = ("revised", "multihop", "distractors", "poison")
METRICS = ("recall@1", "recall@5", "MRR", "MRR@5", "all_gold@1", "all_gold@5",
           "unsafe@1", "unsafe@5")


@dataclass(frozen=True)
class Fact:
    memory_id: str
    text: str
    version: int = 1
    status: str = "trusted"

    @property
    def id(self):
        return f"{self.memory_id}:v{self.version}"


@dataclass(frozen=True)
class Query:
    id: str
    text: str
    gold: tuple[str, ...]


@dataclass(frozen=True)
class Case:
    category: str
    facts: tuple[Fact, ...]
    queries: tuple[Query, ...]


def generate(seed=0, n=40, scales=(100, 1000, 10000)):
    """Independent category corpora; nested scale corpora use identical queries."""
    if n < 1 or not scales or min(scales) < n:
        raise ValueError("need 1 <= queries <= smallest scale")
    rng = random.Random(seed)
    names = NameFactory(np.random.default_rng(seed))
    cases = []
    for category in ("plain", *STRUCTURAL):
        facts, queries = [], []
        for i in range(n):
            entity, value = names.place(), names.person()
            mid = f"{category}-{i}"
            text = f"Field note: the leader of {entity} is {value}."
            facts.append(Fact(mid, text))
            gold = (facts[-1].id,)
            question = f"Who is the leader of {entity}?"
            if category == "revised":
                # Apply all original statements before the later updates below.
                facts.append(Fact(mid, f"Update notice: the leader of {entity} is now {names.person()}.", 2))
                gold = (facts[-1].id,)
            elif category == "multihop":
                ally = names.place()
                facts[-1] = Fact(mid, f"Field note: the ally of {entity} is {ally}.")
                facts.append(Fact(mid + "-hop", f"Field note: the leader of {ally} is {value}."))
                gold = (facts[-2].id, facts[-1].id)
                question = f"Who is the leader of the ally of {entity}?"
            elif category == "distractors":
                # Ten very similar but distinct entity identifiers per group.
                target = rng.randrange(10)
                facts.pop()
                for j in range(10):
                    f = Fact(f"{mid}-{j}", f"Field note: the leader of {entity}_{j:02d} is {names.person()}.")
                    facts.append(f)
                    if j == target:
                        gold = (f.id,)
                        question = f"Who is the leader of {entity}_{j:02d}?"
            elif category == "poison":
                # Rotate explicit quarantine, untrusted source, and revocation.
                status = ("quarantined", "untrusted", "tombstoned")[i % 3]
                facts.append(Fact(mid + "-poison", f"The leader of {entity} is {names.person()}.", status=status))
            queries.append(Query(f"{category}-q{i}", question, gold))
        # Random insertion order independent of query order; updates stay later.
        original = [f for f in facts if f.version == 1]
        updates = [f for f in facts if f.version > 1]
        rng.shuffle(original)
        rng.shuffle(updates)
        rng.shuffle(queries)
        cases.append(Case(category, tuple(original + updates), tuple(queries)))

    facts = [Fact(f"scale-{i}", f"Field note: the leader of {names.place()} is {names.person()}.")
             for i in range(max(scales))]
    rng.shuffle(facts)
    targets = rng.sample(facts[:min(scales)], n)
    queries = tuple(Query(f"scale-q{i}", f"Who is the leader of {f.text.split(' of ', 1)[1].split(' is ', 1)[0]}?", (f.id,))
                    for i, f in enumerate(targets))
    cases.extend(Case(f"scale_{size}", tuple(facts[:size]), queries) for size in scales)
    return cases


class NoLanguageModel:
    """Fail closed if any arm tries to use a language model, even a fake one."""
    def __getattr__(self, name):
        raise AssertionError(f"language model access forbidden: {name}")


def prepare(case):
    embedder = HashingEmbedder()  # Same instance and default settings for vectors.
    hippo = Hippocampus(embedder=embedder)
    rag = Rag(llm=NoLanguageModel(), embedder=embedder)
    notes = TextNotes(llm=NoLanguageModel())
    # Public state APIs give both arms the same verbatim entries, including stale
    # and unsafe ones. No generation, note cap, pruning, or answer feedback.
    rag.load_state({"pending": {}, "episodes": [{"text": f.text} for f in case.facts]})
    notes.load_state({"pending": {}, "episodes": [], "notes": [
        {"text": f.text, "uses": 0, "successes": 0} for f in case.facts]})
    for tick, f in enumerate(case.facts, 1):
        if f.version > 1:
            hippo.update(f.memory_id, f.text, t=tick)
        else:
            hippo.add(f.text, id=f.memory_id, t=tick,
                      source="forum" if f.status == "untrusted" else "user",
                      quarantine=f.status == "quarantined")
    for f in case.facts:
        if f.status == "tombstoned":
            hippo.tombstone(f.memory_id, t=len(case.facts) + 1)
    # No nap, plastic training, explicit entities, or forced deep recall.
    def recall(query, k):
        return [f"{h.id}:v{h.version}" for h in hippo.recall(query, k=k, now=len(case.facts) + 2)]

    return {"hippocampus": recall,
            "rag": lambda q, k: [case.facts[i].id for i, _ in rag.retrieve(q, k)],
            "text_notes": lambda q, k: [case.facts[i].id for i, _ in notes.retrieve(q, k)]}


def score(gold, ranked, forbidden=(), k=5):
    """Fractional recall, first-relevant reciprocal rank, all-gold, unsafe flag.

    Any forbidden hit in this cutoff zeroes all success metrics. Duplicate ids
    never increase recall. MRR is explicitly truncated at the retrieval cutoff.
    """
    gold = set(gold)
    if not gold:
        raise ValueError("query has no gold fact")
    top = ranked[:k]
    unsafe = bool(set(top) & set(forbidden))
    found = gold & set(top)
    rr = next((1 / rank for rank, mid in enumerate(top, 1) if mid in gold), 0.0)
    return {"recall": 0.0 if unsafe else len(found) / len(gold),
            "rr": 0.0 if unsafe else rr,
            "all_gold": float(not unsafe and found == gold), "unsafe": float(unsafe)}


def measure(case, seed):
    retrievers = prepare(case)
    forbidden = {f.id for f in case.facts if f.status != "trusted"}
    rows = []
    for query in case.queries:
        # Rotate system order to reduce systematic latency ordering effects.
        offset = int(query.id.rsplit("q", 1)[1]) % len(SYSTEMS)
        for system in SYSTEMS[offset:] + SYSTEMS[:offset]:
            row = {"seed": seed, "category": case.category, "query_id": query.id,
                   "system": system, "gold": list(query.gold)}
            for k in (1, 5):
                start = time.perf_counter_ns()
                ranked = retrievers[system](query.text, k)
                row[f"latency_ms@{k}"] = (time.perf_counter_ns() - start) / 1e6
                result = score(query.gold, ranked, forbidden, k)
                row[f"retrieved@{k}"] = ranked
                for name in ("recall", "all_gold", "unsafe"):
                    row[f"{name}@{k}"] = result[name]
                if k == 5:
                    row["MRR@5"] = result["rr"]
            # Request the same full corpus size from every system for MRR.
            # Native fast recall may return fewer candidates; missing gold is 0.
            ranked = retrievers[system](query.text, len(case.facts))
            row["MRR"] = score(query.gold, ranked, forbidden, len(case.facts))["rr"]
            row["full_ranking"] = {
                "requested_k": len(case.facts), "returned": len(ranked),
                "first_gold_rank": next((i for i, mid in enumerate(ranked, 1) if mid in query.gold), None),
                "unsafe_count": len(set(ranked) & forbidden),
                "sha256": hashlib.sha256(json.dumps(ranked).encode()).hexdigest()}
            rows.append(row)
    return rows


def summarize(rows, seeds, categories):
    output = []
    for category in [*categories, "overall"]:
        for system in SYSTEMS:
            selected = [r for r in rows if r["system"] == system and
                        (category == "overall" or r["category"] == category)]
            per_seed = []
            for seed in seeds:
                subset = [r for r in selected if r["seed"] == seed]
                per_seed.append({"seed": seed, **{m: statistics.mean(r[m] for r in subset) for m in METRICS},
                                 **{f"latency_ms@{k}": statistics.median(r[f"latency_ms@{k}"] for r in subset)
                                    for k in (1, 5)}})
            metrics = {m: {"mean": statistics.mean(s[m] for s in per_seed),
                           "min": min(s[m] for s in per_seed), "max": max(s[m] for s in per_seed)}
                       for m in (*METRICS, "latency_ms@1", "latency_ms@5")}
            output.append({"category": category, "system": system,
                           "queries_per_seed": len(selected) // len(seeds), "metrics": metrics,
                           "median_latency_ms@5": statistics.median(r["latency_ms@5"] for r in selected),
                           "per_seed": per_seed})
    return output


def gate(summary):
    """Strict improvements on each structural category, in mean and every seed."""
    checks = []
    for category in STRUCTURAL:
        h, r = (next(s for s in summary if s["category"] == category and s["system"] == system)
                for system in ("hippocampus", "rag"))
        for metric in ("recall@1", "recall@5"):
            checks.append({"category": category, "metric": metric,
                           "mean_passed": h["metrics"][metric]["mean"] > r["metrics"][metric]["mean"],
                           "every_seed_passed": all(a[metric] > b[metric] for a, b in zip(h["per_seed"], r["per_seed"]))})
    return {"passed": all(c["mean_passed"] and c["every_seed_passed"] for c in checks), "checks": checks}


def table(summary):
    lines = ["| Category | System | Recall@1 | Recall@5 | MRR | All gold@5 | Median ms@5 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for s in summary:
        cells = []
        for m in ("recall@1", "recall@5", "MRR", "all_gold@5"):
            v = s["metrics"][m]
            cells.append(f"{v['mean']:.3f} [{v['min']:.3f}, {v['max']:.3f}]")
        lines.append(f"| {s['category']} | {s['system']} | " + " | ".join(cells) + f" | {s['median_latency_ms@5']:.3f} |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, nargs="+", default=[0, 1, 2], help="one or more seeds (default: 0 1 2)")
    parser.add_argument("--queries", type=int, default=40, help="queries per category per seed (default: 40)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)
    if len(set(args.seed)) != len(args.seed) or not 1 <= args.queries <= 100:
        parser.error("seeds must be distinct and queries must be between 1 and 100")
    rows, datasets = [], []
    for seed in args.seed:
        cases = generate(seed, args.queries)
        payload = [asdict(c) for c in cases]
        datasets.append({"seed": seed, "sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
                         "cases": payload})
        for case in cases:
            print(f"seed={seed} {case.category}: {len(case.facts)} fact versions, {len(case.queries)} queries", flush=True)
            rows.extend(measure(case, seed))
    summary = summarize(rows, args.seed, [c.category for c in cases])
    decision = gate(summary)
    rendered = table(summary)
    means = {(s["category"], s["system"]): {m: v["mean"] for m, v in s["metrics"].items()}
             for s in summary}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {"seeds": args.seed, "queries_per_category": args.queries,
              "ks": [1, 5], "mrr_k": "number of stored fact versions (same request for all systems)",
              "embedder": "HashingEmbedder(dim=256, ngram=3, ngram_weight=0.3)",
              "python": sys.version, "numpy": np.__version__, "platform": sys.platform}
    (args.output_dir / "retrieval_check.json").write_text(json.dumps(
        {"config": config, "gate": decision, "summary": summary, "queries": rows, "datasets": datasets}, indent=2) + "\n", encoding="utf-8")
    report = f"""# Retrieval-only results

Gate: **{'PASS' if decision['passed'] else 'FAIL'}**. Strictly higher hippocampus recall@1 AND recall@5 than
RAG in each of revised, multihop, distractors, and poison, both in the seed mean and every seed.
Ties fail. This gate was specified before measuring; no parameters were tuned.

Command: `python scripts/retrieval_check.py --seed {' '.join(map(str, args.seed))} --queries {args.queries}`

{len(args.seed)} seeds; {args.queries * len(cases)} queries/seed; {len(rows) // 3} total queries.
Cells show mean [min, max] across seeds. Latency is the pooled median per system/category,
in milliseconds at k=5, including query embedding and RAG's native corpus re-embedding,
excluding ingestion. JSON also includes per-seed median latency at k=1 and k=5 and its mean/range.
First queries are included (no warmup); CPU wall times are not deterministic.
Calls run in k=1, k=5, then full-ranking order per system/query; graph caches may
already be warm at k=5. Systems rotate order across queries.

{rendered}

## Findings

- The strict structural gate {'passed' if decision['passed'] else 'failed'}; overall recall alone does not establish it.
- Multihop both-gold@5: hippocampus {means['multihop', 'hippocampus']['all_gold@5']:.3f},
  RAG {means['multihop', 'rag']['all_gold@5']:.3f}. Their recall@1 values are
  {means['multihop', 'hippocampus']['recall@1']:.3f} and {means['multihop', 'rag']['recall@1']:.3f},
  respectively (the ceiling is 0.5). MRR can be perfect while the second fact is missing.
- Distractor recall@5: hippocampus {means['distractors', 'hippocampus']['recall@5']:.3f},
  RAG {means['distractors', 'rag']['recall@5']:.3f}. Graph structure provides no universal advantage here.
- TextNotes scores reflect native query-independent ordering over verbatim notes, not model quality.

## Protocol and limitations

- CPU only, offline, numpy/stdlib, no model objects or downloads. An inert sentinel raises on any
  language-model access. Neither `answer()` nor `sleep()` is called.
- Default `HashingEmbedder` shared by Hippocampus and RAG. TextNotes uses no embeddings:
  its existing ranking is uses descending then newest first, independent of the query.
  Adding vector ranking to TextNotes would change the arm's behavior, so it is preserved.
- Added small read-only `Rag.retrieve(query, k)` and `TextNotes.retrieve(query, k)` helpers;
  `context()` delegates to them without changing ordering or limits.
- Both baseline corpora are loaded via public `load_state()` with the identical verbatim fact
  versions; TextNotes receives oracle verbatim notes, all uses=0. Model distillation, the default
  200-note cap, pruning, and prompt token budgets are deliberately bypassed. This measures native
  note retrieval from a shared corpus, not end-to-end TextNotes learning or note quality.
- Hippocampus uses public `add/update/tombstone/recall`, default extractor/config and normal fast/deep
  dispatch. No explicit entities, gold-derived links, outcomes, naps, or forced-deep queries.
  Update identities and quarantine/revocation metadata are supplied as structured events.
  RAG/notes retain all text versions because they have no corresponding lifecycle support.
  This tests Hippocampus core lifecycle APIs with explicit identities. Ordinary `add()` also
  versions trusted update notices in the controlled fact grammar; this benchmark uses explicit
  `update()` calls, so that additional ingestion path does not affect its measurements.
- Fictional names follow `harness.worlds.NameFactory`; fact/update wording and ally chains follow
  the harness pattern. Categories have separate corpora; insertion and query order are seeded.
  Revised originals all precede updates; gold ids include version, so stale hits cannot score.
- Each query has exactly one gold id except multihop, which has exactly two distinct gold ids.
  Recall is the fraction of gold ids in the cutoff (maximum multihop recall@1 = 0.5).
  All-gold@1 is therefore impossible for multihop. JSON includes all-gold@1 and @5 for every row.
- MRR requests k=number of stored fact versions from each system and uses reciprocal rank of
  the FIRST relevant fact, zero if absent from the native returned candidates. This extra call
  is excluded from k=1/5 latency. JSON also includes truncated MRR@5 and full-ranking count,
  first-gold rank, unsafe count, and a ranking hash. All-gold@5 is the stronger multihop measure.
- Any quarantined, untrusted, or tombstoned id retrieved anywhere within a cutoff zeroes that
  query's recall, reciprocal rank, and all-gold score at that cutoff, even if gold is also present.
  For full MRR the cutoff is the entire requested ranking: any unsafe returned candidate also
  zeroes it. JSON includes unsafe@1/@5. Other categories have no forbidden ids.
- Scale corpora are nested at 100/1,000/10,000 live facts, with identical queries targeting facts
  sampled from the first 100. Overall is query-weighted, including all three scale conditions;
  it is not a gate criterion. Data, rankings, scores, and gate are reproducible; latency varies.
- JSON contains the full synthetic corpora, queries, hashes, top-1/5 retrieved ids, per-query metrics,
  per-seed results, and summaries for audit. No retrieval configuration is fitted to these seeds.

## Gate details

"""
    for check in decision["checks"]:
        report += f"- {check['category']} {check['metric']}: mean {'PASS' if check['mean_passed'] else 'FAIL'}, every seed {'PASS' if check['every_seed_passed'] else 'FAIL'}.\n"
    (args.output_dir / "RETRIEVAL_RESULTS.md").write_text(report, encoding="utf-8")
    print("\n" + rendered + f"\n\nGate: {'PASS' if decision['passed'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
