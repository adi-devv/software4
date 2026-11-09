"""Check that all v1 seed runs share source hash and config apart from world_seed.

usage: python scripts/v1_consistency.py <run_dir> <run_dir> ...   (dirs holding results.json)
Exit 1 on any mismatch. Kept outside step4/analyze.py so the source digest is unchanged.
"""
import json
import sys
from pathlib import Path

cfgs = {}
for d in sys.argv[1:]:
    cfg = json.loads((Path(d) / "results.json").read_text(encoding="utf-8"))["config"]
    if not cfg.get("source_sha256"):
        sys.exit(f"{d}: no source_sha256 in config")
    cfgs[d] = cfg
base = None
bad = False
for d, cfg in cfgs.items():
    rest = {k: v for k, v in cfg.items() if k != "world_seed"}
    base = base or rest
    if rest != base:
        bad = True
        print(f"MISMATCH {d}: {sorted(k for k in set(rest) | set(base) if rest.get(k) != base.get(k))}")
print("seeds:", sorted(c["world_seed"] for c in cfgs.values()), "source:", base["source_sha256"][:12])
sys.exit(1 if bad else 0)
