"""Freeze read-only repo imports into step4/upload/run.py; never submit or launch."""
import argparse
import base64
import io
import json
from pathlib import Path
import zipfile


def prepare(full=False):
    root = Path(__file__).resolve().parents[1]
    target = root / "step4/upload"
    target.mkdir(parents=True, exist_ok=True)
    files = [p for folder in ("baselines", "harness", "hippocampus")
             for p in (root / folder).glob("*.py")]
    files += [root / "scripts/smoke_sleep.py", root / "step4/hf.py", root / "step4/run.py"]
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(files):
            info = zipfile.ZipInfo(path.relative_to(root).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, path.read_bytes())
    script = (root / "step4/run.py").read_text(encoding="utf-8")
    script = script.replace('SOURCE_B64 = ""', 'SOURCE_B64 = ' + repr(base64.b64encode(archive.getvalue()).decode()), 1)
    if full:
        script = script.replace('DEFAULT_SMOKE = "1"', 'DEFAULT_SMOKE = "0"', 1)
    (target / "run.py").write_text(script, encoding="utf-8")
    metadata = json.loads((root / "step4/kernel-metadata.json").read_text(encoding="utf-8"))
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {target} ({len(files)} imported source files; default {'full' if full else 'smoke'}). No launch.")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="bake full default; matching smoke receipt still required")
    prepare(parser.parse_args().full)
