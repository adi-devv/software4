"""Download and prepare fixed GSM8K and ARC-Challenge test canaries."""
import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.canary_data import arc_rows, gsm8k_rows, write_jsonl


def _server_rows(dataset, config):
    rows = []
    while True:
        query = urlencode({"dataset": dataset, "config": config, "split": "test",
                           "offset": len(rows), "length": 100})
        with urlopen("https://datasets-server.huggingface.co/rows?" + query, timeout=60) as response:
            page = json.load(response)
        batch = [entry["row"] for entry in page["rows"]]
        rows.extend(batch)
        if len(rows) >= page["num_rows_total"]:
            return rows
        if not batch:
            raise RuntimeError(f"Dataset server returned an empty page for {dataset}")


def _load(dataset, config):
    try:
        from datasets import load_dataset
    except ImportError:
        return _server_rows(dataset, config)
    return list(load_dataset(dataset, config, split="test"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=400, help="number of canaries per dataset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("data/canaries"))
    args = parser.parse_args()
    if args.n <= 0:
        parser.error("--n must be positive")
    gsm = gsm8k_rows(_load("openai/gsm8k", "main"), args.n, args.seed)
    arc = arc_rows(_load("allenai/ai2_arc", "ARC-Challenge"), args.n, args.seed)
    if len(gsm) < args.n or len(arc) < args.n:
        raise ValueError(f"Not enough valid unique rows: GSM8K {len(gsm)}, ARC {len(arc)}; requested {args.n}")
    for name, rows in (("gsm8k", gsm), ("arc", arc)):
        path = args.out / f"{name}.jsonl"
        write_jsonl(rows, path)
        print(f"{path}: {len(rows)} rows")


if __name__ == "__main__":
    main()
