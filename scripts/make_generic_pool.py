"""Generate a fixed, reproducible pool of ordinary vocabulary questions."""
import json
from pathlib import Path


def main():
    rows = [dict(prompt=f"Which word completes this phrase: {word} and ___?", answer=answer)
            for word, answer in [("salt", "pepper"), ("bread", "butter"),
                                 ("needle", "thread"), ("lock", "key"),
                                 ("pen", "paper"), ("fish", "chips"),
                                 ("black", "white"), ("day", "night")]]
    path = Path(__file__).resolve().parents[1] / "data/generic/pool.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
