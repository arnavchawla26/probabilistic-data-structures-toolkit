"""Regenerate the sample data files under examples/sample_data/.

These files aren't checked into the repo (they're generated, not
hand-authored data) -- run this once before trying the CLI examples in the
README that read from ``examples/sample_data/*.txt``:

    python examples/generate_sample_data.py

Both generators are seeded, so re-running this script always reproduces the
exact same files (and the exact same numbers quoted in the README).
"""
import random
from pathlib import Path

SAMPLE_DIR = Path(__file__).parent / "sample_data"


def generate_emails(count: int = 3000) -> list:
    return [f"user{i}@example.com" for i in range(count)]


def generate_fruit_stream(seed: int = 42) -> list:
    rng = random.Random(seed)
    words = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew", "kiwi", "lemon"]
    weights = [50, 30, 15, 10, 8, 6, 4, 3, 2, 1]
    lines = []
    for word, weight in zip(words, weights):
        lines.extend([word] * (weight * 20))
    rng.shuffle(lines)
    return lines


def main() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    emails = generate_emails()
    (SAMPLE_DIR / "emails.txt").write_text("\n".join(emails) + "\n", encoding="utf-8")
    print(f"wrote {len(emails)} lines to {SAMPLE_DIR / 'emails.txt'}")

    fruit_stream = generate_fruit_stream()
    (SAMPLE_DIR / "fruit_stream.txt").write_text("\n".join(fruit_stream) + "\n", encoding="utf-8")
    print(f"wrote {len(fruit_stream)} lines ({len(set(fruit_stream))} distinct) to {SAMPLE_DIR / 'fruit_stream.txt'}")


if __name__ == "__main__":
    main()
