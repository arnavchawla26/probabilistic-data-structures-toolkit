"""``pdst``: run each structure over a real text/CSV stream and compare it
against the exact answer computed with Python's own ``set``/``Counter``.

Three subcommands, one per structure:

    pdst membership   --input FILE [--queries FILE] [--fp-rate 0.01]
    pdst frequency    --input FILE [--epsilon 0.01] [--delta 0.01] [--top-k 10]
    pdst cardinality  --input FILE [--precision 12]

Each reads one token per line (or a CSV column via ``--column``), builds the
probabilistic structure, and prints an accuracy-vs-exact comparison report
(add ``--json`` for machine-readable output).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from typing import List, Optional, Sequence

from .bloom_filter import BloomFilter
from .count_min_sketch import CountMinSketch
from .hyperloglog import HyperLogLog


def read_tokens(path: str, column: Optional[int] = None) -> List[str]:
    """Read one token per non-empty line from ``path``.

    If ``column`` is given, each line is treated as CSV and that (0-indexed)
    column is used as the token instead of the whole line.
    """
    tokens: List[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            if column is not None:
                fields = line.split(",")
                if column >= len(fields):
                    raise ValueError(f"line has only {len(fields)} column(s), --column {column} is out of range")
                token = fields[column].strip()
            else:
                token = line
            if token:
                tokens.append(token)
    return tokens


def _exact_storage_bytes(distinct_tokens: Sequence[str]) -> int:
    """Rough lower bound on the memory an exact ``set`` of these strings would use."""
    return sum(len(t.encode("utf-8")) for t in distinct_tokens)


def run_membership(input_path: str, queries_path: Optional[str], fp_rate: float, column: Optional[int]) -> dict:
    tokens = read_tokens(input_path, column)
    distinct = list(dict.fromkeys(tokens))
    exact_set = set(distinct)

    bf = BloomFilter(expected_items=max(len(distinct), 1), false_positive_rate=fp_rate)
    bf.update(distinct)

    if queries_path:
        query_tokens = read_tokens(queries_path, column)
    else:
        # No held-out queries given: synthesize negatives guaranteed absent from
        # the input, so we can still measure a real empirical false-positive rate.
        probe_count = max(len(distinct), 100)
        query_tokens = [f"__pdst_negative_probe__{i}" for i in range(probe_count)]
        query_tokens = [t for t in query_tokens if t not in exact_set] + distinct

    false_positives = false_negatives = true_positives = true_negatives = 0
    for token in query_tokens:
        actual = token in exact_set
        predicted = token in bf
        if predicted and actual:
            true_positives += 1
        elif predicted and not actual:
            false_positives += 1
        elif not predicted and actual:
            false_negatives += 1
        else:
            true_negatives += 1

    negatives_tested = false_positives + true_negatives
    observed_fp_rate = (false_positives / negatives_tested) if negatives_tested else None
    bloom_bytes = (bf.num_bits + 7) // 8
    exact_bytes = _exact_storage_bytes(distinct)

    return {
        "structure": "bloom_filter",
        "distinct_items_added": len(distinct),
        "num_bits": bf.num_bits,
        "num_hashes": bf.num_hashes,
        "size_bytes": bloom_bytes,
        "exact_set_size_bytes_approx": exact_bytes,
        "memory_savings_ratio": round(exact_bytes / bloom_bytes, 2) if bloom_bytes else None,
        "target_false_positive_rate": fp_rate,
        "current_estimated_false_positive_rate": round(bf.current_false_positive_rate(), 6),
        "queries_tested": len(query_tokens),
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives_should_always_be_zero": false_negatives,
        "observed_false_positive_rate": round(observed_fp_rate, 6) if observed_fp_rate is not None else None,
    }


def run_frequency(input_path: str, epsilon: float, delta: float, top_k: int, column: Optional[int]) -> dict:
    tokens = read_tokens(input_path, column)
    exact_counts = Counter(tokens)

    cms = CountMinSketch(epsilon=epsilon, delta=delta)
    cms.update(tokens)

    top_items = exact_counts.most_common(top_k)
    per_item = []
    errors = []
    for item, exact_count in top_items:
        estimate = cms.estimate(item)
        error = estimate - exact_count
        errors.append(error)
        per_item.append({"item": item, "exact_count": exact_count, "estimated_count": estimate, "error": error})

    max_error = max(errors) if errors else 0
    mean_abs_error = (sum(abs(e) for e in errors) / len(errors)) if errors else 0.0

    return {
        "structure": "count_min_sketch",
        "total_tokens_processed": cms.total_count,
        "distinct_tokens": len(exact_counts),
        "width": cms.width,
        "depth": cms.depth,
        "size_counters": cms.width * cms.depth,
        "error_bound_epsilon_times_total": round(cms.error_bound(), 4),
        "top_k": per_item,
        "max_observed_error": max_error,
        "mean_absolute_error": round(mean_abs_error, 4),
        "all_estimates_within_error_bound": max_error <= cms.error_bound() + 1e-9,
    }


def run_cardinality(input_path: str, precision: int, column: Optional[int]) -> dict:
    tokens = read_tokens(input_path, column)
    exact_distinct = len(set(tokens))

    hll = HyperLogLog(precision=precision)
    hll.update(tokens)
    estimate = hll.cardinality()

    relative_error = abs(estimate - exact_distinct) / exact_distinct if exact_distinct else 0.0
    hll_bytes = hll.num_registers  # one small int per register; stored packed this would be ~6 bits each
    exact_bytes = _exact_storage_bytes(set(tokens))

    return {
        "structure": "hyperloglog",
        "stream_length": len(tokens),
        "exact_distinct_count": exact_distinct,
        "estimated_distinct_count": round(estimate, 2),
        "precision": precision,
        "num_registers": hll.num_registers,
        "size_bytes_approx": hll_bytes,
        "exact_set_size_bytes_approx": exact_bytes,
        "memory_savings_ratio": round(exact_bytes / hll_bytes, 2) if hll_bytes else None,
        "relative_error": round(relative_error, 4),
        "relative_error_pct": round(relative_error * 100, 2),
    }


def _print_report(title: str, result: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, indent=2))
        return
    print(f"=== {title} ===")
    for key, value in result.items():
        if key == "top_k":
            print("  top_k:")
            for row in value:
                print(f"    {row['item']!r}: exact={row['exact_count']} estimated={row['estimated_count']} error={row['error']:+d}")
            continue
        print(f"  {key}: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pdst", description="Probabilistic data structures: build, query, compare to exact.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    membership = subparsers.add_parser("membership", help="Bloom filter set-membership demo")
    membership.add_argument("--input", required=True, help="File of items to insert, one per line")
    membership.add_argument("--queries", default=None, help="File of items to query (default: synthetic negatives + the input itself)")
    membership.add_argument("--fp-rate", type=float, default=0.01, help="Target false positive rate (default: 0.01)")
    membership.add_argument("--column", type=int, default=None, help="0-indexed CSV column to use instead of whole line")
    membership.add_argument("--json", action="store_true", help="Print JSON instead of a text report")

    frequency = subparsers.add_parser("frequency", help="Count-Min Sketch stream-frequency demo")
    frequency.add_argument("--input", required=True, help="File of stream tokens, one per line")
    frequency.add_argument("--epsilon", type=float, default=0.01, help="Error factor (default: 0.01)")
    frequency.add_argument("--delta", type=float, default=0.01, help="Failure probability (default: 0.01)")
    frequency.add_argument("--top-k", type=int, default=10, help="How many most-frequent items to report (default: 10)")
    frequency.add_argument("--column", type=int, default=None, help="0-indexed CSV column to use instead of whole line")
    frequency.add_argument("--json", action="store_true", help="Print JSON instead of a text report")

    cardinality = subparsers.add_parser("cardinality", help="HyperLogLog distinct-count demo")
    cardinality.add_argument("--input", required=True, help="File of stream tokens, one per line")
    cardinality.add_argument("--precision", type=int, default=12, help="Register-index bits, 4-24 (default: 12)")
    cardinality.add_argument("--column", type=int, default=None, help="0-indexed CSV column to use instead of whole line")
    cardinality.add_argument("--json", action="store_true", help="Print JSON instead of a text report")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "membership":
            result = run_membership(args.input, args.queries, args.fp_rate, args.column)
            _print_report("Bloom Filter: Set Membership", result, args.json)
        elif args.command == "frequency":
            result = run_frequency(args.input, args.epsilon, args.delta, args.top_k, args.column)
            _print_report("Count-Min Sketch: Stream Frequency", result, args.json)
        elif args.command == "cardinality":
            result = run_cardinality(args.input, args.precision, args.column)
            _print_report("HyperLogLog: Cardinality Estimation", result, args.json)
        else:  # pragma: no cover - argparse enforces valid choices
            parser.error(f"unknown command {args.command!r}")
            return 2
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
