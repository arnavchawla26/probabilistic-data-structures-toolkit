"""Library-level demo of HyperLogLog: how many distinct visitors, and merging.

Run from the repo root:

    python examples/cardinality_demo.py

Uses ``pdst`` directly against the same seeded synthetic email data
(``generate_sample_data.generate_emails``) that the CLI's ``pdst
cardinality`` command reads from ``examples/sample_data/emails.txt`` (run
``generate_sample_data.py`` once to write that file if you want to try the
CLI version too), plus a merge demo showing how two independently-built
sketches combine into one that estimates the cardinality of their *union*
without ever seeing the raw data together.
"""
from generate_sample_data import generate_emails

from pdst import HyperLogLog


def main() -> None:
    emails = generate_emails()
    exact_distinct = len(set(emails))

    hll = HyperLogLog(precision=12)
    hll.update(emails)
    estimate = hll.cardinality()
    relative_error = abs(estimate - exact_distinct) / exact_distinct

    print(f"Exact distinct count: {exact_distinct}")
    print(f"HyperLogLog estimate: {estimate:.1f} (relative error {relative_error:.2%})")
    print(f"Memory: {hll.num_registers} single-byte registers "
          f"vs. ~{sum(len(e.encode()) for e in set(emails))} bytes for the raw set.\n")

    # Merge demo: split the stream into two halves, build a sketch for each
    # independently (as if collected on two different servers), then merge.
    midpoint = len(emails) // 2
    half_a, half_b = HyperLogLog(precision=12), HyperLogLog(precision=12)
    half_a.update(emails[:midpoint])
    half_b.update(emails[midpoint:])

    merged = half_a.merge(half_b)
    merged_estimate = merged.cardinality()
    print(f"Two independent sketches (halves of the stream) merged: "
          f"{merged_estimate:.1f} (true union size: {exact_distinct})")


if __name__ == "__main__":
    main()
