"""Library-level demo of BloomFilter: is this email address one we've seen?

Run from the repo root:

    python examples/membership_demo.py

This uses the ``pdst`` package directly (not the CLI) to show the object
API, against the same seeded synthetic email data
(``generate_sample_data.generate_emails``) that ``pdst membership`` uses via
``examples/sample_data/emails.txt`` (run ``generate_sample_data.py`` once to
write that file if you want to try the CLI version too).
"""
from generate_sample_data import generate_emails

from pdst import BloomFilter


def main() -> None:
    known_emails = generate_emails()

    bf = BloomFilter(expected_items=len(known_emails), false_positive_rate=0.01)
    bf.update(known_emails)

    print(f"Inserted {len(known_emails)} known addresses.")
    print(f"Bit array size: {bf.num_bits} bits ({(bf.num_bits + 7) // 8} bytes), {bf.num_hashes} hash functions.")
    exact_bytes = sum(len(e.encode()) for e in known_emails)
    print(f"Storing the raw strings in a Python set would take ~{exact_bytes} bytes -- "
          f"the filter uses {round(exact_bytes / ((bf.num_bits + 7) // 8), 1)}x less space.")

    # A definite hit: something we actually inserted.
    probe = known_emails[0]
    print(f"\n{probe!r} in filter -> {probe in bf} (must be True: no false negatives)")

    # Definite misses -- these were never inserted, so any "yes" here is a
    # false positive.
    never_inserted = ["not-a-real-user@example.com", "nobody@nowhere.test", "ghost@void.io"]
    false_positives = 0
    for candidate in never_inserted:
        present = candidate in bf
        print(f"{candidate!r} in filter -> {present}" + ("  <- false positive!" if present else ""))
        false_positives += present

    print(f"\nEstimated current false-positive rate from bit-array fill: "
          f"{bf.current_false_positive_rate():.4%} (target was 1%).")


if __name__ == "__main__":
    main()
