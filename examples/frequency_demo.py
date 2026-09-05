"""Library-level demo of CountMinSketch: how often did each fruit appear?

Run from the repo root:

    python examples/frequency_demo.py

Uses ``pdst`` directly against the same seeded skewed stream
(``generate_sample_data.generate_fruit_stream``) that the CLI's
``pdst frequency`` command reads from ``examples/sample_data/fruit_stream.txt``
(run ``generate_sample_data.py`` once to write that file if you want to try
the CLI version too).
"""
from collections import Counter

from generate_sample_data import generate_fruit_stream

from pdst import CountMinSketch


def main() -> None:
    stream = generate_fruit_stream()
    exact = Counter(stream)

    cms = CountMinSketch(epsilon=0.01, delta=0.01)
    cms.update(stream)

    print(f"Processed {cms.total_count} tokens with a sketch of only "
          f"{cms.width}x{cms.depth} = {cms.width * cms.depth} counters "
          f"(vs. {len(exact)} distinct keys an exact Counter would need).")
    print(f"Error bound (epsilon * total_count): {cms.error_bound():.1f}\n")

    print(f"{'item':<12} {'exact':>7} {'estimated':>10} {'error':>7}")
    for item, exact_count in exact.most_common():
        estimate = cms.estimate(item)
        print(f"{item:<12} {exact_count:>7} {estimate:>10} {estimate - exact_count:>+7}")


if __name__ == "__main__":
    main()
