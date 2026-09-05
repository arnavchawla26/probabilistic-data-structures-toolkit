# probabilistic-data-structures-toolkit

Bloom filter, Count-Min Sketch, and HyperLogLog implemented from scratch in
pure Python (standard library only -- no `mmh3`, no `bitarray`, no `numpy`),
plus a `pdst` command-line tool that runs each one against a real text/CSV
stream and reports its accuracy against the exact answer computed with
Python's own `set`/`Counter`.

These three structures answer three different questions about a stream of
data using far less memory than storing the data itself, at the cost of a
small, tunable, provable amount of error:

| Structure | Question it answers | Can be wrong how |
|---|---|---|
| **Bloom filter** | "Have I seen this item before?" | False positives only, never false negatives |
| **Count-Min Sketch** | "How many times have I seen this item?" | Overestimates only, never underestimates |
| **HyperLogLog** | "How many *distinct* items have I seen?" | Estimate can be a bit high or low, tightly bounded |

## Tech stack

Python 3.8+, standard library only (`hashlib`, `dataclasses`, `argparse`,
`json`, `math`). Tests use `pytest`. No third-party runtime dependencies.

## How it works

### Shared hashing (`pdst/hashing.py`)

Every structure needs several independent-looking hash functions per item.
Instead of paying for a fresh cryptographic hash per function, this package
derives two independent 64-bit hashes per item with BLAKE2b (keyed with
distinct salts) and combines them via the double-hashing technique from
Kirsch & Mitzenmacher, *"Less Hashing, Same Performance: Building a Better
Bloom Filter"* (2006): `h_i(x) = h1(x) + i * h2(x) (mod m)`. That paper shows
this is statistically as good as `k` fully independent hash functions for
this kind of application, at a fraction of the cost.

### Bloom filter (`pdst/bloom_filter.py`)

Sizes its bit array and hash-function count from a target
`(expected_items, false_positive_rate)` using the standard formulas
(`m = ceil(-n*ln(p)/ln(2)^2)`, `k = round(m/n*ln(2))`). Backed by a
from-scratch packed `BitArray` (`pdst/bitarray.py`), not the third-party
`bitarray` library. Supports JSON serialization (`to_dict`/`from_dict`,
`save`/`load`) so a filter can be built once and reused elsewhere.

### Count-Min Sketch (`pdst/count_min_sketch.py`)

Sizes its `width x depth` counter table from an `(epsilon, delta)` accuracy
target (Cormode & Muthukrishnan, 2005): estimates overshoot the true count by
at most `epsilon * total_count` with probability >= `1 - delta`. Estimates
are always >= the true count -- `estimate()` takes the min across independent
rows to cancel out as much hash-collision noise as possible.

### HyperLogLog (`pdst/hyperloglog.py`)

Implements the original Flajolet et al. (2007) algorithm: each item's hash
selects a register by its top `precision` bits, and that register stores the
longest run of leading zeros seen in the remaining bits. The estimate is the
bias-corrected harmonic mean across registers, with small-range correction
(linear counting) for low cardinalities where the harmonic-mean estimator is
otherwise biased. Supports `merge()` for combining two independently-built
sketches into an estimate of their union's cardinality (exact via
per-register max -- no accuracy lost by merging this way). Large-range
correction (relevant only as cardinality approaches 2^64) is not implemented;
out of scope for the stream sizes this toolkit targets.

## CLI usage

```
pip install -e ".[dev]"

pdst membership  --input FILE [--queries FILE] [--fp-rate 0.01] [--column N] [--json]
pdst frequency   --input FILE [--epsilon 0.01] [--delta 0.01] [--top-k 10] [--column N] [--json]
pdst cardinality --input FILE [--precision 12] [--column N] [--json]
```

Each subcommand reads one token per line of `--input` (or a 0-indexed CSV
column via `--column`), builds the corresponding structure, and prints a
report comparing it against the exact answer.

The `examples/sample_data/*.txt` files used below are generated, not
committed -- run `python examples/generate_sample_data.py` once to write them
(both generators are seeded, so this always reproduces the exact numbers
quoted here).

### `pdst membership` -- real output, 3000 synthetic email addresses

```
$ pdst membership --input examples/sample_data/emails.txt --fp-rate 0.01
=== Bloom Filter: Set Membership ===
  structure: bloom_filter
  distinct_items_added: 3000
  num_bits: 28756
  num_hashes: 7
  size_bytes: 3595
  exact_set_size_bytes_approx: 58890
  memory_savings_ratio: 16.38
  target_false_positive_rate: 0.01
  current_estimated_false_positive_rate: 0.010338
  queries_tested: 6000
  true_positives: 3000
  true_negatives: 2972
  false_positives: 28
  false_negatives_should_always_be_zero: 0
  observed_false_positive_rate: 0.009333
```

No `--queries` file was given, so `pdst` synthesized negative probes
guaranteed absent from the input to measure a real empirical false-positive
rate (0.93%, right where the 1% target says it should land) alongside the
input itself as positive queries.

### `pdst frequency` -- real output, a skewed 2580-token fruit stream

```
$ pdst frequency --input examples/sample_data/fruit_stream.txt --top-k 5
=== Count-Min Sketch: Stream Frequency ===
  structure: count_min_sketch
  total_tokens_processed: 2580
  distinct_tokens: 10
  width: 272
  depth: 5
  size_counters: 1360
  error_bound_epsilon_times_total: 25.8
  top_k:
    'apple': exact=1000 estimated=1000 error=+0
    'banana': exact=600 estimated=600 error=+0
    'cherry': exact=300 estimated=300 error=+0
    'date': exact=200 estimated=200 error=+0
    'elderberry': exact=160 estimated=160 error=+0
  max_observed_error: 0
  mean_absolute_error: 0.0
  all_estimates_within_error_bound: True
```

With only 10 distinct keys and a 1360-counter table, there's no collision
noise at all here -- every estimate is exact. The error bound only starts to
bite with much larger, more collision-prone vocabularies (see
`tests/test_count_min_sketch.py::test_estimates_stay_within_error_bound_with_high_probability`
for a 500-key stress test).

### `pdst cardinality` -- real output, the same 3000 email addresses

```
$ pdst cardinality --input examples/sample_data/emails.txt --precision 12
=== HyperLogLog: Cardinality Estimation ===
  structure: hyperloglog
  stream_length: 3000
  exact_distinct_count: 3000
  estimated_distinct_count: 2989.87
  precision: 12
  num_registers: 4096
  size_bytes_approx: 4096
  exact_set_size_bytes_approx: 58890
  memory_savings_ratio: 14.38
  relative_error: 0.0034
  relative_error_pct: 0.34
```

4096 single-byte registers (4KB) estimate the distinct count of 58KB worth of
raw email strings to within 0.34%.

## Library usage

The three structures are also usable directly as a library --
see `examples/membership_demo.py`, `examples/frequency_demo.py`, and
`examples/cardinality_demo.py` for runnable scripts (`python
examples/cardinality_demo.py`, etc.) that build each structure from the same
seeded sample data as the CLI examples above (via `examples/
generate_sample_data.py`'s generator functions, in-memory -- no need to run
the generator first) and print the same kind of accuracy comparison, plus a
HyperLogLog `merge()` demo:

```python
from pdst import BloomFilter, CountMinSketch, HyperLogLog

bf = BloomFilter(expected_items=10_000, false_positive_rate=0.01)
bf.update(["a@example.com", "b@example.com"])
"a@example.com" in bf  # True

cms = CountMinSketch(epsilon=0.01, delta=0.01)
cms.update(["error", "error", "warning", "error"])
cms.estimate("error")  # 3

hll = HyperLogLog(precision=12)
hll.update(str(i) for i in range(100_000))
hll.cardinality()  # ~100,000
```

## Running the tests

```
pip install -e ".[dev]"
pytest
```

58 tests across 6 files (`test_hashing.py`, `test_bitarray.py`,
`test_bloom_filter.py`, `test_count_min_sketch.py`, `test_hyperloglog.py`,
`test_cli.py`), covering correctness properties for each structure (no false
negatives for Bloom, no underestimation for Count-Min Sketch, standard-error
bounds and small-range correction for HyperLogLog), serialization round
trips, and the CLI end-to-end against temp files.

## Current status

v1, shipped complete and tested. All three structures, their CLI, and all
three example scripts are implemented and passing. Ideas for a future pass
(not started): a `--benchmark` mode that plots memory-vs-accuracy tradeoffs
across a range of parameter settings, and a Scalable Bloom Filter variant
that grows automatically instead of requiring `expected_items` up front.
