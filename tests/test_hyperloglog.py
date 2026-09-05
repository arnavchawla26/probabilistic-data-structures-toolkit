import random

import pytest

from pdst.hyperloglog import HyperLogLog


def test_rejects_precision_out_of_range():
    with pytest.raises(ValueError):
        HyperLogLog(precision=3)
    with pytest.raises(ValueError):
        HyperLogLog(precision=25)


def test_empty_estimate_is_zero_ish():
    hll = HyperLogLog(precision=10)
    # Linear counting with all registers empty: m * ln(m / m) == 0.
    assert hll.cardinality() == pytest.approx(0.0, abs=1e-6)


def test_adding_duplicates_does_not_increase_cardinality():
    hll = HyperLogLog(precision=10)
    for _ in range(1000):
        hll.add("same-item-every-time")
    assert hll.cardinality() == pytest.approx(1.0, abs=0.5)


def test_small_cardinality_uses_linear_counting_and_is_accurate():
    # Small counts are exactly where the naive harmonic-mean estimator is
    # biased, and linear counting is supposed to correct for it.
    hll = HyperLogLog(precision=12)
    true_count = 25
    for i in range(true_count):
        hll.add(f"small-item-{i}")
    estimate = hll.cardinality()
    assert abs(estimate - true_count) / true_count < 0.15


def test_moderate_cardinality_within_standard_error():
    # Standard error of HyperLogLog is ~1.04 / sqrt(m). For precision=12,
    # m=4096, so ~1.6%. Use a generous 3x tolerance band to keep this
    # non-flaky across random seeds while still catching real regressions.
    rng = random.Random(123)
    hll = HyperLogLog(precision=12)
    true_count = 100_000
    for i in range(true_count):
        # Use rng-derived unique-looking strings so hashing isn't fed a
        # trivially sequential pattern.
        hll.add(f"item-{i}-{rng.random()}")
    estimate = hll.cardinality()
    relative_error = abs(estimate - true_count) / true_count
    assert relative_error < 0.05


def test_merge_of_disjoint_sets_approximates_union():
    hll_a = HyperLogLog(precision=12)
    hll_b = HyperLogLog(precision=12)
    for i in range(5000):
        hll_a.add(f"a-{i}")
    for i in range(5000):
        hll_b.add(f"b-{i}")

    merged = hll_a.merge(hll_b)
    estimate = merged.cardinality()
    true_union = 10000
    assert abs(estimate - true_union) / true_union < 0.1


def test_merge_of_overlapping_sets_deduplicates():
    hll_a = HyperLogLog(precision=12)
    hll_b = HyperLogLog(precision=12)
    for i in range(5000):
        hll_a.add(f"shared-{i}")
    for i in range(5000):
        hll_b.add(f"shared-{i}")  # fully overlapping with hll_a

    merged = hll_a.merge(hll_b)
    estimate = merged.cardinality()
    true_union = 5000
    assert abs(estimate - true_union) / true_union < 0.1


def test_merge_rejects_mismatched_precision():
    hll_a = HyperLogLog(precision=10)
    hll_b = HyperLogLog(precision=12)
    with pytest.raises(ValueError):
        hll_a.merge(hll_b)


def test_rho_all_zero_maps_to_num_bits_plus_one():
    assert HyperLogLog._rho(0, 10) == 11


def test_rho_matches_leading_zero_count():
    # value=1 with num_bits=8: binary 00000001 -> 7 leading zeros -> rho=8
    assert HyperLogLog._rho(1, 8) == 8
    # value=0b10000000 with num_bits=8: 0 leading zeros -> rho=1
    assert HyperLogLog._rho(0b10000000, 8) == 1


def test_serialization_round_trip(tmp_path):
    hll = HyperLogLog(precision=10)
    for i in range(2000):
        hll.add(f"item-{i}")
    original_estimate = hll.cardinality()

    restored = HyperLogLog.from_dict(hll.to_dict())
    assert restored.precision == hll.precision
    assert restored.cardinality() == pytest.approx(original_estimate)

    path = tmp_path / "hll.json"
    hll.save(str(path))
    loaded = HyperLogLog.load(str(path))
    assert loaded.cardinality() == pytest.approx(original_estimate)


def test_from_dict_rejects_wrong_register_count():
    with pytest.raises(ValueError):
        HyperLogLog.from_dict({"precision": 10, "registers": [0, 0, 0]})
