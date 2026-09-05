import random

import pytest

from pdst.bloom_filter import BloomFilter, optimal_parameters


def test_optimal_parameters_matches_known_textbook_values():
    # n=1000, p=0.01 is a commonly cited example: m ~= 9585 bits, k ~= 7.
    params = optimal_parameters(1000, 0.01)
    assert 9500 <= params.num_bits <= 9700
    assert params.num_hashes == 7


def test_optimal_parameters_rejects_bad_input():
    with pytest.raises(ValueError):
        optimal_parameters(0, 0.01)
    with pytest.raises(ValueError):
        optimal_parameters(-5, 0.01)
    with pytest.raises(ValueError):
        optimal_parameters(1000, 0.0)
    with pytest.raises(ValueError):
        optimal_parameters(1000, 1.0)


def test_no_false_negatives():
    bf = BloomFilter(expected_items=1000, false_positive_rate=0.01)
    items = [f"item-{i}" for i in range(1000)]
    bf.update(items)
    for item in items:
        assert item in bf


def test_added_count_tracks_number_of_adds():
    bf = BloomFilter(expected_items=10, false_positive_rate=0.1)
    assert bf.added_count == 0
    bf.add("a")
    bf.add("b")
    assert bf.added_count == 2


def test_empirical_false_positive_rate_is_in_the_right_ballpark():
    # Statistical test with a generous tolerance to avoid flakiness: insert
    # 2000 items sized for a 1% target, then query 5000 items guaranteed
    # absent and check the observed rate isn't wildly off from the target.
    rng = random.Random(42)
    target_fp_rate = 0.01
    bf = BloomFilter(expected_items=2000, false_positive_rate=target_fp_rate)
    inserted = [f"real-item-{i}" for i in range(2000)]
    bf.update(inserted)

    absent = [f"absent-item-{rng.randint(0, 10**9)}-{i}" for i in range(5000)]
    false_positives = sum(1 for item in absent if item in bf)
    observed_rate = false_positives / len(absent)

    assert observed_rate < target_fp_rate * 5


def test_current_false_positive_rate_grows_with_fill():
    bf = BloomFilter(expected_items=100, false_positive_rate=0.1)
    empty_rate = bf.current_false_positive_rate()
    bf.update(f"item-{i}" for i in range(300))  # overfill well past sizing target
    full_rate = bf.current_false_positive_rate()
    assert full_rate > empty_rate


def test_serialization_round_trip_preserves_behavior():
    bf = BloomFilter(expected_items=500, false_positive_rate=0.02)
    items = [f"x-{i}" for i in range(500)]
    bf.update(items)

    restored = BloomFilter.from_dict(bf.to_dict())
    assert restored.num_bits == bf.num_bits
    assert restored.num_hashes == bf.num_hashes
    assert restored.added_count == bf.added_count
    for item in items:
        assert item in restored
    # the restored filter must agree with the original on an absent item too,
    # since both share the same bit array and hash parameters
    absent = "definitely-not-in-here"
    assert (absent in restored) == (absent in bf)


def test_save_and_load_round_trip(tmp_path):
    bf = BloomFilter(expected_items=200, false_positive_rate=0.05)
    items = [f"saved-{i}" for i in range(200)]
    bf.update(items)

    path = tmp_path / "filter.json"
    bf.save(str(path))
    restored = BloomFilter.load(str(path))

    for item in items:
        assert item in restored
    assert restored.num_bits == bf.num_bits
    assert restored.num_hashes == bf.num_hashes


def test_explicit_num_bits_and_num_hashes():
    bf = BloomFilter(num_bits=64, num_hashes=3)
    assert bf.num_bits == 64
    assert bf.num_hashes == 3
    bf.add("only-item")
    assert "only-item" in bf


def test_explicit_params_reject_non_positive_values():
    with pytest.raises(ValueError):
        BloomFilter(num_bits=0, num_hashes=3)
    with pytest.raises(ValueError):
        BloomFilter(num_bits=64, num_hashes=0)
