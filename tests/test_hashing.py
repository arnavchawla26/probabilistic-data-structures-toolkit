import pytest

from pdst.hashing import double_hash_pair, hash64, kth_hash


def test_hash64_is_deterministic():
    assert hash64("hello") == hash64("hello")
    assert hash64("hello", seed=5) == hash64("hello", seed=5)


def test_hash64_differs_by_seed():
    assert hash64("hello", seed=0) != hash64("hello", seed=1)


def test_hash64_differs_by_item():
    assert hash64("hello") != hash64("world")


def test_hash64_accepts_non_string_items():
    # int/float items get str()-ed before hashing; just check it doesn't blow up
    # and stays deterministic.
    assert hash64(42) == hash64(42)
    assert hash64(3.14) == hash64(3.14)


def test_hash64_range_is_64_bit_unsigned():
    for item in ["a", "b", "a longer string of text", ""]:
        h = hash64(item)
        assert 0 <= h < (1 << 64)


def test_double_hash_pair_h2_is_odd():
    for item in ["x", "y", "z", "a much longer item to hash"]:
        _, h2 = double_hash_pair(item)
        assert h2 % 2 == 1


def test_kth_hash_rejects_non_positive_modulus():
    with pytest.raises(ValueError):
        kth_hash("x", 0, 0)
    with pytest.raises(ValueError):
        kth_hash("x", 0, -5)


def test_kth_hash_is_within_modulus():
    for k in range(10):
        h = kth_hash("some-item", k, 997)
        assert 0 <= h < 997


def test_kth_hash_family_looks_roughly_uniform():
    # Loose statistical sanity check: hashing many distinct items into a
    # modest number of buckets with one fixed k should not pile up in a tiny
    # subset of buckets. Not a rigorous uniformity test -- just a guard
    # against an obviously broken (e.g. constant) hash function.
    modulus = 64
    buckets = [0] * modulus
    n = 5000
    for i in range(n):
        buckets[kth_hash(f"item-{i}", 3, modulus)] += 1
    expected = n / modulus
    # every bucket should be within a generous factor of the expected count
    assert all(expected * 0.5 <= count <= expected * 1.5 for count in buckets)
