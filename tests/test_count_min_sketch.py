import random
from collections import Counter

import pytest

from pdst.count_min_sketch import CountMinSketch


def test_rejects_bad_params():
    with pytest.raises(ValueError):
        CountMinSketch(epsilon=0.0, delta=0.01)
    with pytest.raises(ValueError):
        CountMinSketch(epsilon=1.0, delta=0.01)
    with pytest.raises(ValueError):
        CountMinSketch(epsilon=0.01, delta=0.0)
    with pytest.raises(ValueError):
        CountMinSketch(epsilon=0.01, delta=1.0)


def test_explicit_width_and_depth_reject_non_positive():
    with pytest.raises(ValueError):
        CountMinSketch(width=0, depth=3)
    with pytest.raises(ValueError):
        CountMinSketch(width=10, depth=0)


def test_add_rejects_negative_count():
    cms = CountMinSketch(epsilon=0.1, delta=0.1)
    with pytest.raises(ValueError):
        cms.add("x", count=-1)


def test_exact_match_when_no_collisions_possible():
    # A single distinct item with a huge table can't collide with anything.
    cms = CountMinSketch(width=10_000, depth=5)
    cms.add("only-item", count=7)
    assert cms.estimate("only-item") == 7
    assert cms.estimate("never-added") == 0


def test_estimate_never_underestimates_true_count():
    rng = random.Random(1)
    cms = CountMinSketch(epsilon=0.05, delta=0.05)
    vocabulary = [f"word-{i}" for i in range(200)]
    exact = Counter()
    for _ in range(20_000):
        word = rng.choice(vocabulary)
        cms.add(word)
        exact[word] += 1

    for word in vocabulary:
        assert cms.estimate(word) >= exact[word]


def test_estimates_stay_within_error_bound_with_high_probability():
    # Statistical property test: build a sketch with a modest (epsilon, delta),
    # feed it a skewed stream (Zipf-like via repeated small vocabulary), and
    # check that every tracked item's overestimation stays within
    # epsilon * total_count -- the guarantee the sketch is supposed to provide
    # with probability >= 1 - delta.
    rng = random.Random(7)
    epsilon, delta = 0.02, 0.01
    cms = CountMinSketch(epsilon=epsilon, delta=delta)
    vocabulary = [f"key-{i}" for i in range(500)]
    exact = Counter()
    for _ in range(50_000):
        key = rng.choice(vocabulary)
        cms.add(key)
        exact[key] += 1

    bound = cms.error_bound()
    violations = 0
    for key in vocabulary:
        error = cms.estimate(key) - exact[key]
        assert error >= 0
        if error > bound:
            violations += 1
    # with delta=0.01 we'd expect roughly at most ~1% of per-item estimates to
    # violate the bound in the worst case; allow generous slack for a single
    # random trial.
    assert violations / len(vocabulary) < 0.1


def test_update_is_equivalent_to_repeated_add():
    a = CountMinSketch(width=500, depth=4)
    b = CountMinSketch(width=500, depth=4)
    items = ["p", "q", "p", "r", "p", "q"]
    for item in items:
        a.add(item)
    b.update(items)
    for item in set(items):
        assert a.estimate(item) == b.estimate(item)
    assert a.total_count == b.total_count == len(items)


def test_serialization_round_trip(tmp_path):
    cms = CountMinSketch(epsilon=0.05, delta=0.05)
    cms.update(["a", "b", "a", "c", "a", "b"])

    restored = CountMinSketch.from_dict(cms.to_dict())
    assert restored.width == cms.width
    assert restored.depth == cms.depth
    assert restored.total_count == cms.total_count
    for item in ["a", "b", "c", "never-seen"]:
        assert restored.estimate(item) == cms.estimate(item)

    path = tmp_path / "cms.json"
    cms.save(str(path))
    loaded = CountMinSketch.load(str(path))
    for item in ["a", "b", "c"]:
        assert loaded.estimate(item) == cms.estimate(item)
