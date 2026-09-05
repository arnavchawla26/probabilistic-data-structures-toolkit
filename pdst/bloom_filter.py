"""Bloom filter: a space-efficient probabilistic set-membership structure.

A Bloom filter never produces false negatives -- if it reports "not present",
the item was definitely never added -- but it can produce false positives at
a rate that trades off against memory. This implementation sizes its bit
array and hash-function count from a target ``(expected_items,
false_positive_rate)`` pair using the standard formulas (Bloom, 1970;
Broder & Mitzenmacher, "Network Applications of Bloom Filters: A Survey", 2004):

    m = ceil(-n * ln(p) / ln(2)^2)      # number of bits
    k = round(m/n * ln(2))              # number of hash functions

where n is the expected number of distinct items and p is the target false
positive rate.
"""
from __future__ import annotations

import base64
import json
import math
from dataclasses import dataclass
from typing import Iterable, Optional

from .bitarray import BitArray
from .hashing import Item, kth_hash


@dataclass(frozen=True)
class BloomFilterParams:
    num_bits: int
    num_hashes: int
    expected_items: int
    false_positive_rate: float


def optimal_parameters(expected_items: int, false_positive_rate: float) -> BloomFilterParams:
    """Compute the (num_bits, num_hashes) that minimize memory for a target fp rate."""
    if expected_items <= 0:
        raise ValueError("expected_items must be positive")
    if not 0 < false_positive_rate < 1:
        raise ValueError("false_positive_rate must be in (0, 1)")
    num_bits = math.ceil(-expected_items * math.log(false_positive_rate) / (math.log(2) ** 2))
    num_bits = max(num_bits, 8)
    num_hashes = max(1, round((num_bits / expected_items) * math.log(2)))
    return BloomFilterParams(num_bits, num_hashes, expected_items, false_positive_rate)


class BloomFilter:
    """A Bloom filter sized either from (expected_items, false_positive_rate)
    or from explicit (num_bits, num_hashes) -- the latter is mainly used by
    ``from_dict``/``load`` to reconstruct a filter exactly as it was saved.
    """

    def __init__(
        self,
        expected_items: int = 10_000,
        false_positive_rate: float = 0.01,
        *,
        num_bits: Optional[int] = None,
        num_hashes: Optional[int] = None,
    ):
        if num_bits is not None and num_hashes is not None:
            if num_bits <= 0:
                raise ValueError("num_bits must be positive")
            if num_hashes <= 0:
                raise ValueError("num_hashes must be positive")
            self._params = BloomFilterParams(num_bits, num_hashes, expected_items, false_positive_rate)
        else:
            self._params = optimal_parameters(expected_items, false_positive_rate)
        self._bits = BitArray(self._params.num_bits)
        self._count = 0  # total add() calls made (not necessarily distinct) -- stats only

    @property
    def num_bits(self) -> int:
        return self._params.num_bits

    @property
    def num_hashes(self) -> int:
        return self._params.num_hashes

    @property
    def added_count(self) -> int:
        return self._count

    def add(self, item: Item) -> None:
        for i in range(self._params.num_hashes):
            self._bits.set(kth_hash(item, i, self._params.num_bits))
        self._count += 1

    def update(self, items: Iterable[Item]) -> None:
        for item in items:
            self.add(item)

    def __contains__(self, item: Item) -> bool:
        return all(
            self._bits.get(kth_hash(item, i, self._params.num_bits))
            for i in range(self._params.num_hashes)
        )

    def might_contain(self, item: Item) -> bool:
        """Alias for ``item in filter`` -- reads better at call sites."""
        return item in self

    def current_false_positive_rate(self) -> float:
        """Estimate the *actual* current false-positive rate from the bit array's fill ratio.

        p ~= (1 - e^{-k*n/m})^k ~= fill_ratio^k, using the bits actually set
        rather than the target ``expected_items`` -- so this reflects reality
        even if more or fewer items were added than the filter was sized for.
        """
        fill_ratio = self._bits.count_set() / self._bits.num_bits
        return fill_ratio ** self._params.num_hashes

    def to_dict(self) -> dict:
        return {
            "type": "bloom_filter",
            "num_bits": self._params.num_bits,
            "num_hashes": self._params.num_hashes,
            "expected_items": self._params.expected_items,
            "false_positive_rate": self._params.false_positive_rate,
            "added_count": self._count,
            "bits_b64": base64.b64encode(self._bits.to_bytes()).decode("ascii"),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BloomFilter":
        bf = cls(
            expected_items=data["expected_items"],
            false_positive_rate=data["false_positive_rate"],
            num_bits=data["num_bits"],
            num_hashes=data["num_hashes"],
        )
        bf._bits = BitArray.from_bytes(base64.b64decode(data["bits_b64"]), data["num_bits"])
        bf._count = data["added_count"]
        return bf

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "BloomFilter":
        with open(path) as f:
            return cls.from_dict(json.load(f))
