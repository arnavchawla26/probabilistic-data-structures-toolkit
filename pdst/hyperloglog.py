"""HyperLogLog: a probabilistic distinct-count (cardinality) estimator.

Based on Flajolet, Fusy, Gandouet & Meunier, "HyperLogLog: the analysis of a
near-optimal cardinality estimation algorithm" (2007). Each item is hashed to
a 64-bit value; its top ``precision`` bits select one of ``2**precision``
registers, and that register stores the largest "rho" (position of the
leftmost 1-bit, 1-indexed) seen so far in the remaining bits for any item
mapped to it. Seeing a longer run of leading zeros is exponentially rarer, so
the maximum run length across many registers gives a low-variance estimate of
how many distinct items have been seen -- using far less memory than storing
the items themselves.

The estimate is the bias-corrected harmonic mean across registers (the
"indicator function" below), with Flajolet et al.'s small-range correction
(falling back to linear counting when many registers are still empty) for
low cardinalities where the harmonic-mean estimator is biased. This
implementation targets streams where the number of distinct items stays well
under 2**64 - large-range correction (needed only when cardinality approaches
the hash space size) is not implemented.
"""
from __future__ import annotations

import json
import math
from typing import Iterable, List

from .hashing import Item, hash64

_HASH_BITS = 64
_HASH_MASK = (1 << _HASH_BITS) - 1


def _alpha(m: int) -> float:
    """Bias-correction constant alpha_m from the original HyperLogLog paper."""
    if m == 16:
        return 0.673
    if m == 32:
        return 0.697
    if m == 64:
        return 0.709
    return 0.7213 / (1 + 1.079 / m)


class HyperLogLog:
    def __init__(self, precision: int = 12):
        if not 4 <= precision <= 24:
            raise ValueError("precision must be between 4 and 24")
        self._precision = precision
        self._m = 1 << precision
        self._registers: List[int] = [0] * self._m
        self._alpha = _alpha(self._m)

    @property
    def precision(self) -> int:
        return self._precision

    @property
    def num_registers(self) -> int:
        return self._m

    def add(self, item: Item) -> None:
        h = hash64(item, seed=0)
        index = h >> (_HASH_BITS - self._precision)
        remaining_bits = _HASH_BITS - self._precision
        remaining = h & ((1 << remaining_bits) - 1)
        rho = self._rho(remaining, remaining_bits)
        if rho > self._registers[index]:
            self._registers[index] = rho

    def update(self, items: Iterable[Item]) -> None:
        for item in items:
            self.add(item)

    @staticmethod
    def _rho(value: int, num_bits: int) -> int:
        """Position of the leftmost 1-bit in a ``num_bits``-wide value, 1-indexed.

        All-zero maps to ``num_bits + 1`` (one past the field width), matching
        the convention used by the reference HyperLogLog algorithm.
        """
        if value == 0:
            return num_bits + 1
        leading_zeros = num_bits - value.bit_length()
        return leading_zeros + 1

    def cardinality(self) -> float:
        m = self._m
        indicator = sum(2.0 ** -r for r in self._registers)
        raw_estimate = self._alpha * m * m / indicator
        if raw_estimate <= 2.5 * m:
            zeros = self._registers.count(0)
            if zeros != 0:
                return m * math.log(m / zeros)  # small-range correction: linear counting
        return raw_estimate

    def merge(self, other: "HyperLogLog") -> "HyperLogLog":
        """Return a new HyperLogLog estimating the cardinality of the *union*
        of the two streams (registers merge by taking the max per slot, which
        is exact for HLL -- no information is lost by merging this way).
        """
        if other.precision != self.precision:
            raise ValueError("cannot merge HyperLogLogs with different precision")
        merged = HyperLogLog(self._precision)
        merged._registers = [max(a, b) for a, b in zip(self._registers, other._registers)]
        return merged

    def to_dict(self) -> dict:
        return {
            "type": "hyperloglog",
            "precision": self._precision,
            "registers": self._registers,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HyperLogLog":
        hll = cls(precision=data["precision"])
        registers = data["registers"]
        if len(registers) != hll._m:
            raise ValueError(f"expected {hll._m} registers for precision {data['precision']}, got {len(registers)}")
        hll._registers = list(registers)
        return hll

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "HyperLogLog":
        with open(path) as f:
            return cls.from_dict(json.load(f))
