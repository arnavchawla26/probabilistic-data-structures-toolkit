"""Count-Min Sketch: an approximate per-item frequency counter for streams.

Sized from an ``(epsilon, delta)`` accuracy guarantee following Cormode &
Muthukrishnan, "An Improved Data Stream Summary: The Count-Min Sketch and its
Applications" (2005): with width ``w = ceil(e / epsilon)`` and depth
``d = ceil(ln(1 / delta))``, the estimate for any item overestimates its true
count by at most ``epsilon * total_count`` with probability >= ``1 - delta``.

Estimates are always >= the true count (the sketch only ever adds "noise"
from unrelated items colliding into the same counter, and ``estimate()``
takes the minimum across independent rows to cancel out as much of that noise
as possible) -- it never underestimates.
"""
from __future__ import annotations

import json
import math
from typing import Iterable, List, Optional

from .hashing import Item, hash64

# seed offset so CMS's row hashes don't collide with BloomFilter's/HyperLogLog's
# use of the same underlying hash64 primitive with small seeds.
_ROW_SEED_OFFSET = 100


class CountMinSketch:
    def __init__(
        self,
        epsilon: float = 0.01,
        delta: float = 0.01,
        *,
        width: Optional[int] = None,
        depth: Optional[int] = None,
    ):
        if width is not None and depth is not None:
            if width <= 0:
                raise ValueError("width must be positive")
            if depth <= 0:
                raise ValueError("depth must be positive")
            self._width, self._depth = width, depth
        else:
            if not 0 < epsilon < 1:
                raise ValueError("epsilon must be in (0, 1)")
            if not 0 < delta < 1:
                raise ValueError("delta must be in (0, 1)")
            self._width = math.ceil(math.e / epsilon)
            self._depth = math.ceil(math.log(1 / delta))
        self._epsilon = epsilon
        self._delta = delta
        self._table: List[List[int]] = [[0] * self._width for _ in range(self._depth)]
        self._total = 0

    @property
    def width(self) -> int:
        return self._width

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def total_count(self) -> int:
        return self._total

    def _row_index(self, item: Item, row: int) -> int:
        return hash64(item, seed=_ROW_SEED_OFFSET + row) % self._width

    def add(self, item: Item, count: int = 1) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        for row in range(self._depth):
            self._table[row][self._row_index(item, row)] += count
        self._total += count

    def update(self, items: Iterable[Item]) -> None:
        for item in items:
            self.add(item)

    def estimate(self, item: Item) -> int:
        return min(self._table[row][self._row_index(item, row)] for row in range(self._depth))

    def error_bound(self) -> float:
        """Upper bound on overestimation error: epsilon * total_count (see module docstring).

        If the sketch was built from explicit width/depth (no epsilon given),
        the equivalent epsilon is recovered from the width (``e / width``).
        """
        eps = self._epsilon if self._epsilon is not None else math.e / self._width
        return eps * self._total

    def to_dict(self) -> dict:
        return {
            "type": "count_min_sketch",
            "width": self._width,
            "depth": self._depth,
            "epsilon": self._epsilon,
            "delta": self._delta,
            "total": self._total,
            "table": self._table,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CountMinSketch":
        cms = cls(
            epsilon=data["epsilon"],
            delta=data["delta"],
            width=data["width"],
            depth=data["depth"],
        )
        cms._table = [row[:] for row in data["table"]]
        cms._total = data["total"]
        return cms

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "CountMinSketch":
        with open(path) as f:
            return cls.from_dict(json.load(f))
