"""Hashing utilities shared by every structure in this package.

Each structure needs several "independent-looking" hash functions per item
(a Bloom filter needs k of them, a Count-Min Sketch needs one per row). Rather
than compute a fresh cryptographic hash for every one of them, we derive
exactly two independent 64-bit hashes per item -- using BLAKE2b keyed with
distinct salts -- and combine them with the double-hashing technique of
Kirsch & Mitzenmacher, "Less Hashing, Same Performance: Building a Better
Bloom Filter" (2006): h_i(x) = h1(x) + i * h2(x) (mod m). That paper shows
this combination is, for Bloom-filter-style applications, statistically as
good as using k fully independent hash functions, at a fraction of the cost.
"""
from __future__ import annotations

import hashlib
from typing import Tuple, Union

Item = Union[str, bytes, int, float]

_HASH_BITS = 64
_MASK_64 = (1 << _HASH_BITS) - 1


def _to_bytes(item: Item) -> bytes:
    if isinstance(item, bytes):
        return item
    return str(item).encode("utf-8")


def _pack_seed(seed: int) -> bytes:
    # blake2b's salt parameter must be at most 16 bytes; 8 bytes comfortably
    # holds any seed we use (row/hash-function indices are always small).
    return (seed & _MASK_64).to_bytes(8, byteorder="little")


def hash64(item: Item, seed: int = 0) -> int:
    """Return a deterministic, uniformly-distributed 64-bit unsigned hash.

    Different ``seed`` values produce independent-looking hashes of the same
    item -- this is what lets CountMinSketch build several hash "rows" and
    HyperLogLog build its single hash from one primitive (BLAKE2b) instead of
    needing a whole family of distinct hash functions.
    """
    digest = hashlib.blake2b(_to_bytes(item), digest_size=8, salt=_pack_seed(seed)).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def double_hash_pair(item: Item) -> Tuple[int, int]:
    """Return the (h1, h2) pair that ``kth_hash`` derives all k hashes from."""
    h1 = hash64(item, seed=1)
    h2 = hash64(item, seed=2) | 1  # force odd: avoids short cycles when reducing mod a power of two
    return h1, h2


def kth_hash(item: Item, k: int, modulus: int) -> int:
    """Return the k-th (0-indexed) derived hash of ``item``, reduced mod ``modulus``.

    Uses Kirsch-Mitzenmacher double hashing: h_k(x) = (h1(x) + k*h2(x)) mod modulus.
    """
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    h1, h2 = double_hash_pair(item)
    return (h1 + k * h2) % modulus
