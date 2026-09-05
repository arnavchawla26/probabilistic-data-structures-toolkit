"""A minimal, dependency-free mutable bit array, used internally by BloomFilter.

Bits are packed 8 per byte into a ``bytearray``. This package intentionally
avoids the third-party ``bitarray`` library so the whole toolkit stays
pure-stdlib and the bit-packing itself stays visible as part of the "from
scratch" implementation.
"""
from __future__ import annotations


class BitArray:
    __slots__ = ("_bytes", "_num_bits")

    def __init__(self, num_bits: int):
        if num_bits <= 0:
            raise ValueError("num_bits must be positive")
        self._num_bits = num_bits
        self._bytes = bytearray((num_bits + 7) // 8)

    @property
    def num_bits(self) -> int:
        return self._num_bits

    def set(self, index: int) -> None:
        self._check_index(index)
        byte_index, bit_offset = divmod(index, 8)
        self._bytes[byte_index] |= 1 << bit_offset

    def get(self, index: int) -> bool:
        self._check_index(index)
        byte_index, bit_offset = divmod(index, 8)
        return bool(self._bytes[byte_index] & (1 << bit_offset))

    def count_set(self) -> int:
        """Number of bits currently set to 1 (the Bloom filter's "fill count")."""
        return sum(bin(byte).count("1") for byte in self._bytes)

    def to_bytes(self) -> bytes:
        return bytes(self._bytes)

    @classmethod
    def from_bytes(cls, data: bytes, num_bits: int) -> "BitArray":
        expected_len = (num_bits + 7) // 8
        if len(data) != expected_len:
            raise ValueError(f"expected {expected_len} bytes for {num_bits} bits, got {len(data)}")
        arr = cls(num_bits)
        arr._bytes = bytearray(data)
        return arr

    def _check_index(self, index: int) -> None:
        if not (0 <= index < self._num_bits):
            raise IndexError(f"bit index {index} out of range for size {self._num_bits}")

    def __len__(self) -> int:
        return self._num_bits

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BitArray):
            return NotImplemented
        return self._num_bits == other._num_bits and self._bytes == other._bytes
