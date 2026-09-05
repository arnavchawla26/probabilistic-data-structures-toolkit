import pytest

from pdst.bitarray import BitArray


def test_rejects_non_positive_size():
    with pytest.raises(ValueError):
        BitArray(0)
    with pytest.raises(ValueError):
        BitArray(-1)


def test_all_bits_start_unset():
    arr = BitArray(16)
    assert all(not arr.get(i) for i in range(16))
    assert arr.count_set() == 0


def test_set_and_get():
    arr = BitArray(16)
    arr.set(0)
    arr.set(15)
    arr.set(7)
    assert arr.get(0) and arr.get(15) and arr.get(7)
    assert not arr.get(1)
    assert arr.count_set() == 3


def test_setting_twice_is_idempotent():
    arr = BitArray(8)
    arr.set(3)
    arr.set(3)
    assert arr.count_set() == 1


def test_out_of_range_raises_index_error():
    arr = BitArray(8)
    with pytest.raises(IndexError):
        arr.get(8)
    with pytest.raises(IndexError):
        arr.get(-1)
    with pytest.raises(IndexError):
        arr.set(100)


def test_len_matches_num_bits():
    arr = BitArray(37)
    assert len(arr) == 37
    assert arr.num_bits == 37


def test_to_bytes_from_bytes_round_trip():
    arr = BitArray(20)
    for i in (0, 3, 5, 19):
        arr.set(i)
    data = arr.to_bytes()
    restored = BitArray.from_bytes(data, 20)
    assert restored == arr
    for i in range(20):
        assert restored.get(i) == arr.get(i)


def test_from_bytes_rejects_wrong_length():
    with pytest.raises(ValueError):
        BitArray.from_bytes(b"\x00\x00", 20)  # 20 bits needs 3 bytes, not 2


def test_equality_considers_size_and_content():
    a = BitArray(16)
    b = BitArray(16)
    assert a == b
    a.set(4)
    assert a != b
    c = BitArray(8)
    assert a != c
    assert a != "not a bitarray"
