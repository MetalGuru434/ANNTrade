import pytest

from table_formatter import format_table_descending_with_frequency


def _build_sample_table():
    size = 49
    return [[(i + j) % 3 + 1 for j in range(size)] for i in range(size)]


def test_descending_format_includes_frequencies():
    table = _build_sample_table()

    result = format_table_descending_with_frequency(table)

    assert result[0] == "1. 3; 800"
    assert result[1] == "2. 2; 800"
    assert result[2] == "3. 1; 801"
    assert len(result) == 3


def test_invalid_shape_raises_value_error():
    with pytest.raises(ValueError):
        format_table_descending_with_frequency([[1, 2], [3, 4]])


def test_non_numeric_values_raise_type_error():
    table = _build_sample_table()
    table[0][0] = "oops"

    with pytest.raises(TypeError):
        format_table_descending_with_frequency(table)
