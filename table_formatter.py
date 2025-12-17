from collections import Counter
from typing import Iterable, List, Sequence, Union


Number = Union[float, int]


def _ensure_square_table(table: Sequence[Sequence[Number]], expected_size: int = 49) -> List[List[Number]]:
    """Validate that the table is a square grid of the expected size and numeric."""
    if len(table) != expected_size:
        raise ValueError(f"Ожидалась таблица {expected_size}x{expected_size}, получено {len(table)} строк(и).")

    normalized: List[List[Number]] = []
    for row_index, row in enumerate(table):
        if len(row) != expected_size:
            raise ValueError(
                f"Ожидалась таблица {expected_size}x{expected_size}, в строке {row_index} обнаружено {len(row)} столбцов."
            )
        validated_row: List[Number] = []
        for value in row:
            if not isinstance(value, (int, float)):
                raise TypeError("Все элементы таблицы должны быть числами.")
            validated_row.append(value)
        normalized.append(validated_row)

    return normalized


def _format_number(value: Number) -> str:
    """Return a compact, human-friendly representation for ints and floats."""
    return f"{value:g}"


def format_table_descending_with_frequency(table: Sequence[Sequence[Number]]) -> list[str]:
    """
    Отсортировать элементы таблицы 49x49 по убыванию и вернуть строки вида:

    23. 3.45; 51

    где 23 — порядковый номер элемента в отсортированном списке,
    3.45 — значение элемента, 51 — частота встречаемости.
    """
    normalized = _ensure_square_table(table)
    flattened: Iterable[Number] = (value for row in normalized for value in row)

    frequencies = Counter(flattened)
    sorted_items = sorted(frequencies.items(), key=lambda item: item[0], reverse=True)

    result = [f"{index}. {_format_number(value)}; {count}" for index, (value, count) in enumerate(sorted_items, start=1)]
    return result
