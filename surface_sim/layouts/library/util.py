def invert_shift(row_shift: int, col_shift: int) -> tuple[int, int]:
    """Inverts a row and column shift.

    Parameters
    ----------
    row_shift
        The row shift.
    col_shift
        The column shift.

    Returns
    -------
    tuple[int, int]
        The inverted row and column shift.
    """
    return -row_shift, -col_shift


def is_valid(row: int, col: int, max_size_row: int, max_size_col: int) -> bool:
    """Checks if a row and column are valid for a grid of a given size.

    Parameters
    ----------
    row
        The row.
    col
        The column.
    max_size_row
        The row size of the grid.
    max_size_col
        The column size of the grid.

    Returns
    -------
    bool
        Whether the row and column are valid.
    """
    if not 0 <= row < max_size_row:
        return False
    if not 0 <= col < max_size_col:
        return False
    return True


def _check_distance(distance: int) -> None:
    """Checks if the distance is valid.

    Parameters
    ----------
    distance
        The distance of the code.

    Raises
    ------
    ValueError
        If the distance is not a positive integer.
    """
    if not isinstance(distance, int):
        raise ValueError("distance provided must be an integer")
    if distance < 0:
        raise ValueError("distance must be a positive integer")
