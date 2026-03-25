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


def is_valid(col: int, row: int, max_size_col: int, max_size_row: int) -> bool:
    """Checks if a row and column are valid for a grid of a given size.

    Parameters
    ----------
    col
        The column.
    row
        The row.
    max_size_col
        The column size of the grid.
    max_size_row
        The row size of the grid.

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


def check_distance(distance: int) -> None:
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


def set_missing_neighbours_to_none(
    neighbor_data: dict[str, dict[str, str | None]],
) -> None:
    """
    Adds ``None`` for missing neighbours in the neighbor data for
    surface codes. Note that this modifies the dictionary in place.

    Parameters
    ----------
    neighbor_data
        The neighbor data - a dictionary that contains information
        about the layout connectivity (defined on a square grid,
        connection run diagonally).
    """
    directions = ["north_east", "north_west", "south_east", "south_west"]
    for neighbors in neighbor_data.values():
        for direction in directions:
            if direction not in neighbors:
                neighbors[direction] = None
    return
