"""Module implementing layout generator for the surface code."""

from collections import defaultdict
from functools import partial
from itertools import count, cycle, product
from typing import Dict, Tuple

from .layout import Layout


def get_data_index(row: int, col: int, col_size: int, start_ind: int = 1) -> int:
    """
    Converts row and column to data qubit index.

    The data qubits are numbered starting from the bottom-left data qubit,
    and increasing the index by 1 on the horizontal direction.
    Assumes the initial index is 1 (as opposed to 0).

    Parameters
    ----------
    row
        The row of the data qubit.
    col
        The column of the data qubit.
    col_size
        Column size of the code.
    start_ind
        The starting index for the data qubits, by default 1.

    Returns
    -------
    int
        The index of the data qubit.
    """
    row_ind = row // 2
    col_ind = col // 2
    index = start_ind + (row_ind * col_size) + col_ind
    return index


def shift_direction(row_shift: int, col_shift: int) -> str:
    """
    Translates a row and column shift to a direction.

    Parameters
    ----------
    row_shift : int
        The row shift.
    col_shift : int
        The column shift.

    Returns
    -------
    str
        The direction.
    """
    ver_direction = "north" if row_shift > 0 else "south"
    hor_direction = "east" if col_shift > 0 else "west"
    direction = f"{ver_direction}_{hor_direction}"
    return direction


def invert_shift(row_shift: int, col_shift: int) -> Tuple[int, int]:
    """
    Inverts a row and column shift.

    Parameters
    ----------
    row_shift
        The row shift.
    col_shift
        The column shift.

    Returns
    -------
    Tuple[int, int]
        The inverted row and column shift.
    """
    return -row_shift, -col_shift


def is_valid(row: int, col: int, max_size_row: int, max_size_col: int) -> bool:
    """
    Checks if a row and column are valid for a grid of a given size.

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


def add_missing_neighbours(neighbor_data: Dict) -> None:
    """
    Adds None for missing neighbours in the neighbor data.
    Note that this modifies the dictionary in place.

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


def rot_surf_code_rectangle(distance_x: int, distance_z: int) -> Layout:
    """
    Generates a rotated surface code layout.

    Parameters
    ----------
    distance_x
        The logical X distance of the code.
    distance_z
        The logical Z distance of the code.

    Returns
    -------
    Layout
        The layout of the code.
    """
    _check_distance(distance_x)
    _check_distance(distance_z)

    name = f"Rotated dx-{distance_x} dz-{distance_z} surface code layout."
    code = "rotated_surface_code"
    description = None

    freq_order = ["low", "mid", "high"]

    int_order = dict(
        x_type=["north_east", "north_west", "south_east", "south_west"],
        z_type=["north_east", "south_east", "north_west", "south_west"],
    )

    log_z = [f"D{i+1}" for i in range(distance_z)]
    log_x = [f"D{i*distance_z+1}" for i in range(distance_x)]

    layout_setup = dict(
        name=name,
        code=code,
        description=description,
        distance_x=distance_x,
        distance_z=distance_z,
        freq_order=freq_order,
        interaction_order=int_order,
        log_z=log_z,
        log_x=log_x,
    )
    if distance_x == distance_z:
        layout_setup["distance"] = distance_z

    col_size = 2 * distance_z + 1
    row_size = 2 * distance_x + 1
    data_indexer = partial(get_data_index, col_size=distance_z, start_ind=1)
    valid_coord = partial(is_valid, max_size_col=col_size, max_size_row=row_size)

    pos_shifts = (1, -1)
    nbr_shifts = tuple(product(pos_shifts, repeat=2))

    layout_data = []
    neighbor_data = defaultdict(dict)

    freq_seq = cycle(("low", "high"))

    for row in range(1, row_size, 2):
        freq_group = next(freq_seq)
        for col in range(1, col_size, 2):
            index = data_indexer(row, col)

            qubit_info = dict(
                qubit=f"D{index}",
                role="data",
                coords=(row, col),
                freq_group=freq_group,
                stab_type=None,
            )
            layout_data.append(qubit_info)

    x_index = count(1)
    for row in range(0, row_size, 2):
        for col in range(2 + row % 4, col_size - 1, 4):
            anc_qubit = f"X{next(x_index)}"
            qubit_info = dict(
                qubit=anc_qubit,
                role="anc",
                coords=(row, col),
                freq_group="mid",
                stab_type="x_type",
            )
            layout_data.append(qubit_info)

            for row_shift, col_shift in nbr_shifts:
                data_row, data_col = row + row_shift, col + col_shift
                if not valid_coord(data_row, data_col):
                    continue
                data_index = data_indexer(data_row, data_col)
                data_qubit = f"D{data_index}"

                direction = shift_direction(row_shift, col_shift)
                neighbor_data[anc_qubit][direction] = data_qubit

                inv_shifts = invert_shift(row_shift, col_shift)
                inv_direction = shift_direction(*inv_shifts)
                neighbor_data[data_qubit][inv_direction] = anc_qubit

    z_index = count(1)
    for row in range(2, row_size - 1, 2):
        for col in range(row % 4, col_size, 4):
            anc_qubit = f"Z{next(z_index)}"
            qubit_info = dict(
                qubit=anc_qubit,
                role="anc",
                coords=(row, col),
                freq_group="mid",
                stab_type="z_type",
            )
            layout_data.append(qubit_info)

            for row_shift, col_shift in nbr_shifts:
                data_row, data_col = row + row_shift, col + col_shift
                if not valid_coord(data_row, data_col):
                    continue
                data_index = data_indexer(data_row, data_col)
                data_qubit = f"D{data_index}"

                direction = shift_direction(row_shift, col_shift)
                neighbor_data[anc_qubit][direction] = data_qubit

                inv_shifts = invert_shift(row_shift, col_shift)
                inv_direction = shift_direction(*inv_shifts)
                neighbor_data[data_qubit][inv_direction] = anc_qubit

    add_missing_neighbours(neighbor_data)

    for qubit_info in layout_data:
        qubit = qubit_info["qubit"]
        qubit_info["neighbors"] = neighbor_data[qubit]

    layout_setup["layout"] = layout_data
    layout = Layout(layout_setup)
    return layout


def rot_surf_code(distance: int) -> Layout:
    """
    Generates a rotated surface code layout.

    Parameters
    ----------
    distance
        The distance of the code.

    Returns
    -------
    Layout
        The layout of the code.
    """
    return rot_surf_code_rectangle(distance_x=distance, distance_z=distance)


def _check_distance(distance: int) -> None:
    """
    Checks if the distance is valid.

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
