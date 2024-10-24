from collections import defaultdict
from functools import partial
from itertools import count

from ..layout import Layout
from .util import is_valid, invert_shift, _check_distance


def get_data_index(row: int, col: int, col_size: int, start_ind: int = 1) -> int:
    """Converts row and column to data qubit index.

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
    if row % 2 == 1:
        row -= 1
        col += col_size

    row_ind = row // 2
    col_ind = col // 2
    index = start_ind + row_ind * col_size + col_ind
    return index


def shift_direction(shift: tuple[int, int]) -> str:
    """Translates a row and column shift to a direction.

    Parameters
    ----------
    row_shift
        The row shift.
    col_shift
        The column shift.

    Returns
    -------
    str
        The direction.
    """
    if shift == (0, 1):
        return "south_east"
    elif shift == (0, -1):
        return "north_west"
    elif shift == (1, 0):
        return "north_east"
    elif shift == (-1, 0):
        return "south_west"
    else:
        raise ValueError("The shift does not correspond to a known direction.")


def unrot_surf_code_rectangle(
    distance_x: int, distance_z: int, logical_qubit_label: str = "L0"
) -> Layout:
    """Generates a rotated surface code layout.

    Parameters
    ----------
    distance_x
        The logical X distance of the code.
    distance_z
        The logical Z distance of the code.
    logical_qubit_label
        Label for the logical qubit, by default ``"L0"``.

    Returns
    -------
    Layout
        The layout of the code.
    """
    _check_distance(distance_x)
    _check_distance(distance_z)

    name = f"Rotated dx-{distance_x} dz-{distance_z} surface code layout."
    code = "unrotated_surface_code"
    description = None

    int_order = dict(
        x_type=["north_east", "north_west", "south_east", "south_west"],
        z_type=["north_east", "south_east", "north_west", "south_west"],
    )

    log_z = [f"D{i+1}" for i in range(distance_z)]
    log_x = [f"D{i*(2*distance_z - 1)+1}" for i in range(distance_x)]

    layout_setup = dict(
        name=name,
        code=code,
        logical_qubit_labels=[logical_qubit_label],
        description=description,
        distance_x=distance_x,
        distance_z=distance_z,
        interaction_order=int_order,
        log_z={logical_qubit_label: log_z},
        log_x={logical_qubit_label: log_x},
    )
    if distance_x == distance_z:
        layout_setup["distance"] = distance_z

    col_size = 2 * distance_z - 1
    row_size = 2 * distance_x - 1
    data_indexer = partial(get_data_index, col_size=col_size, start_ind=1)
    valid_coord = partial(is_valid, max_size_col=col_size, max_size_row=row_size)

    nbr_shifts = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    layout_data = []
    neighbor_data = defaultdict(dict)

    x_index = count(start=1)
    z_index = count(start=1)
    for row in range(row_size):
        for col in range(col_size):
            role = "data" if (row + col) % 2 == 0 else "anc"

            if role == "data":
                index = data_indexer(row, col)

                qubit_info = dict(
                    qubit=f"D{index}",
                    role="data",
                    coords=(row, col),
                    stab_type=None,
                )
                layout_data.append(qubit_info)

            else:
                stab_type = "x_type" if row % 2 == 0 else "z_type"
                if stab_type == "x_type":
                    anc_qubit = f"X{next(x_index)}"
                else:
                    anc_qubit = f"Z{next(z_index)}"

                qubit_info = dict(
                    qubit=anc_qubit,
                    role="anc",
                    coords=(row, col),
                    stab_type=stab_type,
                )
                layout_data.append(qubit_info)

                for row_shift, col_shift in nbr_shifts:
                    data_row, data_col = row + row_shift, col + col_shift
                    if not valid_coord(data_row, data_col):
                        continue
                    data_index = data_indexer(data_row, data_col)
                    data_qubit = f"D{data_index}"

                    direction = shift_direction((row_shift, col_shift))
                    neighbor_data[anc_qubit][direction] = data_qubit

                    inv_shifts = invert_shift(row_shift, col_shift)
                    inv_direction = shift_direction(inv_shifts)
                    neighbor_data[data_qubit][inv_direction] = anc_qubit

    for qubit_info in layout_data:
        qubit = qubit_info["qubit"]
        qubit_info["neighbors"] = neighbor_data[qubit]

    layout_setup["layout"] = layout_data
    layout = Layout(layout_setup)
    return layout


def unrot_surf_code(distance: int) -> Layout:
    """Generates an unrotated surface code layout.

    Parameters
    ----------
    distance
        The distance of the code.

    Returns
    -------
    Layout
        The layout of the code.
    """
    return unrot_surf_code_rectangle(distance_x=distance, distance_z=distance)
