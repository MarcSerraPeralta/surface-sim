from collections.abc import Sequence, Mapping
from collections import defaultdict
from functools import partial
from itertools import count, product

from ..layout import Layout, QubitDict
from .util import is_valid, invert_shift, check_distance, set_missing_neighbours_to_none
from ...log_gates.rot_surface_code_css import (
    set_fold_trans_s,
    set_x,
    set_z,
    set_idle,
    set_trans_cnot,
    set_trans_cnot_mid_cycle_css,
)


DEFAULT_INTERACTION_ORDER = dict(
    x_type=["north_east", "north_west", "south_east", "south_west"],
    z_type=["north_east", "south_east", "north_west", "south_west"],
)


def get_data_index(col: int, row: int, row_size: int, start_ind: int = 1) -> int:
    """Converts row and column to data qubit index.

    The data qubits are numbered starting from the bottom-left data qubit,
    and increasing the index by 1 on the horizontal direction.
    Assumes the initial index is 1 (as opposed to 0).

    Parameters
    ----------
    col
        The column of the data qubit.
    row
        The row of the data qubit.
    row_size
        Row size of the code.
    start_ind
        The starting index for the data qubits, by default 1.

    Returns
    -------
    int
        The index of the data qubit.
    """
    row_ind = row // 2
    col_ind = col // 2
    index = start_ind + (col_ind * row_size) + row_ind
    return index


def shift_direction(col_shift: int, row_shift: int) -> str:
    """Translates a row and column shift to a direction.

    Parameters
    ----------
    col_shift
        The column shift.
    row_shift
        The row shift.

    Returns
    -------
    str
        The direction.
    """
    ver_direction = "north" if row_shift > 0 else "south"
    hor_direction = "east" if col_shift > 0 else "west"
    direction = f"{ver_direction}_{hor_direction}"
    return direction


def rot_surface_code_rectangle(
    distance_x: int,
    distance_z: int,
    logical_qubit_label: str = "L0",
    init_point: tuple[int | float, int | float] = (1, 1),
    init_data_qubit_id: int = 1,
    init_zanc_qubit_id: int = 1,
    init_xanc_qubit_id: int = 1,
    init_ind: int = 0,
    init_logical_ind: int = 0,
    interaction_order: Mapping[str, Sequence[str]] = DEFAULT_INTERACTION_ORDER,
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
    init_point
        Coordinates for the bottom left (i.e. southest west) data qubit.
        By default ``(1, 1)``.
    init_data_qubit_id
        Index for the bottom left (i.e. southest west) data qubit.
        By default ``1``, so the label is ``"D1"``.
    init_zanc_qubit_id
        Index for the bottom left (i.e. southest west) Z-type ancilla qubit.
        By default ``1``, so the label is ``"Z1"``.
    init_xanc_qubit_id
        Index for the bottom left (i.e. southest west) X-type ancilla qubit.
        By default ``1``, so the label is ``"X1"``.
    init_ind
        Minimum index that is going to be associated to a qubit.
    init_logical_ind
        Minimum index that is going to be associated to a logical qubit.
    interaction_order
        Dictionary specifying the interaction order for the ``x_type`` and
        ``z_type`` stabilizers. The possible interaction directions are:
        ``"north_east"``, ``"north_west"``, ``"south_east"``, and ``"south_west"``.

    Returns
    -------
    Layout
        The layout of the code.
    """
    check_distance(distance_x)
    check_distance(distance_z)
    if not isinstance(init_point, tuple):
        raise TypeError(
            f"'init_point' must be a tuple, but {type(init_point)} was given."
        )
    if (len(init_point) != 2) or any(
        not isinstance(p, (float, int)) for p in init_point
    ):
        raise TypeError(f"'init_point' must have two elements that are floats or ints.")
    if not isinstance(logical_qubit_label, str):
        raise TypeError(
            "'logical_qubit_label' must be a string, "
            f"but {type(logical_qubit_label)} was given."
        )
    if not isinstance(init_data_qubit_id, int):
        raise TypeError(
            "'init_data_qubit_id' must be an int, "
            f"but {type(init_data_qubit_id)} was given."
        )
    if not isinstance(init_zanc_qubit_id, int):
        raise TypeError(
            "'init_zanc_qubit_id' must be an int, "
            f"but {type(init_zanc_qubit_id)} was given."
        )
    if not isinstance(init_xanc_qubit_id, int):
        raise TypeError(
            "'init_xanc_qubit_id' must be an int, "
            f"but {type(init_xanc_qubit_id)} was given."
        )

    name = f"Rotated dx-{distance_x} dz-{distance_z} surface code layout."
    code = "rotated_surface_code"
    description = ""

    log_z = [f"D{i + init_data_qubit_id}" for i in range(distance_z)]
    log_x = [f"D{i * distance_z + init_data_qubit_id}" for i in range(distance_x)]
    logical_qubits = {
        logical_qubit_label: dict(log_x=log_x, log_z=log_z, ind=init_logical_ind)
    }

    layout_setup = dict(
        name=name,
        code=code,
        logical_qubits=logical_qubits,
        description=description,
        distance_x=distance_x,
        distance_z=distance_z,
        interaction_order=interaction_order,
    )
    if distance_x == distance_z:
        layout_setup["distance"] = distance_z

    col_size = 2 * distance_x + 1
    row_size = 2 * distance_z + 1
    data_indexer = partial(
        get_data_index, row_size=distance_z, start_ind=init_data_qubit_id
    )
    valid_coord = partial(is_valid, max_size_col=col_size, max_size_row=row_size)

    pos_shifts = (1, -1)
    nbr_shifts = tuple(product(pos_shifts, repeat=2))

    layout_data: list[QubitDict] = []
    neighbor_data: defaultdict[str, dict[str, str | None]] = defaultdict(dict)
    ind = init_ind

    # change initial point because by default the code places the "D1" qubit
    # in the (1,1) point.
    init_point = (init_point[0] - 1, init_point[1] - 1)

    for col in range(1, col_size, 2):
        for row in range(1, row_size, 2):
            index = data_indexer(col, row)

            qubit_info = dict(
                qubit=f"D{index}",
                role="data",
                coords=[col + init_point[0], row + init_point[1]],
                stab_type=None,
                ind=ind,
            )
            layout_data.append(qubit_info)

            ind += 1

    x_index = count(start=init_xanc_qubit_id)
    for col in range(0, col_size, 2):
        for row in range(2 + col % 4, row_size - 1, 4):
            anc_qubit = f"X{next(x_index)}"
            qubit_info = dict(
                qubit=anc_qubit,
                role="anc",
                coords=[col + init_point[0], row + init_point[1]],
                stab_type="x_type",
                ind=ind,
            )
            layout_data.append(qubit_info)

            ind += 1

            for col_shift, row_shift in nbr_shifts:
                data_row, data_col = row + row_shift, col + col_shift
                if not valid_coord(data_col, data_row):
                    continue
                data_index = data_indexer(data_col, data_row)
                data_qubit = f"D{data_index}"

                direction = shift_direction(col_shift, row_shift)
                neighbor_data[anc_qubit][direction] = data_qubit

                inv_shifts = invert_shift(col_shift, row_shift)
                inv_direction = shift_direction(*inv_shifts)
                neighbor_data[data_qubit][inv_direction] = anc_qubit

    z_index = count(start=init_zanc_qubit_id)
    for col in range(2, col_size - 1, 2):
        for row in range(col % 4, row_size, 4):
            anc_qubit = f"Z{next(z_index)}"
            qubit_info = dict(
                qubit=anc_qubit,
                role="anc",
                coords=[col + init_point[0], row + init_point[1]],
                stab_type="z_type",
                ind=ind,
            )
            layout_data.append(qubit_info)

            ind += 1

            for col_shift, row_shift in nbr_shifts:
                data_row, data_col = row + row_shift, col + col_shift
                if not valid_coord(data_col, data_row):
                    continue
                data_index = data_indexer(data_col, data_row)
                data_qubit = f"D{data_index}"

                direction = shift_direction(col_shift, row_shift)
                neighbor_data[anc_qubit][direction] = data_qubit

                inv_shifts = invert_shift(col_shift, row_shift)
                inv_direction = shift_direction(*inv_shifts)
                neighbor_data[data_qubit][inv_direction] = anc_qubit

    set_missing_neighbours_to_none(neighbor_data)

    for qubit_info in layout_data:
        qubit = qubit_info["qubit"]
        qubit_info["neighbors"] = neighbor_data[qubit]

    layout_setup["layout"] = layout_data
    layout = Layout(layout_setup)
    return layout


def rot_surface_code(
    distance: int,
    logical_qubit_label: str = "L0",
    init_point: tuple[int | float, int | float] = (1, 1),
    init_data_qubit_id: int = 1,
    init_zanc_qubit_id: int = 1,
    init_xanc_qubit_id: int = 1,
    init_ind: int = 0,
    init_logical_ind: int = 0,
) -> Layout:
    """Generates a rotated surface code layout.

    Parameters
    ----------
    distance
        The distance of the code.
    logical_qubit_label
        Label for the logical qubit, by default ``"L0"``.
    init_point
        Coordinates for the bottom left (i.e. southest west) data qubit.
        By default ``1``, so the label is ``"D1"``.
    init_data_qubit_id
        Index for the bottom left (i.e. southest west) data qubit.
        By default ``1``, so the label is ``"D1"``.
    init_zanc_qubit_id
        Index for the bottom left (i.e. southest west) Z-type ancilla qubit.
        By default ``1``, so the label is ``"Z1"``.
    init_xanc_qubit_id
        Index for the bottom left (i.e. southest west) X-type ancilla qubit.
        By default ``1``, so the label is ``"X1"``.
    init_ind
        Minimum index that is going to be associated to a qubit.
    init_logical_ind
        Minimum index that is going to be associated to a logical qubit.

    Returns
    -------
    Layout
        The layout of the code.
    """
    return rot_surface_code_rectangle(
        distance_x=distance,
        distance_z=distance,
        logical_qubit_label=logical_qubit_label,
        init_point=init_point,
        init_data_qubit_id=init_data_qubit_id,
        init_zanc_qubit_id=init_zanc_qubit_id,
        init_xanc_qubit_id=init_xanc_qubit_id,
        init_ind=init_ind,
        init_logical_ind=init_logical_ind,
    )


def rot_surface_codes(num_layouts: int, distance: int) -> list[Layout]:
    """
    Returns a list of (square) rotated surface codes of the specified distance that
    are set up to be used in any logical circuit (i.e. they have all the
    implemented logical gate attributes).

    Parameters
    ----------
    num_layouts
        Number of layouts to generate.
    distance
        The code distance.

    Returns
    -------
    layouts
        List of layouts.
    """
    if not isinstance(num_layouts, int):
        raise TypeError(
            f"'num_layouts' must be an int, but {type(num_layouts)} was given."
        )

    layouts: list[Layout] = []
    num_data = distance**2
    num_anc = num_data - 1
    for k in range(num_layouts):
        layout = rot_surface_code(
            distance=distance,
            logical_qubit_label=f"L{k}",
            init_point=(0, (2 * distance + 2) * k),
            init_data_qubit_id=1 + k * num_data,
            init_zanc_qubit_id=1 + k * num_anc // 2,
            init_xanc_qubit_id=1 + k * num_anc // 2,
            init_ind=k * (num_data + num_anc),
            init_logical_ind=k,
        )
        layouts.append(layout)

    # set up the parameters for all the logical gates
    for k, layout in enumerate(layouts):
        set_x(layout)
        set_z(layout)
        set_idle(layout)
        for other_layout in layouts:
            if layout == other_layout:
                continue
            set_trans_cnot(layout, other_layout)
            set_trans_cnot_mid_cycle_css(layout, other_layout)

    return layouts


def rot_surface_code_rectangles(num_layouts: int, distance: int) -> list[Layout]:
    """
    Returns a list of rotated surface codes of the specified distance that
    are set up to be used in any logical circuit (i.e. they have all the
    implemented logical gate attributes).

    Parameters
    ----------
    num_layouts
        Number of layouts to generate.
    distance
        The distance of logical Pauli X of the layouts. The distance of logical
        Pauli Z is ``distance + 1``.

    Returns
    -------
    layouts
        List of layouts.
    """
    if not isinstance(num_layouts, int):
        raise TypeError(
            f"'num_layouts' must be an int, but {type(num_layouts)} was given."
        )

    layouts: list[Layout] = []
    num_data = (distance + 1) * distance
    num_anc = num_data - 1
    for k in range(num_layouts):
        layout = rot_surface_code_rectangle(
            distance_x=distance,
            distance_z=distance + 1,
            logical_qubit_label=f"L{k}",
            init_point=(0, (2 * distance + 4) * k),
            init_data_qubit_id=1 + k * num_data,
            init_zanc_qubit_id=1 + k * num_anc // 2,
            init_xanc_qubit_id=1 + k * (num_anc // 2 + 1),
            init_ind=k * (num_data + num_anc),
            init_logical_ind=k,
        )
        layouts.append(layout)

    # set up the parameters for all the logical gates
    for k, layout in enumerate(layouts):
        set_fold_trans_s(layout, data_qubit=f"D{1 + k * num_data}")
        set_x(layout)
        set_z(layout)
        set_idle(layout)
        for other_layout in layouts:
            if layout == other_layout:
                continue
            set_trans_cnot(layout, other_layout)

    return layouts


def rot_surface_stability_rectangle(
    stab_type: str,
    width: int,
    height: int,
    observable: str = "O0",
    init_point: tuple[int | float, int | float] = (1, 1),
    init_data_qubit_id: int = 1,
    init_zanc_qubit_id: int = 1,
    init_xanc_qubit_id: int = 1,
    init_ind: int = 0,
    interaction_order: Mapping[str, Sequence[str]] = DEFAULT_INTERACTION_ORDER,
) -> Layout:
    """
    Generates a rotated surface layout for stability experiments.

    Parameters
    ----------
    stab_type
        Type of the stabilizer that lead to the identity when multiplying all
        stabilizers of that type. It must be ``"x_type"`` or ``"z_type"``.
    width
        Width of the rotated_surface code layout in terms of the number
        of data qubit along the horizontal dimension.
    height
        Heigh of the rotated_surface code layout in terms of the number
        of data qubit along the vertical dimension.
    observable
        Label for the observable, by default ``"L0"``.
    init_point
        Coordinates for the bottom left (i.e. southest west) data qubit.
        By default ``(1, 1)``.
    init_data_qubit_id
        Index for the bottom left (i.e. southest west) data qubit.
        By default ``1``, so the label is ``"D1"``.
    init_zanc_qubit_id
        Index for the bottom left (i.e. southest west) Z-type ancilla qubit.
        By default ``1``, so the label is ``"Z1"``.
    init_xanc_qubit_id
        Index for the bottom left (i.e. southest west) X-type ancilla qubit.
        By default ``1``, so the label is ``"X1"``.
    init_ind
        Minimum index that is going to be associated to a qubit.
    interaction_order
        Dictionary specifying the interaction order for the ``x_type`` and
        ``z_type`` stabilizers. The possible interaction directions are:
        ``"north_east"``, ``"north_west"``, ``"south_east"``, and ``"south_west"``.

    Returns
    -------
    Layout
        The layout.
    """
    check_distance(width)
    check_distance(height)
    if stab_type not in ("x_type", "z_type"):
        raise ValueError(
            f"'stab_type' must be 'x_type' or 'z_type', but {stab_type} was given."
        )
    if not isinstance(init_point, tuple):
        raise TypeError(
            f"'init_point' must be a tuple, but {type(init_point)} was given."
        )
    if (len(init_point) != 2) or any(
        not isinstance(p, (float, int)) for p in init_point
    ):
        raise TypeError(f"'init_point' must have two elements that are floats or ints.")
    if not isinstance(observable, str):
        raise TypeError(
            f"'observable' must be a string, but {type(observable)} was given."
        )
    if not isinstance(init_data_qubit_id, int):
        raise TypeError(
            "'init_data_qubit_id' must be an int, "
            f"but {type(init_data_qubit_id)} was given."
        )
    if not isinstance(init_zanc_qubit_id, int):
        raise TypeError(
            "'init_zanc_qubit_id' must be an int, "
            f"but {type(init_zanc_qubit_id)} was given."
        )
    if not isinstance(init_xanc_qubit_id, int):
        raise TypeError(
            "'init_xanc_qubit_id' must be an int, "
            f"but {type(init_xanc_qubit_id)} was given."
        )

    name = f"Rotated w-{width} h-{height} surface layout for stability experiments."
    code = "rotated_surface_stability"
    description = ""

    col_size = 2 * width
    row_size = 2 * height
    data_indexer = partial(
        get_data_index, row_size=height, start_ind=init_data_qubit_id
    )
    valid_coord = partial(is_valid, max_size_col=col_size, max_size_row=row_size)

    pos_shifts = (1, -1)
    nbr_shifts = tuple(product(pos_shifts, repeat=2))

    layout_data: list[QubitDict] = []
    neighbor_data: defaultdict[str, dict[str, str | None]] = defaultdict(dict)
    ind = init_ind

    # change initial point because by default the code places the "D1" qubit
    # in the (1,1) point.
    init_point = (init_point[0] - 1, init_point[1] - 1)

    for row in range(1, row_size, 2):
        for col in range(1, col_size, 2):
            index = data_indexer(col, row)

            qubit_info = dict(
                qubit=f"D{index}",
                role="data",
                coords=[col + init_point[0], row + init_point[1]],
                stab_type=None,
                ind=ind,
            )
            layout_data.append(qubit_info)

            ind += 1

    if stab_type == "x_type":
        init_sanc_qubit_id = init_xanc_qubit_id
        init_oanc_qubit_id = init_zanc_qubit_id
        other_stab_type = "z_type"
        s, o = "X", "Z"
    else:
        init_sanc_qubit_id = init_zanc_qubit_id
        init_oanc_qubit_id = init_xanc_qubit_id
        other_stab_type = "x_type"
        s, o = "Z", "X"

    s_index = count(start=init_sanc_qubit_id)
    for row in range(0, row_size + 1, 2):
        for col in range((2 + row) % 4, col_size + 1, 4):
            anc_qubit = f"{s}{next(s_index)}"
            qubit_info = dict(
                qubit=anc_qubit,
                role="anc",
                coords=[col + init_point[0], row + init_point[1]],
                stab_type=stab_type,
                ind=ind,
            )
            layout_data.append(qubit_info)

            ind += 1

            for col_shift, row_shift in nbr_shifts:
                data_row, data_col = row + row_shift, col + col_shift
                if not valid_coord(data_col, data_row):
                    continue
                data_index = data_indexer(data_col, data_row)
                data_qubit = f"D{data_index}"

                direction = shift_direction(col_shift, row_shift)
                neighbor_data[anc_qubit][direction] = data_qubit

                inv_shifts = invert_shift(col_shift, row_shift)
                inv_direction = shift_direction(*inv_shifts)
                neighbor_data[data_qubit][inv_direction] = anc_qubit

    o_index = count(start=init_oanc_qubit_id)
    for row in range(2, row_size - 1, 2):
        for col in range(2 + (2 + row) % 4, col_size, 4):
            anc_qubit = f"{o}{next(o_index)}"
            qubit_info = dict(
                qubit=anc_qubit,
                role="anc",
                coords=[col + init_point[0], row + init_point[1]],
                stab_type=other_stab_type,
                ind=ind,
            )
            layout_data.append(qubit_info)

            ind += 1

            for col_shift, row_shift in nbr_shifts:
                data_row, data_col = row + row_shift, col + col_shift
                if not valid_coord(data_col, data_row):
                    continue
                data_index = data_indexer(data_col, data_row)
                data_qubit = f"D{data_index}"

                direction = shift_direction(col_shift, row_shift)
                neighbor_data[anc_qubit][direction] = data_qubit

                inv_shifts = invert_shift(col_shift, row_shift)
                inv_direction = shift_direction(*inv_shifts)
                neighbor_data[data_qubit][inv_direction] = anc_qubit

    set_missing_neighbours_to_none(neighbor_data)

    anc_redundant_stab_type: list[str] = []
    for qubit_info in layout_data:
        qubit = qubit_info["qubit"]
        qubit_info["neighbors"] = neighbor_data[qubit]

        if qubit_info["role"] == "anc" and qubit_info["stab_type"] == stab_type:
            anc_redundant_stab_type.append(qubit)

    layout_setup = dict(
        name=name,
        code=code,
        observables={observable: anc_redundant_stab_type},
        description=description,
        interaction_order=interaction_order,
    )
    layout_setup["layout"] = layout_data

    layout = Layout(layout_setup)
    return layout
