from collections import defaultdict
from functools import partial
from itertools import count, product

from ..layout import Layout
from .rot_surface_codes import (
    rot_surface_stability_rectangle,
    get_data_index,
    shift_direction,
)
from .util import is_valid, invert_shift, check_distance, set_missing_neighbours_to_none


def repetition_code(
    distance: int,
    stab_type: str,
    logical_qubit_label: str = "L0",
    init_point: tuple[int | float, int | float] = (1, 1),
    init_data_qubit_id: int = 1,
    init_zanc_qubit_id: int = 1,
    init_xanc_qubit_id: int = 1,
    init_ind: int = 0,
    init_logical_ind: int = 0,
) -> Layout:
    """Generates a repetition code layout.

    Parameters
    ----------
    distance
        The distance of the code.
    stab_type
        Stabilizer type for the repetition code: ``"x_type"`` or ``"z_type"``.
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
    check_distance(distance)
    if stab_type == "x_type":
        distance_x, distance_z = 1, distance
    elif stab_type == "z_type":
        distance_x, distance_z = distance, 1
    else:
        raise ValueError(
            f"'stab_type' must be 'x_type' or 'z_type', but '{stab_type}' was given."
        )
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

    name = f"d-{distance} repetition code layout."
    code = "repetition_code"
    description = ""

    int_order = dict(steps=["left", "right"])

    log_op = [f"D{i+init_data_qubit_id}" for i in range(distance)]
    log_other_op = [f"D{init_data_qubit_id}"]
    if stab_type == "x_type":
        log_x, log_z = log_other_op, log_op
    else:
        log_x, log_z = log_op, log_other_op
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
        distance=distance,
        interaction_order=int_order,
    )

    col_size = 2 * distance + 1
    row_size = 2 * 1 + 1
    data_indexer = partial(get_data_index, row_size=1, start_ind=init_data_qubit_id)
    valid_coord = partial(is_valid, max_size_col=col_size, max_size_row=row_size)

    pos_shifts = (1, -1)
    nbr_shifts = tuple(product(pos_shifts, repeat=2))

    layout_data = []
    neighbor_data = defaultdict(dict)
    ind = init_ind

    # change initial point because by default the code places the "D1" qubit
    # in the (1,1) point.
    init_point = (init_point[0] - 1, init_point[1] - 1)

    for col in range(1, col_size, 2):
        index = data_indexer(col, 1)

        qubit_info = dict(
            qubit=f"D{index}",
            role="data",
            coords=[col + init_point[0], 1 + init_point[1]],
            stab_type=None,
            ind=ind,
        )
        layout_data.append(qubit_info)

        ind += 1

    s_index = count(start=init_zanc_qubit_id)
    for col in range(2, col_size - 1, 2):
        for row in range(col % 4, row_size, 4):
            anc_qubit = f"Z{next(s_index)}"
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

    set_missing_neighbours_to_none(neighbor_data)

    for qubit_info in layout_data:
        qubit = qubit_info["qubit"]
        qubit_info["neighbors"] = neighbor_data[qubit]

    layout_setup["layout"] = layout_data
    layout = Layout(layout_setup)
    return layout

    return layout


def repetition_stability(
    stab_type: str,
    num_stabs: int,
    observable: str = "O0",
    init_point: tuple[int | float, int | float] = (1, 1),
    init_data_qubit_id: int = 1,
    init_zanc_qubit_id: int = 1,
    init_xanc_qubit_id: int = 1,
    init_ind: int = 0,
) -> Layout:
    """
    Generates a repetition layout for stability experiments.

    Parameters
    ----------
    stab_type
        Stabilizer type for the repetition layout: ``"x_type"`` or ``"z_type"``.
    num_stabs
        Number of stabilizers that this layout will contain. Note that all stabilizers
        will be of type ``stab_type``.
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

    Returns
    -------
    Layout
        The layout.
    """
    return rot_surface_stability_rectangle(
        stab_type=stab_type,
        width=num_stabs - 1,
        height=1,
        observable=observable,
        init_point=init_point,
        init_data_qubit_id=init_data_qubit_id,
        init_zanc_qubit_id=init_zanc_qubit_id,
        init_xanc_qubit_id=init_xanc_qubit_id,
        init_ind=init_ind,
    )
