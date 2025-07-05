from ..layout import Layout
from .rot_surface_codes import rot_surface_code_rectangle


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
    if stab_type == "x_type":
        distance_x, distance_z = 1, distance
    elif stab_type == "z_type":
        distance_x, distance_z = distance, 1
    else:
        raise ValueError(
            f"'stab_type' must be 'x_type' or 'z_type', but '{stab_type}' was given."
        )

    return rot_surface_code_rectangle(
        distance_x=distance_x,
        distance_z=distance_z,
        logical_qubit_label=logical_qubit_label,
        init_point=init_point,
        init_data_qubit_id=init_data_qubit_id,
        init_zanc_qubit_id=init_zanc_qubit_id,
        init_xanc_qubit_id=init_xanc_qubit_id,
        init_ind=init_ind,
        init_logical_ind=init_logical_ind,
    )
