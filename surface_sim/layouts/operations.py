from .layout import Layout


def check_overlap_layout_pair(layout_1: Layout, layout_2: Layout) -> None:
    """Checks if the two given layouts share any qubits for when doing
    parallel logical computation with both of them.

    It checks that there are no shared qubit (1) labels, (2) indices,
    (3) coordinates, and (4) logical qubit labels.

    Parameters
    ----------
    layout_1
        One of the layouts.
    layout_2
        The other layout.
    """
    qubits_1 = set(layout_1.get_qubits())
    qubits_2 = set(layout_2.get_qubits())
    if qubits_1.intersection(qubits_2) != set():
        raise ValueError("The layouts have qubits with the same label.")

    inds_1 = set(layout_1.get_inds(qubits_1))
    inds_2 = set(layout_2.get_inds(qubits_2))
    if inds_1.intersection(inds_2) != set():
        raise ValueError("The layouts have qubits with the same indices.")

    coords_1 = set(map(tuple, layout_1.get_coords(qubits_1)))
    coords_2 = set(map(tuple, layout_2.get_coords(qubits_2)))
    if coords_1.intersection(coords_2) != set():
        raise ValueError("The layouts have qubits with the same coordinates.")

    log_qubits_1 = set(layout_1.get_logical_qubits())
    log_qubits_2 = set(layout_2.get_logical_qubits())
    if log_qubits_1.intersection(log_qubits_2) != set():
        raise ValueError("The layouts have logical qubits with the same label.")

    return


def check_overlap_layouts(*layouts: Layout) -> None:
    """Checks if the given layouts share any qubits for when doing
    parallel logical computation with them.

    It checks that there are no shared qubit (1) labels, (2) indices,
    (3) coordinates, and (4) logical qubit labels.

    Parameters
    ----------
    *layouts
        Layouts.
    """
    if len(layouts) == 1:
        return

    for k, layout in enumerate(layouts):
        for other_layout in layouts[k + 1 :]:
            check_overlap_layout_pair(layout, other_layout)

    return
