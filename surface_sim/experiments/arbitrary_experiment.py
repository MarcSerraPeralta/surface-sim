from collections.abc import Callable, Sequence, Iterable

import numpy as np
import stim

from ..util.circuit_operations import merge_ops, merge_qec_rounds
from ..layouts.layout import Layout

SCHEDULE = list[
    list[tuple[Callable] | tuple[Callable, Layout] | tuple[Callable, Layout, Layout]]
]


def schedule_from_circuit(
    circuit: stim.Circuit, layouts: list[Layout], gate_to_iterator: dict[str, Callable]
) -> SCHEDULE:
    """
    Returns the equivalent schedule from a stim circuit.

    Parameters
    ----------
    circuit
        Stim circuit.
    layouts
        List of layouts whose index match the qubit index in ``circuit``.
        This function only works for layouts that only have one logical qubit.
    gate_to_iterator
        Dictionary mapping the names of stim circuit instructions used in ``circuit``
        to the functions that generate the equivalent logical circuit.
        Note that ``TICK`` always refers to a QEC cycle for all layouts.

    Returns
    -------
    schedule
        List of operations to be applied in a single time slice for all (active)
        layouts. See Notes for more information about the format.

    Notes
    -----
    The format of the schedule is the following. Each element of the list
    is a list of operations to be applied in a single time slice. These operations
    are have the following form:
    - ``tuple[Callable]`` when performing a QEC cycle to all layouts
    - ``tuple[Callable, Layout]`` when performing a single-qubit operation
    - ``tuple[Callable, Layout, Layout]`` when performing a two-qubit gate.

    For example, the following circuit

    .. code:
        R 0 1
        TICK
        X 1
        M 0
        TICK

    is translated to

    .. code:
        [
                [
                    (reset_z_iterator, layout_0),
                    (reset_z_iterator, layout_1),
                ],
                [
                    (qec_round_iterator,),
                ],
                [
                    (log_x_iterator, layout_1),
                    (log_meas_z, layout_0),
                ],
                [
                    (qec_round_iterator,),
                ],
        ]

    """
    if not isinstance(circuit, stim.Circuit):
        raise TypeError(
            f"'circuit' must be a stim.Circuit, but {type(circuit)} was given."
        )
    circuit = circuit.flattened()
    if not isinstance(layouts, Sequence):
        raise TypeError(f"'layouts' must be a list, but {type(layouts)} was given.")
    if any(not isinstance(l, Layout) for l in layouts):
        raise TypeError("All elements in 'layouts' must be a Layout.")
    if not isinstance(gate_to_iterator, dict):
        raise TypeError(
            f"'gate_to_iterator' must be a dict, but {type(gate_to_iterator)} was given."
        )
    if any(not isinstance(f, Callable) for f in gate_to_iterator.values()):
        raise TypeError("All values of 'gate_to_iterator' must be Callable.")
    if any("log_op_type" not in dir(f) for f in gate_to_iterator.values()):
        raise TypeError(
            "All values of 'gate_to_iterator' must be have the 'log_op_type' attribute. "
            "See 'surface_sim.circuit_blocks.decorators' for more information."
        )
    if gate_to_iterator["TICK"].log_op_type != "qec_cycle":
        raise TypeError("'TICK' must correspond to a QEC cycle.")

    unique_names = set(i.name for i in circuit)
    if unique_names > set(gate_to_iterator):
        raise ValueError(
            "Not all operations in 'circuit' are present in 'gate_to_iterator'."
        )

    schedule = [[]]
    for instr in circuit:
        if instr.name == "TICK":
            schedule.append([])
            schedule[-1].append((gate_to_iterator["TICK"],))
            schedule.append([])
            continue

        func_iter = gate_to_iterator[instr.name]
        targets = instr.targets_copy()

        if func_iter.log_op_type == "tq_unitary_gate":
            for i, j in _grouper(targets, 2):
                schedule.append((func_iter, layouts[i], layouts[j]))
        else:
            for i in targets:
                schedule.append((func_iter, layouts[i]))

    return schedule


def experiment_from_schedule(
    schedule: SCHEDULE,
    anc_reset: bool = True,
    anc_detectors: list[str] | None = None,
) -> stim.Circuit:
    """
    TODO
    """
    experiment = stim.Circuit()
    return experiment


def _grouper(iterable: Iterable, n: int):
    args = [iter(iterable)] * n
    return zip(*args, strict=True)
