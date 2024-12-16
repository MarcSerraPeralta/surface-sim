from collections.abc import Callable, Sequence, Iterable

import stim

from ..util.circuit_operations import merge_ops, merge_qec_rounds
from ..layouts.layout import Layout
from ..models.model import Model
from ..detectors.detectors import Detectors
from ..circuit_blocks.util import qubit_coords

SCHEDULE = list[
    tuple[Callable] | tuple[Callable, Layout] | tuple[Callable, Layout, Layout]
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
        List of operations to be applied to a single qubit or pair of qubits.
        See Notes for more information about the format.

    Notes
    -----
    The format of the schedule is the following. Each element of the list
    is an operation to be applied to the qubits:
    - ``tuple[Callable]`` performs a QEC cycle to all layouts
    - ``tuple[Callable, Layout]`` performs a single-qubit operation
    - ``tuple[Callable, Layout, Layout]`` performs a two-qubit gate.

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
            (reset_z_iterator, layout_0),
            (reset_z_iterator, layout_1),
            (qec_round_iterator,),
            (log_x_iterator, layout_1),
            (log_meas_z, layout_0),
            (qec_round_iterator,),
        ]

    """
    if not isinstance(circuit, stim.Circuit):
        raise TypeError(
            f"'circuit' must be a stim.Circuit, but {type(circuit)} was given."
        )
    circuit = circuit.flattened()
    if not isinstance(layouts, Sequence):
        raise TypeError(f"'layouts' must be a list, but {type(layouts)} was given.")
    if circuit.num_qubits > len(layouts):
        raise ValueError("There are more qubits in the circuit than in 'layouts'.")
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

    schedule = []
    for instr in circuit:
        if instr.name == "TICK":
            schedule.append((gate_to_iterator["TICK"],))
            continue

        func_iter = gate_to_iterator[instr.name]
        targets = [t.value for t in instr.targets_copy()]

        if func_iter.log_op_type == "tq_unitary_gate":
            for i, j in _grouper(targets, 2):
                schedule.append((func_iter, layouts[i], layouts[j]))
        else:
            for i in targets:
                schedule.append((func_iter, layouts[i]))

    return schedule


def experiment_from_schedule(
    schedule: SCHEDULE,
    model: Model,
    detectors: Detectors,
    anc_reset: bool = True,
    anc_detectors: Sequence[str] | None = None,
    ensure_idling: bool = True,
) -> stim.Circuit:
    """
    Returns a stim circuit corresponding to a logical experiment
    corresponding to the given schedule.

    Parameters
    ----------
    schedule
        List of operations to be applied to a single qubit or pair of qubits.
        See Notes of ``schedule_from_circuit`` for more information about the format.
    model
        Noise model for the gates.
    detectors
        Object to build the detectors.
    anc_reset
        If ``True``, ancillas are reset at the beginning of the QEC cycle.
        By default ``True``.
    anc_detectors
        List of ancilla qubits for which to define the detectors.
        If ``None``, adds all detectors.
        By default ``None``.
    ensure_idling
        Flag to check that all active layouts are participating in one operation
        in each time slice. This ensures that idling time has not been forgotten
        when building the schedule.
        By default ``True``.


    Returns
    -------
    experiment
        Stim circuit corresponding to the logical equivalent of the
        given schedule.
    """
    layouts = set()
    for op in schedule:
        layouts.update(set(op[1:]))

    experiment = stim.Circuit()
    active_layouts = {l: False for l in layouts}
    num_gates = {l: 0 for l in layouts}
    num_log_meas = 0
    log_obs_inds = {}
    curr_block = []

    experiment += qubit_coords(model, *layouts)
    for op in schedule:
        func = op[0]

        if func.log_op_type == "qec_cycle":
            # flush all stored operations in current block
            curr_num_gates = set(n for l, n in num_gates.items() if active_layouts[l])
            if ensure_idling and curr_num_gates != set([1]):
                raise ValueError(
                    "Not all active layouts are participating in an operation. "
                    f"active layouts: {active_layouts}\noperations: {num_gates}"
                )

            experiment += merge_ops(
                ops=curr_block,
                model=model,
                detectors=detectors,
                log_obs_inds=log_obs_inds,
                anc_reset=anc_reset,
                anc_detectors=anc_detectors,
            )
            num_gates = {l: 0 for l in layouts}
            num_log_meas = 0
            log_obs_inds = {}
            curr_block = []

            # run QEC cycle
            experiment += merge_qec_rounds(
                qec_round_iterator=func,
                model=model,
                layouts=[l for l, a in active_layouts.items() if a],
                detectors=detectors,
                anc_reset=anc_reset,
                anc_detectors=anc_detectors,
            )
            continue

        # update the number of gates so that we know if we need to flush the
        # current operations or if we need to store the current one in 'curr_block'
        for l in op[1:]:
            num_gates[l] += 1

        # check for flushing the operations in case a layout would be doing more
        # more than one operation. If not, store current operation in 'curr_block'
        if any(n > 1 for n in num_gates.values()):
            for l in op[1:]:
                num_gates[l] -= 1
            curr_num_gates = set(n for l, n in num_gates.items() if active_layouts[l])
            if ensure_idling and curr_num_gates != set([1]):
                raise ValueError(
                    "Not all active layouts are participating in an operation. "
                    f"active layouts: {active_layouts}\noperations: {num_gates}"
                )
            experiment += merge_ops(
                ops=curr_block,
                model=model,
                detectors=detectors,
                log_obs_inds=log_obs_inds,
                anc_reset=anc_reset,
                anc_detectors=anc_detectors,
            )
            num_gates = {l: 0 for l in layouts}
            num_log_meas = 0
            log_obs_inds = {}
            curr_block = [op]
        else:
            curr_block.append(op)

        if func.log_op_type == "measurement":
            active_layouts[op[1]] = False
            log_obs_inds[op[1].get_logical_qubits()[0]] = num_log_meas
            num_log_meas += 1
        if func.log_op_type == "qubit_init":
            active_layouts[op[1]] = True

    # flush remaining operations
    if len(curr_block) != 0:
        curr_num_gates = set(n for l, n in num_gates.items() if active_layouts[l])
        if ensure_idling and curr_num_gates != set([1]):
            raise ValueError(
                "Not all active layouts are participating in an operation. "
                f"active layouts: {active_layouts}\noperations: {num_gates}"
            )
        experiment += merge_ops(
            ops=curr_block,
            model=model,
            detectors=detectors,
            log_obs_inds=log_obs_inds,
            anc_reset=anc_reset,
            anc_detectors=anc_detectors,
        )

    return experiment


def _grouper(iterable: Iterable, n: int):
    args = [iter(iterable)] * n
    return zip(*args, strict=True)
