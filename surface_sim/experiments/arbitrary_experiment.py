from collections.abc import Collection, Iterable, Sequence
from itertools import chain
from typing import TypeVar

import stim

from ..circuit_blocks.decorators import LogicalOperation, LogOpCallable
from ..circuit_blocks.util import idle_iterator, qubit_coords
from ..detectors.detectors import Detectors
from ..layouts.layout import Layout
from ..models.model import Model
from ..util.circuit_operations import (
    QEC_OP_TYPES,
    RESET_OP_TYPES,
    group_logical_operations,
    merge_logical_noise,
    merge_logical_operations,
)
from ..util.observables import move_observables_to_end

T = TypeVar("T")
Instructions = list[tuple[LogOpCallable] | LogicalOperation]
Schedule = list[list[LogicalOperation]]


def schedule_from_circuit(
    circuit: stim.Circuit,
    layouts: list[Layout],
    gate_to_iterator: dict[str, LogOpCallable],
) -> Schedule:
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
        Note that ``TICK`` always refers to a QEC round for all layouts.

    Returns
    -------
    schedule
        List of operations to be applied to a single qubit or pair of qubits.
        See Notes for more information about the format.

    Notes
    -----
    The format of the schedule is the following. Each element of the list
    is an operation to be applied to the qubits:
    - ``tuple[LogOpCallable, Layout]`` performs a (logical) single-layout operation
    - ``tuple[LogOpCallable, Layout, Layout]`` performs a (logical) two-qubit gate.

    The OBSERVABLE_INCLUDE instructions are not included.

    For example, the following circuit

    .. code:
        R 0 1
        TICK
        M 1
        X 0
        TICK
        OBSERVABLE_INCLUDE(0) rec[-1]

    is translated to

    .. code:
        [
            [
                (reset_z_iterator, layout_0),
                (reset_z_iterator, layout_1),
            ],
            [
                (qec_round_iterator, layout_0),
                (qec_round_iterator, layout_1),
            ],
            [
                (log_meas_iterator, layout_1),
                (idle_iterator, layout_0),
            ],
            [
                (qec_round_iterator, layout_0),
            ],
        ]

    """
    if not isinstance(circuit, stim.Circuit):
        raise TypeError(
            f"'circuit' must be a stim.Circuit, but {type(circuit)} was given."
        )
    circuit = circuit.flattened()
    if not isinstance(layouts, Collection):
        raise TypeError(f"'layouts' must be a list, but {type(layouts)} was given.")
    if circuit.num_qubits > len(layouts):
        raise ValueError("There are more qubits in the circuit than in 'layouts'.")
    if any(not isinstance(l, Layout) for l in layouts):
        raise TypeError("All elements in 'layouts' must be a Layout.")
    if not isinstance(gate_to_iterator, dict):
        raise TypeError(
            f"'gate_to_iterator' must be a dict, but {type(gate_to_iterator)} was given."
        )
    if any(not isinstance(f, LogOpCallable) for f in gate_to_iterator.values()):
        raise TypeError("All values of 'gate_to_iterator' must be LogOpCallable.")
    if set(["qec_round"]).intersection(gate_to_iterator["TICK"].log_op_type) == set():
        raise TypeError("'TICK' must correspond to a QEC round.")

    unique_names = set(i.name for i in circuit)
    if unique_names > set(gate_to_iterator):
        raise ValueError(
            "Not all operations in 'circuit' are present in 'gate_to_iterator'."
        )

    instructions: Instructions = []
    for instr in circuit:
        if instr.name == "OBSERVABLE_INCLUDE":
            continue
        if instr.name == "TICK":
            instructions.append((gate_to_iterator["TICK"],))
            continue

        func_iter = gate_to_iterator[instr.name]
        targets: list[int] = [t.value for t in instr.targets_copy()]

        if set(["tq_unitary_gate"]).intersection(func_iter.log_op_type):
            for i, j in _grouper(targets, 2):
                instructions.append((func_iter, layouts[i], layouts[j]))
        else:
            for i in targets:
                instructions.append((func_iter, layouts[i]))

    schedule = schedule_from_instructions(instructions)

    return schedule


def schedule_from_mid_cycle_circuit(
    circuit: stim.Circuit,
    layouts: list[Layout],
    gate_to_iterator: dict[str, LogOpCallable],
    tick_iterators: Sequence[LogOpCallable],
) -> Schedule:
    """
    Returns the equivalent schedule from a stim circuit for
    contains mid cycle gates and/or codes.

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
        The value for ``TICK`` will be ommited as they should be specified in
        ``tick_iterators``.
    tick_iterators
        Sequence of (half) QEC round iterators that are going to be used in cyclic
        order when encountering ``TICK`` instructions. Each iterator is applied
        to all the active layouts.

    Returns
    -------
    schedule
        List of operations to be applied to a single qubit or pair of qubits.
        See Notes for more information about the format.

    Notes
    -----
    The format of the schedule is the following. Each element of the list
    is an operation to be applied to the qubits:
    - ``tuple[LogOpCallable, Layout]`` performs a (logical) single-layout operation
    - ``tuple[LogOpCallable, Layout, Layout]`` performs a (logical) two-qubit gate.

    For example, the following circuit

    .. code:
        R 0 1
        TICK
        CNOT 0 1
        TICK
        M 0 1

    with ``tick_iterators = [to_mid_cycle_iterator, to_end_cycle_iterator]`` is translated to

    .. code:
        [
            [
                (reset_z_iterator, layout_0),
                (reset_z_iterator, layout_1),
            ],
            [
                (to_mid_cycle_iterator, layout_0),
                (to_mid_cycle_iterator, layout_1),
            ],
            [
                (log_trans_cnot_iterator, layout_0, layout_1),
            ],
            [
                (to_end_cycle_iterator, layout_0),
                (to_end_cycle_iterator, layout_1),
            ],
            [
                (measurement_z_iterator, layout_0),
                (measurement_z_iterator, layout_1),
            ],
        ]

    """
    if not isinstance(circuit, stim.Circuit):
        raise TypeError(
            f"'circuit' must be a stim.Circuit, but {type(circuit)} was given."
        )
    circuit = circuit.flattened()
    if not isinstance(layouts, Collection):
        raise TypeError(f"'layouts' must be a list, but {type(layouts)} was given.")
    if circuit.num_qubits > len(layouts):
        raise ValueError("There are more qubits in the circuit than in 'layouts'.")
    if any(not isinstance(l, Layout) for l in layouts):
        raise TypeError("All elements in 'layouts' must be a Layout.")
    if not isinstance(gate_to_iterator, dict):
        raise TypeError(
            f"'gate_to_iterator' must be a dict, but {type(gate_to_iterator)} was given."
        )
    if any(not isinstance(f, LogOpCallable) for f in gate_to_iterator.values()):
        raise TypeError("All values of 'gate_to_iterator' must be LogOpCallable.")
    if not isinstance(tick_iterators, Sequence):
        raise TypeError(
            f"'tick_iterators' must be a Sequence, but {type(tick_iterators)} was given."
        )
    if any(not isinstance(f, LogOpCallable) for f in tick_iterators):
        raise TypeError("All elements of 'tick_iterators' must be LogOpCallable.")
    if any(
        set(["to_mid_cycle_circuit", "to_end_cycle_circuit"]).intersection(
            i.log_op_type
        )
        == set()
        for i in tick_iterators
    ):
        raise TypeError(
            "All elements of 'tick_iterators' must be either "
            "a 'to_mid_cycle_circuit' or 'to_end_cycle_circuit' LogOpCallable type."
        )

    unique_names = set(i.name for i in circuit)
    unique_names.discard("TICK")
    if unique_names > set(gate_to_iterator):
        raise ValueError(
            "Not all operations in 'circuit' are present in 'gate_to_iterator'."
        )

    instructions: Instructions = []
    num_ticks: int = 0
    for instr in circuit:
        if instr.name == "TICK":
            instructions.append((tick_iterators[num_ticks % len(tick_iterators)],))
            num_ticks += 1
            continue

        func_iter = gate_to_iterator[instr.name]
        targets: list[int] = [t.value for t in instr.targets_copy()]

        if set(["tq_unitary_gate"]).intersection(func_iter.log_op_type):
            for i, j in _grouper(targets, 2):
                instructions.append((func_iter, layouts[i], layouts[j]))
        else:
            for i in targets:
                instructions.append((func_iter, layouts[i]))

    schedule = schedule_from_instructions(instructions)

    return schedule


def schedule_from_instructions(instructions: Instructions) -> Schedule:
    """
    Builds a schedule from a list of instructions.
    In each block, layouts only participate in a single operation and
    QEC-like rounds are only performed to active layouts. Addling is automatically
    added to layouts not participating in a logical operation.

    Parameters
    ----------
    instructions
        List of operations to be applied to a single qubit or pair of qubits.
        See Notes for more information.

    Returns
    -------
    blocks
        List of blocks from the schedule. Each block contains a set of logical
        operations for the active layouts. Each layout only performs a single
        logical operation in each block. If a layout is not performing any
        logical operation while others are (and it is not a QEC-like round), then
        ``idle_iterator`` is inserted with this layout. QEC-like rounds and logical
        operations cannot be mixed together.

    Notes
    -----
    The term QEC-like rounds refers to any ``LogOpCallable`` that is in
    ``surface_sim.util.circuit_operations.QEC_OP_TYPES``.

    Adding ``idle_iterator`` is needed to have ``Model.incoming_noise``
    for a layout that is idling.

    As an example, the code

    .. code:
        [
            (reset_z_iterator, layout_0),
            (reset_z_iterator, layout_1),
            (qec_round_iterator,),
            (log_meas_iterator, layout_1),
            (qec_round_iterator,),
        ]

    is transformed into

    .. code:
        [
            [
                (reset_z_iterator, layout_0),
                (reset_z_iterator, layout_1),
            ],
            [
                (qec_round_iterator, layout_0),
                (qec_round_iterator, layout_1),
            ],
            [
                (log_meas_iterator, layout_1),
                (idle_iterator, layout_0),
            ],
            [
                (qec_round_iterator, layout_0),
            ],
        ]
    """
    if not isinstance(instructions, Collection):
        raise TypeError(
            f"'instructions' must be a sequence, but {type(instructions)} was given."
        )
    if any(not isinstance(op, Collection) for op in instructions):
        raise TypeError("Elements of 'instructions' must be sequences.")
    for op in instructions:
        if not isinstance(op[0], LogOpCallable):
            raise TypeError("Elements in 'instructions[i][0]' must be LogOpCallable.")
        if any(not isinstance(l, Layout) for l in op[1:]):
            raise TypeError("Elements in 'instructions[i][1:]' must be Layouts.")

    blocks: list[list[LogicalOperation]] = []
    curr_block: list[LogicalOperation] = []
    counter: dict[Layout, int] = {}

    def flush(
        blocks: list[list[LogicalOperation]],
        curr_block: list[LogicalOperation],
        counter: dict[Layout, int],
    ):
        # if necessary, add idling.
        # only add idling if at least one layout is performing an operation.
        # the situation where no layout is performing anything can happen when
        # performing more than one QEC round between logical gates
        if len(curr_block) == 0:
            return blocks, curr_block, counter
        for l, k in counter.items():
            if k == 0:
                curr_block.append((idle_iterator, l))

        # add current block and reset variables
        blocks.append(curr_block)
        curr_block = []
        counter = {l: 0 for l in counter}
        return blocks, curr_block, counter

    for operation in instructions:
        op = operation[0]
        if set(QEC_OP_TYPES).intersection(op.log_op_type):
            # flush all logical operations and
            blocks, curr_block, counter = flush(blocks, curr_block, counter)
            # if there are no active layouts, raise error as it is not possible
            # to perform a QEC round nothing
            if len(counter) == 0:
                raise ValueError("No active layout found when performing a QEC round.")
            # add a QEC round for all active layouts
            blocks.append([])
            for layout in counter:
                blocks[-1].append((operation[0], layout))
            continue

        # activate layouts. If not the check for layouts in current operation are
        # active does not work for resets (because the layout is inactive previously).
        if set(RESET_OP_TYPES).intersection(op.log_op_type):
            layouts = operation[1:]
            if any(l in counter for l in layouts):
                raise ValueError(
                    "An activate layout cannot be resetted, it needs to be measured first."
                )

            curr_block.append(operation)
            for l in layouts:
                counter[l] = 1
            continue

        # check if the layouts of the current operation are active.
        layouts = operation[1:]
        if not all(l in counter for l in layouts):
            raise ValueError("An inactive layout is perfoming a logical operation.")
        # check if a layout is already participating in an operation,
        # if true, flush the operations as if not it would be participating
        # in more than one in the current block
        if any(counter[l] == 1 for l in layouts):
            blocks, curr_block, counter = flush(blocks, curr_block, counter)

        curr_block.append(operation)
        if set(["measurement"]).intersection(op.log_op_type):
            for l in layouts:
                counter.pop(l)
        elif set(["sq_unitary_gate", "tq_unitary_gate"]).intersection(op.log_op_type):
            for l in layouts:
                counter[l] += 1
        else:
            raise ValueError(f"Do not know how to process '{op.log_op_type}'.")

    # flush remaining operations
    blocks, curr_block, counter = flush(blocks, curr_block, counter)

    return blocks


def get_layouts_from_schedule(schedule: Schedule) -> list[Layout]:
    """Returns a list of all layouts present in the given schedule."""
    layouts: list[Layout] = []
    for block in schedule:
        for op in block:
            if len(op) > 1:
                layouts += list(op[1:])
    return layouts


def experiment_from_schedule(
    schedule: Schedule,
    model: Model,
    detectors: Detectors,
    anc_reset: bool = True,
    anc_detectors: Collection[str] | None = None,
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
        If ``True``, ancillas are reset at the beginning of the QEC round.
        By default ``True``.
    anc_detectors
        List of ancilla qubits for which to define the detectors.
        If ``None``, adds all detectors.
        By default ``None``.

    Returns
    -------
    experiment
        Stim circuit corresponding to the logical equivalent of the
        given schedule. For each logical measurement, an observable
        is defined. To redefine them, see ``redefine_obs_from_circuit``.

    Notes
    -----
    The scheduling of the gates between QEC rounds is not optimal as there could
    be more idling than necessary. This is caused by using ``merge_logical_operations``.
    """
    if not isinstance(model, Model):
        raise TypeError(f"'model' must be a Model, but {type(model)} was given.")
    if not isinstance(detectors, Detectors):
        raise TypeError(
            f"'detectors' must be a Detectors, but {type(detectors)} was given."
        )

    layouts = get_layouts_from_schedule(schedule)

    experiment = stim.Circuit()
    model.new_circuit()
    detectors.new_circuit()

    experiment += qubit_coords(model, *layouts)

    for block in schedule:
        pre_log_noise, log_ops, post_log_noise = group_logical_operations(block)
        if pre_log_noise:
            experiment += merge_logical_noise(pre_log_noise, model=model)

        experiment += merge_logical_operations(
            log_ops,
            model=model,
            detectors=detectors,
            init_log_obs_ind=experiment.num_observables,
            anc_reset=anc_reset,
            anc_detectors=anc_detectors,
        )

        if post_log_noise:
            experiment += merge_logical_noise(post_log_noise, model=model)

    return experiment


def redefine_obs_from_circuit(
    encoded_circuit: stim.Circuit, unencoded_circuit: stim.Circuit
) -> stim.Circuit:
    """
    Redefines the observables in the given encoded circuit to match the ones
    in the unencoded circuit.

    Parameters
    ----------
    encoded_circuit
        Encoded stim circuit with observables defined for all the logical measurements.
    unencoded_circuit
        Unencoded stim circuit for the given ``encoded_circuit``.

    Returns
    -------
    new_circuit
        Same as ``encoded_circuit`` except for the redefined observables
        based on the ones in ``unencoded_circuit``.
    """
    if not isinstance(encoded_circuit, stim.Circuit):
        raise TypeError(
            "'encoded_circuit' must be a stim.Circuit, "
            f"but {type(encoded_circuit)} was given."
        )
    if not isinstance(unencoded_circuit, stim.Circuit):
        raise TypeError(
            "'unencoded_circuit' must be a stim.Circuit, "
            f"but {type(unencoded_circuit)} was given."
        )
    if encoded_circuit.num_observables != unencoded_circuit.num_measurements:
        raise ValueError(
            "The number of observables in 'encoded_circuit' "
            f"({encoded_circuit.num_observables}) "
            "must match with the number of measurements in 'unencoded_circuit' "
            f"({unencoded_circuit.num_measurements})."
        )
    if unencoded_circuit.num_measurements == 0:
        return encoded_circuit

    encoded_circuit = move_observables_to_end(encoded_circuit)
    unencoded_circuit = move_observables_to_end(unencoded_circuit)

    observables: dict[int, stim.CircuitInstruction] = {}
    for k, instr in enumerate(encoded_circuit.flattened()[::-1]):
        if instr.name == "OBSERVABLE_INCLUDE":
            obs_ind = instr.gate_args_copy()[0]
            observables[obs_ind - encoded_circuit.num_observables] = list(
                instr.targets_copy()
            )
        else:
            break

    new_circuit = encoded_circuit[:-k]

    for instr in unencoded_circuit.flattened():
        if instr.name != "OBSERVABLE_INCLUDE":
            continue

        new_obs_ind = instr.gate_args_copy()
        new_obs_targets = list(
            chain(*[observables[t.value] for t in instr.targets_copy()])
        )
        new_instr = stim.CircuitInstruction(
            "OBSERVABLE_INCLUDE", gate_args=new_obs_ind, targets=new_obs_targets
        )
        new_circuit.append(new_instr)

    return new_circuit


def experiment_from_circuit(
    circuit: stim.Circuit,
    layouts: list[Layout],
    model: Model,
    detectors: Detectors,
    gate_to_iterator: dict[str, LogOpCallable],
    anc_reset: bool = True,
    anc_detectors: Collection[str] | None = None,
) -> stim.Circuit:
    """
    Returns the encoded version of the given circuit.

    Parameters
    ----------
    circuit
        Stim circuit.
    layouts
        List of layouts whose index match the qubit index in ``circuit``.
        This function only works for layouts that only have one logical qubit.
    model
        Noise model for the gates.
    detectors
        Object to build the detectors.
    gate_to_iterator
        Dictionary mapping the names of stim circuit instructions used in ``circuit``
        to the functions that generate the equivalent logical circuit.
        Note that ``TICK`` always refers to a QEC round for all layouts.
    anc_reset
        If ``True``, ancillas are reset at the beginning of the QEC round.
        By default ``True``.
    anc_detectors
        List of ancilla qubits for which to define the detectors.
        If ``None``, adds all detectors.
        By default ``None``.

    Returns
    -------
    experiment
        Stim circuit corresponding to the encoded version of ``circuit``.
        If ``circuit`` contains observable definitions,
        then the observables in ``experiment`` correspond to those.
        If not, there is one observable for each measurement in ``circuit``.

    Notes
    -----
    For more information, check the documentation of:
    ``schedule_from_circuit``, ``experiment_from_schedule``, and
    ``redefine_obs_from_circuit``.
    """
    schedule = schedule_from_circuit(circuit, layouts, gate_to_iterator)
    stim_circuit = experiment_from_schedule(
        schedule, model, detectors, anc_reset=anc_reset, anc_detectors=anc_detectors
    )
    if circuit.num_observables != 0:
        stim_circuit = redefine_obs_from_circuit(stim_circuit, circuit)

    return stim_circuit


def _grouper(iterable: Iterable[T], n: int) -> Iterable[tuple[T, ...]]:
    args = [iter(iterable)] * n
    return zip(*args, strict=True)
