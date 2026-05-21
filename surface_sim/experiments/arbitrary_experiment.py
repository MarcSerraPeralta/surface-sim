from collections.abc import Collection, Iterable, Sequence
from typing import TypeVar

import stim

from ..circuit_blocks.decorators import LogicalOperation, LogOpCallable, noiseless
from ..circuit_blocks.util import (
    idle_iterator,
    pauli_observable_x_iterator,
    pauli_observable_y_iterator,
    pauli_observable_z_iterator,
    qubit_coords,
)
from ..detectors.detectors import Detectors
from ..layouts.layout import Layout
from ..models.model import Model
from ..util.circuit_operations import (
    FAKE_OP_TYPES,
    MEAS_INSTR,
    MEAS_OP_TYPES,
    QEC_OP_TYPES,
    RESET_OP_TYPES,
    group_logical_operations,
    merge_fake_operations,
    merge_logical_operations,
)

T = TypeVar("T")
Instructions = list[tuple[LogOpCallable] | LogicalOperation]
Schedule = list[list[LogicalOperation]]


def schedule_from_circuit(
    circuit: stim.Circuit,
    layouts: list[Layout],
    gate_to_iterator: dict[str, LogOpCallable],
) -> tuple[Schedule, dict[int, tuple[int, ...]] | None]:
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
    meas_to_obs
        Dictionary with keys corresponding to the logical measurement indices and values
        corresponding to the observable indices that the measurement has support on.
        If no observables are defined in the circuit, ``meas_to_obs = None``.

    Notes
    -----
    The format of the schedule is the following. Each element of the list
    is an operation to be applied to the qubits:
    - ``tuple[LogOpCallable, Layout]`` performs a (logical) single-layout operation
    - ``tuple[LogOpCallable, Layout, Layout]`` performs a (logical) two-qubit gate.

    The OBSERVABLE_INCLUDE instructions are only included in ``schedule`` if they
    are Pauli target (e.g., ``Z0``). The rest (e.g., the ``rec[-k]`` ones) are
    included in ``meas_to_obs``.

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

    circuit = split_observable_definitions(circuit)
    pauli_to_op = {
        "X": pauli_observable_x_iterator,
        "Y": pauli_observable_y_iterator,
        "Z": pauli_observable_z_iterator,
    }

    instructions: Instructions = []
    meas_to_obs: dict[int, tuple[int, ...]] | None = {
        i: tuple() for i in range(circuit.num_measurements)
    }
    curr_meas = 0
    for instr in circuit:
        # observable_include operations are different than the rest
        if instr.name == "OBSERVABLE_INCLUDE":
            # gate_args_copy() always returns a list[float]
            target, obs_ind = instr.targets_copy()[0], int(instr.gate_args_copy()[0])
            if target.is_measurement_record_target:
                meas_to_obs[curr_meas + target.value] += (obs_ind,)
            else:
                func_iter = pauli_to_op[target.pauli_type].copy()
                func_iter.pauli_observable_ind = obs_ind
                instructions.append((func_iter, layouts[target.value]))
            continue

        func_iter = gate_to_iterator[instr.name]
        targets: list[int] = [t.value for t in instr.targets_copy()]

        # add modifications to iterator
        if instr.tag == "noiseless":
            func_iter = noiseless(func_iter)
        if set(["logical_noise"]).intersection(func_iter.log_op_type):
            # copy to not override the parameter for all instances of this function
            func_iter = func_iter.copy()
            func_iter.log_noise_prob = instr.gate_args_copy()[0]

        # add iterator to instructions
        if instr.name == "TICK":
            instructions.append((func_iter,))
            continue

        if set(["tq_unitary_gate"]).intersection(func_iter.log_op_type):
            for i, j in _grouper(targets, 2):
                instructions.append((func_iter, layouts[i], layouts[j]))
        else:
            for i in targets:
                instructions.append((func_iter, layouts[i]))

        if instr.name in MEAS_INSTR:
            curr_meas += len(instr.targets_copy())

    schedule = schedule_from_instructions(instructions)

    if set(meas_to_obs.values()) == set([tuple()]):
        meas_to_obs = None

    return schedule, meas_to_obs


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

        if set(["logical_noise"]).intersection(func_iter.log_op_type):
            # copy to not override the parameter for all instances of this function
            func_iter = func_iter.copy()
            func_iter.log_noise_prob = instr.gate_args_copy()[0]

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
        elif set(FAKE_OP_TYPES).intersection(op.log_op_type):
            # does not count as logical operation
            pass
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
    meas_to_obs: dict[int, tuple[int, ...]] | None = None,
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
    meas_to_obs
        Dictionary with keys corresponding to the logical measurement indices and values
        corresponding to the observable indices that the measurement has support on.
        This information is used to build the observables in the circuit.
        By default, it defines an observable for every logical measurement.

    Returns
    -------
    experiment
        Stim circuit corresponding to the logical equivalent of the given schedule.

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
    num_log_meas = 0

    experiment += qubit_coords(model, *layouts)

    for block in schedule:
        pre_fake_ops, log_ops, post_fake_ops = group_logical_operations(block)
        if pre_fake_ops:
            experiment += merge_fake_operations(pre_fake_ops, model=model)

        experiment += merge_logical_operations(
            log_ops,
            model=model,
            detectors=detectors,
            num_prev_meas=num_log_meas,
            anc_reset=anc_reset,
            anc_detectors=anc_detectors,
            meas_to_obs=meas_to_obs,
        )

        if post_fake_ops:
            experiment += merge_fake_operations(post_fake_ops, model=model)

        log_meas = [
            op
            for op in log_ops
            if set(MEAS_OP_TYPES).intersection(op[0].log_op_type) != set()
        ]
        num_log_meas += len(log_meas)

    return experiment


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
    ``schedule_from_circuit`` and ``experiment_from_schedule``.
    """
    schedule, meas_to_obs = schedule_from_circuit(circuit, layouts, gate_to_iterator)
    stim_circuit = experiment_from_schedule(
        schedule,
        model,
        detectors,
        anc_reset=anc_reset,
        anc_detectors=anc_detectors,
        meas_to_obs=meas_to_obs,
    )
    return stim_circuit


def split_observable_definitions(circuit: stim.Circuit) -> stim.Circuit:
    """Splits the observable definitions so that each of them only
    contains a single target."""
    if not isinstance(circuit, stim.Circuit):
        raise TypeError(
            f"'circuit' must be a stim.Circuit, but {type(circuit)} was given."
        )

    new_circuit = stim.Circuit()
    for instr in circuit.flattened():
        if instr.name != "OBSERVABLE_INCLUDE":
            new_circuit.append(instr)
            continue

        gate_args = instr.gate_args_copy()
        for target in instr.targets_copy():
            new_instr = stim.CircuitInstruction(
                "OBSERVABLE_INCLUDE", targets=[target], gate_args=gate_args
            )
            new_circuit.append(new_instr)

    return new_circuit


def _grouper(iterable: Iterable[T], n: int) -> Iterable[tuple[T, ...]]:
    args = [iter(iterable)] * n
    return zip(*args, strict=True)
