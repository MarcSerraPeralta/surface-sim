from collections.abc import Sequence, Iterable

import stim

from ..util.circuit_operations import merge_logical_operations, merge_qec_rounds
from ..layouts.layout import Layout
from ..models.model import Model
from ..detectors.detectors import Detectors
from ..circuit_blocks.util import qubit_coords, idle_iterator
from ..circuit_blocks.decorators import LogOpCallable, LogicalOperation

Schedule = list[tuple[LogOpCallable] | LogicalOperation]


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
    - ``tuple[LogOpCallable]`` performs a QEC cycle to all layouts
    - ``tuple[LogOpCallable, Layout]`` performs a single-qubit operation
    - ``tuple[LogOpCallable, Layout, Layout]`` performs a two-qubit gate.

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
    if any(not isinstance(f, LogOpCallable) for f in gate_to_iterator.values()):
        raise TypeError("All values of 'gate_to_iterator' must be LogOpCallable.")
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


def blocks_from_schedule(schedule: Schedule) -> list[Schedule]:
    """Groups the logical operations in a schedule by blocks.
    In each block, layouts only participate in a single operation and
    QEC cycles are only performed to active layouts. Addling is automatically
    added to layouts not participating in a logical operation.

    Parameters
    ----------
    schedule
        List of operations to be applied to a single qubit or pair of qubits.
        See ``schedule_from_circuit`` for more information about the format.

    Returns
    -------
    blocks
        List of blocks from the schedule. Each block contains a set of logical
        operations for the active layouts. Each layout only performs a single
        logical operation in each block. If a layout is not performing any
        logical operation while others are (and it is not a QEC cycle), then
        ``idle_iterator`` is inserted with this layout. QEC rounds and logical
        operations cannot be mixed together.

    Notes
    -----
    Adding ``idle_iterator`` is needed to have ``Model.incoming_noise``
    for a layout that is idling.
    As an example, the code

    .. code:
        [
            (reset_z_iterator, layout_0),
            (reset_z_iterator, layout_1),
            (qec_round_iterator,),
            (log_x_iterator, layout_1),
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
                (qec_round_iterator, layout_0, layout_1),
            ],
            [
                (log_x_iterator, layout_1),
                (idle_iterator, layout_0),
            ],
            [
                (qec_round_iterator, layout_0),
            ],
        ]
    """
    if not isinstance(schedule, Sequence):
        raise TypeError(
            f"'schedule' must be a sequence, but {type(schedule)} was given."
        )
    if any(not isinstance(op, Sequence) for op in schedule):
        raise TypeError("Elements of 'schedule' must be sequences.")
    for op in schedule:
        if not isinstance(op[0], LogOpCallable):
            raise TypeError("Elements in 'schedule[i][0]' must be LogOpCallable.")
        if any(not isinstance(l, Layout) for l in op[1:]):
            raise TypeError("Elements in 'schedule[i][1:]' must be Layouts.")

    blocks = []
    curr_block = []
    counter = {}

    def flush(blocks, curr_block, counter):
        # if necessary, add idling.
        # only add idling if at least one layout is performing an operation.
        # the situation where no layout is performing anything can happen when
        # performing more than one QEC cycle between logical gates
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

    for operation in schedule:
        op = operation[0]
        if op.log_op_type == "qec_cycle":
            # flush all logical operations and
            blocks, curr_block, counter = flush(blocks, curr_block, counter)
            # if there are no active layouts, raise error as it is not possible
            # to perform a QEC cycle nothing
            if len(counter) == 0:
                raise ValueError("No active layout found when performing a QEC cycle.")
            # add a QEC round for all active layouts
            blocks.append([(operation[0], *counter.keys())])
            continue

        # activate layouts. If not the check for layouts in current operation are
        # active does not work for resets (because the layout is inactive previously).
        if op.log_op_type == "qubit_init":
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
        if op.log_op_type == "measurement":
            for l in layouts:
                counter.pop(l)
        elif op.log_op_type in ["sq_unitary_gate", "tq_unitary_gate"]:
            for l in layouts:
                counter[l] += 1
        else:
            raise ValueError(f"Do not know how to process '{op.log_op_type}'.")

    # flush remaining operations
    blocks, curr_block, counter = flush(blocks, curr_block, counter)

    return blocks


def get_layouts_from_schedule(schedule: Schedule) -> list[Layout]:
    """Returns a list of all layouts present in the given schedule."""
    layouts = []
    for op in schedule:
        if len(op) > 1:
            layouts += list(op[1:])
    return layouts


def experiment_from_schedule(
    schedule: Schedule,
    model: Model,
    detectors: Detectors,
    anc_reset: bool = True,
    anc_detectors: Sequence[str] | None = None,
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

    Returns
    -------
    experiment
        Stim circuit corresponding to the logical equivalent of the
        given schedule.

    Notes
    -----
    The scheduling of the gates between QEC cycles is not optimal as there could
    be more idling than necessary. This is caused by using ``merge_logical_operations``.
    """
    if not isinstance(model, Model):
        raise TypeError(f"'model' must be a Model, but {type(model)} was given.")
    if not isinstance(detectors, Detectors):
        raise TypeError(
            f"'detectors' must be a Detectors, but {type(detectors)} was given."
        )

    blocks = blocks_from_schedule(schedule)
    layouts = get_layouts_from_schedule(schedule)

    experiment = stim.Circuit()
    model.new_circuit()
    detectors.new_circuit()

    experiment += qubit_coords(model, *layouts)

    for block in blocks:
        if block[0][0].log_op_type == "qec_cycle":
            experiment += merge_qec_rounds(
                qec_round_iterator=block[0][0],
                model=model,
                layouts=block[0][1:],
                detectors=detectors,
                anc_reset=anc_reset,
                anc_detectors=anc_detectors,
            )
        else:
            experiment += merge_logical_operations(
                block,
                model=model,
                detectors=detectors,
                init_log_obs_ind=experiment.num_observables,
                anc_reset=anc_reset,
                anc_detectors=anc_detectors,
            )

    return experiment


def _grouper(iterable: Iterable, n: int):
    args = [iter(iterable)] * n
    return zip(*args, strict=True)
