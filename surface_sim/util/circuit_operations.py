from collections.abc import Sequence
from itertools import chain

import stim

from ..layouts.layout import Layout
from ..detectors import Detectors, get_new_stab_dict_from_layout
from ..models import Model
from ..circuit_blocks.decorators import LogOpCallable


MEAS_INSTR = [
    "M",
    "MR",
    "MRX",
    "MRY",
    "MRZ",
    "MX",
    "MY",
    "MZ",
    "MXX",
    "MYY",
    "MZZ",
    "MPP",
]


def merge_circuits(*circuits: stim.Circuit) -> stim.Circuit:
    """
    Returns a circuit in which the given circuits have been merged
    following the TICK blocks.

    The number of operations between TICKs must be the same for all qubits.
    The circuit must not include any measurement because if they get moved,
    then the ``rec[-i]`` indexes do not work.

    Parameters
    ----------
    *circuits
        Circuits to merge.

    Returns
    -------
    merged_circuit
        Circuit from merging the given circuits.
    """
    if any(not isinstance(c, stim.Circuit) for c in circuits):
        raise TypeError("The given circuits are not stim.Circuits.")
    if len(set(c.num_ticks for c in circuits)) != 1:
        raise ValueError("All the circuits must have the same number of TICKs.")

    # split circuits into TICK blocks
    num_ticks = circuits[0].num_ticks
    blocks = [[stim.Circuit() for _ in range(num_ticks + 1)] for _ in circuits]
    for k, circuit in enumerate(circuits):
        block_id = 0
        for instr in circuit.flattened():
            if instr.name in MEAS_INSTR:
                raise ValueError("Circuits cannot contain measurements.")
            if instr.name == "TICK":
                block_id += 1
                continue
            blocks[k][block_id].append(instr)

    # merge instructions in blocks and into a circuit.
    tick = stim.Circuit("TICK")
    merged_circuit = stim.Circuit()
    for n in range(num_ticks + 1):
        merged_blocks = merge_operation_layers(
            *[blocks[k][n] for k, _ in enumerate(circuits)]
        )
        merged_circuit += merged_blocks
        if n != num_ticks:
            merged_circuit += tick

    return merged_circuit


def merge_operation_layers(*operation_layers: stim.Circuit) -> stim.Circuit:
    """Merges operation layers acting on different qubits to simplify
    the final circuit.
    It tries to merge the different blocks if they have the same sequence
    of operations and noise channels, if not, blocks are stacked together.
    This ensures that the output circuit has the same effect as the stacking
    of all blocks.

    Parameters
    ----------
    operation_layers
        Each operation layer is a ``stim.Circuit`` acting on different qubits.
        A valid operation layer is a ``stim.Circuit`` in which the
        qubits perform exactly one operation (without
        including noise channels).

    Returns
    -------
    merged_blocks
        A ``stim.Circuit`` having the same effect as stacking all the
        given operation layers.

    Notes
    -----
    The instructions in ``merged_blocks`` have been (correctly) merged so that
    the lenght of the output circuit is minimal.
    """
    # check which blocks can be merged to reduce the output circuit length
    ops_blocks = [tuple(instr.name for instr in block) for block in operation_layers]
    mergeable_blocks = {}
    for block, op_block in zip(operation_layers, ops_blocks):
        if op_block not in mergeable_blocks:
            mergeable_blocks[op_block] = [block]
        else:
            mergeable_blocks[op_block].append(block)

    max_length = len(max(ops_blocks, key=lambda x: len(x)))
    merged_circuit = stim.Circuit()
    for t in range(max_length):
        for mblocks in mergeable_blocks.values():
            for block in mblocks:
                if t > len(block):
                    continue
                # the trick with the indices ensures that the returned object
                # is a stim.Circuit instead of a stim.CircuitInstruction
                merged_circuit += block[t : t + 1]

    return merged_circuit


def merge_logical_operations(
    op_iterators: list[
        tuple[LogOpCallable, Layout] | tuple[LogOpCallable, Layout, Layout]
    ],
    model: Model,
    detectors: Detectors,
    log_obs_inds: dict[str, int] | int,
    anc_reset: bool = True,
    anc_detectors: list[str] | None = None,
) -> stim.Circuit:
    """
    Returns a circuit in which the given logical operation iterators have been
    merged and idle noise have been added if the iterators have different lenght.

    Parameters
    ----------
    op_iterators
        List of logical operations to merge represented as a tuple of the operation
        function iterator and the layout(s) to be applied to.
        The functions need to have ``(model, *layouts)`` as signature.
        There must be an entry for each layout except if it is participating
        in a two-qubit gate, then there must be one entry per pair.
        Each layout can only appear once, i.e. it can only perform one
        operation. Operations do not include QEC cycles (see
        ``merge_qec_cycles`` to merge cycles).
        The TICK instructions must appear at the same time in all iterators
        when iterating through them.
    model
        Noise model for the gates.
    detectors
        Detector definitions to use.
    log_obs_inds
        List of dictionaries to be used when defining the logical observable
        arguments. The key specifies the logical qubit label and the value
        specifies the index to be used for the stim arguments.
        It can also be an integer. Then the arguments for the OBSERVABLE_INCLUDE
        will be the given integer and increments of it by 1 so that all
        observables are different.
    anc_reset
        If ``True``, ancillas are reset at the beginning of the QEC cycle.
        By default ``True``.
    anc_detectors
        List of ancilla qubits for which to define the detectors.
        If ``None``, adds all detectors.
        By default ``None``.

    Returns
    -------
    circuit
        Circuit from merging the given circuits.
    """
    if any(not isinstance(i[0], LogOpCallable) for i in op_iterators):
        raise TypeError(
            "The first element for each entry in 'op_iterators' must be LogOpCallable."
        )
    if any(i[0].log_op_type == "qec_cycle" for i in op_iterators):
        raise TypeError(
            "This function only accepts to merge non-QEC-cycle operations. "
            "To merge QEC cycles, use `merge_qec_rounds`."
        )
    if (not isinstance(log_obs_inds, int)) and (not isinstance(log_obs_inds, dict)):
        raise TypeError(
            f"'log_obs_inds' must be a dict, but {type(log_obs_inds)} was given."
        )
    layouts = sum([list(i[1:]) for i in op_iterators], start=[])
    if len(layouts) != len(set(layouts)):
        raise ValueError("Layouts are participating in more than one operation.")

    circuit = stim.Circuit()
    generators = [i[0](model, *i[1:]) for i in op_iterators]
    end = [None for _ in op_iterators]
    tick_instr = stim.Circuit("TICK")[0]

    curr_block = [next(g, None) for g in generators]
    while curr_block != end:
        # merge all ticks into a single tick.
        # [TICK, None, None] still needs to be a single TICK
        # As it is a TICK, no idling needs to be added.
        if any([tick_instr in c for c in curr_block if c is not None]):
            circuit += merge_ticks([c for c in curr_block if c is not None])
            curr_block = [next(g, None) for g in generators]
            continue

        # change 'None' to idling
        for k, _ in enumerate(curr_block):
            if curr_block[k] is not None:
                continue
            qubits = list(chain(*[l.qubits for l in op_iterators[k][1:]]))
            curr_block[k] = model.idle(qubits)

        circuit += merge_operation_layers(*curr_block)

        curr_block = [next(g, None) for g in generators]

    # update the detectors due to unitary gates
    for op in op_iterators:
        func, layouts = op[0], op[1:]
        if func.log_op_type not in ["sq_unitary_gate", "tq_unitary_gate"]:
            continue

        gate_label = func.__name__.replace("_iterator", "_")
        gate_label += "_".join([l.get_logical_qubits()[0] for l in layouts])
        new_stabs, new_stabs_inv = get_new_stab_dict_from_layout(layouts[0], gate_label)
        if len(layouts) == 2:
            new_stabs_2, new_stabs_2_inv = get_new_stab_dict_from_layout(
                layouts[1], gate_label
            )
            new_stabs.update(new_stabs_2)
            new_stabs_inv.update(new_stabs_2_inv)
        detectors.update(new_stabs, new_stabs_inv)

    # check if detectors needs to be built because of measurements
    meas_ops = [
        k for k, i in enumerate(op_iterators) if i[0].log_op_type == "measurement"
    ]
    if len(meas_ops) != 0:
        layouts = [op_iterators[k][1] for k in meas_ops]
        rot_bases = [op_iterators[k][0].rot_basis for k in meas_ops]

        # add detectors
        all_stabs = []
        all_anc_support = {}
        for layout, rot_basis in zip(layouts, rot_bases):
            stab_type = "x_type" if rot_basis else "z_type"
            stabs = layout.get_qubits(role="anc", stab_type=stab_type)
            anc_support = layout.get_support(stabs)
            all_stabs += stabs
            all_anc_support.update(anc_support)

        circuit += detectors.build_from_data(
            model.meas_target,
            all_anc_support,
            anc_reset=anc_reset,
            reconstructable_stabs=all_stabs,
            anc_qubits=anc_detectors,
        )

        # add logicals
        for layout, rot_basis in zip(layouts, rot_bases):
            for log_qubit_label in layout.get_logical_qubits():
                log_op = "log_x" if rot_basis else "log_z"
                log_data_qubits = layout.logical_param(log_op, log_qubit_label)
                targets = [model.meas_target(qubit, -1) for qubit in log_data_qubits]
                instr = stim.CircuitInstruction(
                    name="OBSERVABLE_INCLUDE",
                    targets=targets,
                    gate_args=(
                        [log_obs_inds[log_qubit_label]]
                        if not isinstance(log_obs_inds, int)
                        else [log_obs_inds]
                    ),
                )
                if isinstance(log_obs_inds, int):
                    log_obs_inds += 1
                circuit.append(instr)

    # check if detectors need to be activated or deactivated.
    # This needs to be done after defining the detectors because if not,
    # they won't be defined as they will correspond to inactive ancillas.
    reset_ops = [
        k for k, i in enumerate(op_iterators) if i[0].log_op_type == "qubit_init"
    ]
    if len(meas_ops + reset_ops) != 0:
        for k in meas_ops:
            anc_qubits = op_iterators[k][1].get_qubits(role="anc")
            detectors.deactivate_detectors(anc_qubits)
        for k in reset_ops:
            anc_qubits = op_iterators[k][1].get_qubits(role="anc")
            detectors.activate_detectors(anc_qubits)

    return circuit


def merge_qec_rounds(
    qec_round_iterator: LogOpCallable,
    model: Model,
    layouts: Sequence[Layout],
    detectors: Detectors,
    anc_reset: bool = True,
    anc_detectors: Sequence[str] | None = None,
    **kargs,
) -> stim.Circuit:
    """
    Merges the yielded circuits of the QEC round iterator for each of the layouts
    and returns the circuit corresponding to the join of all these merges and
    the detector definitions.

    Parameters
    ----------
    qec_round_iterator
        LogOpCallable that yields the circuits to be merged of the QEC cycle without
        the detectors.
        Its inputs must include ``model`` and ``layout``.
    model
        Noise model for the gates.
    layouts
        Sequence of code layouts.
    detectors
        Object to build the detectors.
    anc_reset
        If ``True``, ancillas are reset at the beginning of the QEC cycle.
        By default ``True``.
    anc_detectors
        List of ancilla qubits for which to define the detectors.
        If ``None``, adds all detectors.
        By default ``None``.
    kargs
        Extra arguments for ``circuit_iterator`` apart from ``layout``,
        ``model``, and ``anc_reset``.

    Returns
    -------
    circuit
        Circuit corrresponding to the joing of all the merged individual/yielded circuits,
        including the detector definitions.
    """
    if not isinstance(layouts, Sequence):
        raise TypeError(
            f"'layouts' must be a collection, but {type(layouts)} was given."
        )
    if len(layouts) == 0:
        return stim.Circuit()
    if any(not isinstance(l, Layout) for l in layouts):
        raise TypeError("Elements in 'layouts' must be Layout objects.")
    if not isinstance(qec_round_iterator, LogOpCallable):
        raise TypeError(
            f"'qec_round_iterator' must be LogOpCallable, but {type(qec_round_iterator)} was given."
        )
    if qec_round_iterator.log_op_type != "qec_cycle":
        raise TypeError(
            f"'qec_round_iterator' must be a QEC cycle, not a {qec_round_iterator.log_op_type}."
        )
    if anc_detectors is not None:
        data_qubits = [l.get_qubits(role="data") for l in layouts]
        if set(anc_detectors).intersection(sum(data_qubits, start=tuple())) != set():
            raise ValueError("Some elements in 'anc_detectors' are not ancilla qubits.")

    tick_instr = stim.Circuit("TICK")[0]
    circuit = stim.Circuit()
    for blocks in zip(
        *[
            qec_round_iterator(model=model, layout=l, anc_reset=anc_reset, **kargs)
            for l in layouts
        ]
    ):
        # avoid multiple 'TICK's in a single block, but be aware that
        # 'model.tick()' can return noise channels and a 'TICK'.
        # As the iterator is the same for all block, they all have the same structure.
        if tick_instr in blocks[0]:
            circuit += merge_ticks(blocks)
            continue

        circuit += merge_operation_layers(*blocks)

    # add detectors
    circuit += detectors.build_from_anc(
        model.meas_target, anc_reset, anc_qubits=anc_detectors
    )

    return circuit


def merge_ticks(blocks: Sequence[stim.Circuit]) -> stim.Circuit:
    """
    Merges stim circuit containing TICK instructions and noise channels
    so that only one TICK instruction is present while keeping if the noise
    channels happened before of after the TICK.
    It assumes that a TICK instruction is present in each block.
    """
    tick_instr = stim.Circuit("TICK")[0]
    circuit = stim.Circuit()
    after_tick = stim.Circuit()
    for block in blocks:
        tick_idx = [k for k, i in enumerate(block) if i == tick_instr]
        if len(tick_idx) != 1:
            raise ValueError("A block from cannot have more than one TICK.")
        tick_idx = tick_idx[0]
        circuit += block[:tick_idx]
        after_tick += block[tick_idx + 1 :]
    circuit += stim.Circuit("TICK") + after_tick
    return circuit
