from collections.abc import Sequence, Callable
from itertools import chain

import stim

from ..layouts.layout import Layout
from ..detectors import Detectors, get_support_from_adj_matrix
from ..models import Model


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
        merged_blocks = merge_tick_blocks(
            *[blocks[k][n] for k, _ in enumerate(circuits)]
        )
        merged_circuit += merged_blocks
        if n != num_ticks:
            merged_circuit += tick

    return merged_circuit


def merge_ops(
    ops: list[tuple[Callable, Layout] | tuple[Callable, Layout, Layout]],
    model: Model,
    detectors: Detectors,
    log_obs_inds: dict[str, int] | int,
    anc_reset: bool = True,
    anc_detectors: list[str] | None = None,
) -> stim.Circuit:
    """
    Returns a circuit in which the given iterators have been merged and idle
    noise have been added if the iterators have different lenght.

    Parameters
    ----------
    ops
        List of operations to merge represented as a tuple of the operation
        function iterator and the layout(s) to be applied to.
        The functions need to have ``(model, *layouts)`` as signature.
        There must be an entry for each layout except if it is participating
        in a two-qubit gate, then there must be one entry per pair.
        Each layout can only appear once, i.e. it can only perform one
        operation. Operations do not include QEC cycles (see
        ``merge_qec_cycles`` to merge cycles).
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

    Notes
    -----
    This function assumes that the number of stim operations between ticks is
    the same for each gate to perform. This includes both the physical gate and
    the noise channels.
    """
    if any(not isinstance(i[0], Callable) for i in ops):
        raise TypeError("The first element for each entry in 'ops' must be callable.")
    if any(("log_op_type" not in dir(i[0])) for i in ops):
        raise TypeError(
            "The operation functions must have the appropiate decorator. "
            "See `surface_sim.circuit_blocks` for more information."
        )
    if any(i[0].log_op_type == "qec_cycle" for i in ops):
        raise TypeError(
            "This function only accepts to merge non-QEC-cycle operations. "
            "To merge QEC cycles, use `merge_qec_rounds`."
        )
    if (not isinstance(log_obs_inds, int)) and (not isinstance(log_obs_inds, dict)):
        raise TypeError(
            f"'log_obs_inds' must be a dict, but {type(log_obs_inds)} was given."
        )
    layouts = sum([list(i[1:]) for i in ops], start=[])
    if len(layouts) != len(set(layouts)):
        raise ValueError("Layouts are participating in more than one operation.")

    circuit = stim.Circuit()
    generators = [i[0](model, *i[1:]) for i in ops]
    end = [None for _ in ops]
    tick = stim.Circuit("TICK")

    curr_block = [next(g, None) for g in generators]
    while curr_block != end:
        # merge all ticks into a single tick
        if tick in curr_block:
            curr_block = [tick] + [o for o in curr_block if o != tick]

        # change 'None' to idling
        for k, _ in enumerate(curr_block):
            if curr_block[k] is not None:
                continue
            qubits = list(chain(*[l.get_qubits() for l in ops[k][1:]]))
            curr_block[k] = model.idle(qubits)

        circuit += merge_tick_blocks(*curr_block)

        curr_block = [next(g, None) for g in generators]

    # check if detectors needs to be built because of measurements
    meas_ops = [k for k, i in enumerate(ops) if i[0].log_op_type == "measurement"]
    if len(meas_ops) != 0:
        layouts = [ops[k][1] for k in meas_ops]
        rot_bases = [ops[k][0].rot_basis for k in meas_ops]

        # add detectors
        all_stabs = []
        all_anc_support = {}
        for layout, rot_basis in zip(layouts, rot_bases):
            stab_type = "x_type" if rot_basis else "z_type"
            stabs = layout.get_qubits(role="anc", stab_type=stab_type)
            anc_support = get_support_from_adj_matrix(layout.adjacency_matrix(), stabs)
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
                log_qubits_support = getattr(layout, log_op)
                log_data_qubits = log_qubits_support[log_qubit_label]
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
    reset_ops = [k for k, i in enumerate(ops) if i[0].log_op_type == "qubit_init"]
    if len(meas_ops + reset_ops) != 0:
        for k in meas_ops:
            anc_qubits = ops[k][1].get_qubits(role="anc")
            detectors.deactivate_detectors(anc_qubits)
        for k in reset_ops:
            anc_qubits = ops[k][1].get_qubits(role="anc")
            detectors.activate_detectors(anc_qubits)

    return circuit


def merge_tick_blocks(*blocks: stim.Circuit) -> stim.Circuit:
    """Merges tick blocks to simplify the final circuit.

    Parameters
    ----------
    blocks
        Each block is a stim.Circuit.
        A valid TICK block is a ``stim.Circuit`` in which the
        qubits only perform at maximum one operation (without
        including noise channels).

    Notes
    -----
    The instructions in the output have been (correctly) sorted so that
    the lenght of the output circuit is minimal.
    """
    # check which blocks can be merged to reduce the output circuit length
    ops_blocks = [tuple(instr.name for instr in block) for block in blocks]
    mergeable_blocks = {}
    for block, op_block in zip(blocks, ops_blocks):
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


def merge_qec_rounds(
    qec_round_iterator: Callable,
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
        Callable that yields the circuits to be merged of the QEC cycle without
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
    if any(not isinstance(l, Layout) for l in layouts):
        raise TypeError("Elements in 'layouts' must be Layout objects.")
    if not isinstance(qec_round_iterator, Callable):
        raise TypeError(
            f"'qec_round_iterator' must be callable, but {type(qec_round_iterator)} was given."
        )
    if "log_op_type" not in dir(qec_round_iterator):
        raise TypeError(
            "'qec_round_iterator' must have the appropiate decorator. "
            "See `surface_sim.circuit_blocks` for more information."
        )
    if qec_round_iterator.log_op_type != "qec_cycle":
        raise TypeError(
            f"'qec_round_iterator' must be a QEC cycle, not a {qec_round_iterator.log_op_type}."
        )
    if anc_detectors is not None:
        anc_qubits = [l.get_qubits(role="anc") for l in layouts]
        if set(anc_detectors) > set(sum(anc_qubits, start=[])):
            raise ValueError("Some elements in 'anc_detectors' are not ancilla qubits.")

    tick = stim.Circuit("TICK")
    circuit = stim.Circuit()
    for blocks in zip(
        *[
            qec_round_iterator(model=model, layout=l, anc_reset=anc_reset, **kargs)
            for l in layouts
        ]
    ):
        # avoid multiple 'TICK's in a single block
        if tick in blocks:
            blocks = [tick]

        circuit += merge_tick_blocks(*blocks)

    # add detectors
    circuit += detectors.build_from_anc(
        model.meas_target, anc_reset, anc_qubits=anc_detectors
    )

    return circuit


def merge_log_meas(
    log_meas_iterator: Callable,
    model: Model,
    layouts: Sequence[Layout],
    detectors: Detectors,
    rot_bases: Sequence[bool],
    anc_reset: bool = True,
    anc_detectors: Sequence[str] | None = None,
    **kargs,
) -> stim.Circuit:
    """
    Merges the yielded circuits of the logical measurement iterator for each
    of the layouts and returns the circuit corresponding to the join of all
    these merges and the detector and observable definitions.

    IMPORTANT: this function is only kept here for compatibility and it may
    be removed in newer releases. The correct way of merging logical measurement
    is by using ``merge_ops`` and using the specific logical measurement iterators
    of each basis. Note that in this function one needs to specify the basis.

    Parameters
    ----------
    log_meas_iterator
        Callable that yields the circuits to be merged of the logigual
        measuremenet without the detectors.
        Its inputs must include ``model``, ``layout``, and ``rot_basis``.
    model
        Noise model for the gates.
    layouts
        Sequence of code layouts.
    detectors
        Object to build the detectors.
    rot_bases
        Sequence of flags for each code layout specifying the basis
        for the logical measurements. See Notes for more information.
    anc_reset
        If ``True``, ancillas are reset at the beginning of the QEC cycle.
        By default ``True``.
    anc_detectors
        List of ancilla qubits for which to define the detectors.
        If ``None``, adds all detectors.
        By default ``None``.
    kargs
        Extra arguments for ``circuit_iterator`` apart from ``layout``,
        ``model``, ``rot_basis`` and ``anc_reset``.

    Returns
    -------
    circuit
        Circuit corrresponding to the joing of all the merged individual/yielded circuits,
        including the detector definitions.

    Notes
    -----
    This function assumes that the QEC codes are CSS codes, so that the stabilizers
    are either Z-type or X-type Pauli strings. This means that for a code that has
    more than one logical qubit, one cannot measure some qubits in the logical X
    basis and some others in the logical Z basis. Therefore, the ``rot_bases``
    arguments is just a sequence specifying the logical basis in which each layout code
    has been measured on.
    """
    if not isinstance(layouts, Sequence):
        raise TypeError(
            f"'layouts' must be a collection, but {type(layouts)} was given."
        )
    if any(not isinstance(l, Layout) for l in layouts):
        raise TypeError("Elements in 'layouts' must be Layout objects.")

    if not isinstance(log_meas_iterator, Callable):
        raise TypeError(
            f"'log_meas_iterator' must be callable, but {type(log_meas_iterator)} was given."
        )

    if not isinstance(rot_bases, Sequence):
        raise TypeError(
            f"'rot_bases' must be a collection, but {type(rot_bases)} was given."
        )
    if len(rot_bases) != len(layouts):
        raise ValueError("'rot_bases' and 'layouts' must be of same lenght.")

    if anc_detectors is not None:
        anc_qubits = [l.get_qubits(role="anc") for l in layouts]
        if set(anc_detectors) > set(sum(anc_qubits, start=[])):
            raise ValueError("Some elements in 'anc_detectors' are not ancilla qubits.")

    tick = stim.Circuit("TICK")
    circuit = stim.Circuit()
    for blocks in zip(
        *[
            log_meas_iterator(model=model, layout=l, rot_basis=r, **kargs)
            for l, r in zip(layouts, rot_bases)
        ]
    ):
        # avoid multiple 'TICK's in a single block
        if tick in blocks:
            blocks = [tick]

        circuit += merge_tick_blocks(*blocks)

    # add detectors
    all_stabs = []
    all_anc_support = {}
    for layout, rot_basis in zip(layouts, rot_bases):
        stab_type = "x_type" if rot_basis else "z_type"
        stabs = layout.get_qubits(role="anc", stab_type=stab_type)
        anc_support = get_support_from_adj_matrix(layout.adjacency_matrix(), stabs)
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
    for k, (layout, rot_basis) in enumerate(zip(layouts, rot_bases)):
        num_logs = len(layout.get_logical_qubits())
        for l, log_qubit_label in enumerate(layout.get_logical_qubits()):
            log_op = "log_x" if rot_basis else "log_z"
            log_qubits_support = getattr(layout, log_op)
            log_data_qubits = log_qubits_support[log_qubit_label]
            targets = [model.meas_target(qubit, -1) for qubit in log_data_qubits]
            instr = stim.CircuitInstruction(
                "OBSERVABLE_INCLUDE", targets=targets, gate_args=[k * num_logs + l]
            )
            circuit.append(instr)

    return circuit
