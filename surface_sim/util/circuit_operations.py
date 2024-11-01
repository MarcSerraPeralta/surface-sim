from collections.abc import Collection, Callable

import stim

from ..layouts import Layout
from ..detectors import Detectors
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


def merge_circuits(*circuits: stim.Circuit, check_meas: bool = True) -> stim.Circuit:
    """Returns a circuit in which the given circuits have been merged
    following the TICK blocks.

    It assumes that the qubits are only involved in one operation between TICKs.

    Parameters
    ----------
    *circuits
        Circuits to merge.
    check_meas
        Flag for checking that there is only one TICK block with measurement
        instructions in each of the circuits and that it appears in the same TICK block.
        By default ``True``.

    Returns
    -------
    merged_circuit
        Circuit from merging the given circuits.
    """
    if any([not isinstance(c, stim.Circuit) for c in circuits]):
        raise TypeError("The given circuits are not stim.Circuits.")
    if len(set(c.num_ticks for c in circuits)) != 1:
        raise ValueError("All the circuits must have the same number of TICKs.")

    # split circuits into TICK blocks
    blocks = [[] for _ in range(circuits[0].num_ticks + 1)]
    measure_block = [set() for _, _ in enumerate(circuits)]
    for k, circuit in enumerate(circuits):
        block_id = 0
        for instr in circuit.flattened():
            if instr.name == "TICK":
                block_id += 1
            else:
                if instr.name in MEAS_INSTR:
                    measure_block[k].add(block_id)
                blocks[block_id].append(instr)

    if check_meas:
        if any(len(m) > 1 for m in measure_block):
            raise ValueError(
                "A circuit has more than one TICK block with measurements."
            )
        if len(set(m.pop() for m in measure_block if m != set())) > 1:
            raise ValueError(
                "The measurements happen at different TICK blocks in each circuit."
            )

    # sort the instructions in each block for better readability in the circuit
    blocks = [sorted(block, key=lambda x: x.name) for block in blocks]

    # merge instructions in blocks and into a circuit.
    # remember that it is not possible to add stim.CircuitInstruction to stim.Circuit
    tick = stim.Circuit("TICK")
    merged_blocks = [
        stim.Circuit("\n".join(map(str, block))) + tick for block in blocks
    ]
    merged_blocks[-1].pop(-1)  # removes extra TICK at the end
    merged_circuit = sum(merged_blocks, start=stim.Circuit())

    return merged_circuit


def merge_qec_rounds(
    qec_round_iterator: Callable,
    model: Model,
    layouts: Collection[Layout],
    detectors: Detectors,
    anc_reset: bool = True,
    anc_detectors: Collection[str] | None = None,
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
        Collection of code layouts.
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
    if not isinstance(layouts, Collection):
        raise TypeError(
            f"'layouts' must be a collection, but {type(layouts)} was given."
        )
    if any(not isinstance(l, Layout) for l in layouts):
        raise TypeError("Elements in 'layouts' must be Layout objects.")
    if not isinstance(qec_round_iterator, Callable):
        raise TypeError(
            f"'qec_round_iterator' must be callable, but {type(qec_round_iterator)} was given."
        )
    if anc_detectors is not None:
        anc_qubits = [l.get_qubits(role="anc") for l in layouts]
        if set(anc_detectors) > set(sum(anc_qubits, start=[])):
            raise ValueError("Some elements in 'anc_detectors' are not ancilla qubits.")

    circuit = stim.Circuit()
    for blocks in zip(
        *[
            qec_round_iterator(model=model, layout=l, anc_reset=anc_reset, **kargs)
            for l in layouts
        ]
    ):
        merged_block = merge_circuits(*blocks)

        if (len(merged_block) != 0) and (merged_block[0].name == "TICK"):
            # avoid multiple 'TICK's in a single block
            circuit.append(merged_block[0])
            continue

        circuit += merged_block

    # add detectors
    circuit += detectors.build_from_anc(
        model.meas_target, anc_reset, anc_qubits=anc_detectors
    )

    return circuit
