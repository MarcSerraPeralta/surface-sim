import stim

from ..models.model import Model
from ..setups.setup import SQ_GATES, SQ_MEASUREMENTS, SQ_RESETS, TQ_GATES

STIM_TO_MODEL = {
    v: k for k, v in (SQ_GATES | SQ_MEASUREMENTS | SQ_RESETS | TQ_GATES).items()
}
NOISE_CHANNELS = [
    "CORRELATED_ERROR",
    "DEPOLARIZE1",
    "DEPOLARIZE2",
    "E",
    "ELSE_CORRELATED_ERROR",
    "HERALDED_ERASE",
    "HERALDED_PAULI_CHANNEL_1",
    "II_ERROR",
    "I_ERROR",
    "PAULI_CHANNEL_1",
    "PAULI_CHANNEL_2",
    "X_ERROR",
    "Y_ERROR",
    "Z_ERROR",
]


def add_noise_to_circuit(
    noiseless_circuit: stim.Circuit, noise_model: Model
) -> stim.Circuit:
    """
    Adds the given noise to a noiseless Stim circuit.

    Parameters
    ----------
    noiseless_circuit
        Noiseless Stim circuit. To correctly add idling noise, the idling
        gates must be present in the circuit, see ``add_missing_idling_to_circuit``.
    noise_model
        Noise model instance to be used.

    Returns
    -------
    noisy_circuit
        Noisy Stim circuit with the given noise.

    Notes
    -----
    This function obviously does not know when a QEC round starts and
    when it ends, so the only noise models that can be applied are the
    ones that act based on the physical qubit gates that are being applied,
    e.g. ``CircuitNoiseModel`` and ``SI1000NoiseModel``.
    """
    if not isinstance(noiseless_circuit, stim.Circuit):
        raise TypeError(
            f"'noiseless_circuit' must be a stim.Circuit, but {type(noiseless_circuit)} was given."
        )
    if not isinstance(noise_model, Model):
        raise TypeError(
            f"'noise_model' must be a Model, but {type(noise_model)} was given."
        )

    label_to_ind = {q: noise_model.get_inds([q])[0] for q in noise_model.qubits}
    ind_to_label = {v: k for k, v in label_to_ind.items()}

    noisy_circuit = stim.Circuit()
    for instr in noiseless_circuit.without_noise():
        if instr.name in NOISE_CHANNELS:
            raise TypeError("'noiseless_circuit' must be noiseless.")
        elif instr.name in STIM_TO_MODEL:
            qubit_inds = [t.value for t in instr.targets_copy()]
            qubit_labels = [ind_to_label[i] for i in qubit_inds]
            noisy_circuit += getattr(noise_model, STIM_TO_MODEL[instr.name])(
                qubit_labels
            )
        elif instr.name == "TICK":
            noisy_circuit += noise_model.tick()
        else:
            noisy_circuit.append(instr)

    return noisy_circuit


def remove_idling_from_circuit(circuit: stim.Circuit) -> stim.Circuit:
    """Removes idling operations from the given circuit."""
    if not isinstance(circuit, stim.Circuit):
        raise TypeError(
            f"'circuit' must be a stim.Circuit, but {type(circuit)} was given."
        )

    new_circuit = stim.Circuit()
    for instr in circuit.flattened():
        if instr.name in set(["I", "II"]):
            continue
        new_circuit.append(instr)

    return new_circuit


def add_missing_idling_to_circuit(circuit: stim.Circuit) -> stim.Circuit:
    """Adds missing idling channels to the circuit to ensure that all qubits
    are performing a gate or idling between ``TICK``s."""
    if not isinstance(circuit, stim.Circuit):
        raise TypeError(
            f"'circuit' must be a stim.Circuit, but {type(circuit)} was given."
        )
    if len(circuit.flattened()) == 0:
        return circuit.flattened()

    qubits = set(range(circuit.num_qubits))
    new_circuit = stim.Circuit()
    curr_block_qubits = []
    for instr in circuit.flattened():
        if instr.name == "TICK":
            missing_qubits = qubits - set(curr_block_qubits)
            if missing_qubits:
                new_circuit.append(
                    stim.CircuitInstruction("I", targets=list(missing_qubits))
                )
            new_circuit.append(instr)
            curr_block_qubits = []
        elif instr.name in STIM_TO_MODEL:
            exec_qubits = [t.value for t in instr.targets_copy()]
            if set(exec_qubits).intersection(curr_block_qubits):
                raise ValueError(
                    f"At least one of qubits {exec_qubits} is performing two "
                    "(or more) operations between TICKs."
                )
            curr_block_qubits += exec_qubits
            new_circuit.append(instr)
        else:
            new_circuit.append(instr)

    # flush missing idling at the end (if last instruction is not a TICK)
    if instr.name != "TICK":
        missing_qubits = qubits - set(curr_block_qubits)
        if missing_qubits:
            new_circuit.append(
                stim.CircuitInstruction("I", targets=list(missing_qubits))
            )

    return new_circuit


def add_ticks_to_circuit(circuit: stim.Circuit) -> stim.Circuit:
    """Adds (missing) ``TICK``s to circuit so that each qubit only performs
    a single operation between ``TICK``s."""
    if not isinstance(circuit, stim.Circuit):
        raise TypeError(
            f"'circuit' must be a stim.Circuit, but {type(circuit)} was given."
        )

    new_circuit = stim.Circuit()
    curr_block_qubits = []
    curr_block_instrs = stim.Circuit()
    for instr in circuit.flattened():
        if instr.name == "TICK":
            new_circuit += curr_block_instrs
            new_circuit += stim.Circuit("TICK")
            curr_block_qubits = []
            curr_block_instrs = stim.Circuit()
        elif instr.name in STIM_TO_MODEL:
            exec_qubits = [t.value for t in instr.targets_copy()]
            if len(exec_qubits) != len(set(exec_qubits)):
                # one qubit is participating twice in the same operation.
                # this can happen because stim merges same operations from consecutive lines.
                groups = _split_targets(exec_qubits)
                new_circuit += curr_block_instrs
                if set(groups[0]).intersection(curr_block_qubits):
                    new_circuit += stim.Circuit("TICK")

                for group in groups[:-1]:
                    targets = [stim.GateTarget(t) for t in group]
                    new_circuit.append(
                        stim.CircuitInstruction(instr.name, targets=targets)
                    )
                    new_circuit += stim.Circuit("TICK")

                targets = [stim.GateTarget(t) for t in groups[-1]]
                curr_block_instrs = stim.Circuit()
                curr_block_instrs.append(
                    stim.CircuitInstruction(instr.name, targets=targets)
                )
                curr_block_qubits = groups[-1]
                continue

            if set(exec_qubits).intersection(curr_block_qubits):
                new_circuit += curr_block_instrs
                new_circuit += stim.Circuit("TICK")
                curr_block_qubits = []
                curr_block_instrs = stim.Circuit()

            curr_block_qubits += exec_qubits
            curr_block_instrs.append(instr)
        else:
            curr_block_instrs.append(instr)

    # flush remaining operations
    new_circuit += curr_block_instrs

    return new_circuit


def _split_targets(targets: list[int]) -> list[list[int]]:
    groups: list[list[int]] = []
    curr_block = []
    for target in targets:
        if target in curr_block:
            groups.append(curr_block)
            curr_block = [target]
        else:
            curr_block.append(target)
    groups.append(curr_block)
    return groups
