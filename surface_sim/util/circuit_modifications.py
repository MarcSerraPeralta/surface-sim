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
        Noiseless Stim circuit.
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
