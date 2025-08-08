from collections.abc import Sequence

import stim


def remove_nondeterministic_observables(
    circuit: stim.Circuit, deterministic_obs: Sequence[Sequence[int]]
) -> stim.Circuit:
    """Removes all observables from the given circuit and only keeps the specified
    deterministic observables.

    Parameters
    ----------
    circuit
        Stim circuit with observables.
    deterministic_obs
        List of deterministic observables in the circuit, specified by a list of
        indices corresponding to the observables in the circuit. Index ``i``
        corresponds to the ``i``th observable defined in the circuit

    Returns
    -------
    new_circuit
        Stim circuit containing only the observables specified by ``deterministic_obs``.
    """
    if not isinstance(circuit, stim.Circuit):
        raise TypeError(
            f"'circuit' must be stim.Circuit, but {type(circuit)} was given."
        )
    if not isinstance(deterministic_obs, Sequence):
        raise TypeError(
            "'deterministic_obs' must be a Sequence, "
            f"but {type(deterministic_obs)} was given."
        )
    if any(not isinstance(o, Sequence) for o in deterministic_obs):
        raise TypeError("Elements in 'deterministic_obs' must be Sequences.")
    indices = [i for o in deterministic_obs for i in o]
    if any(not isinstance(i, int) for i in indices):
        raise TypeError(
            "Elements inside each element in 'deterministic_obs' must be ints."
        )
    if max(indices) > circuit.num_observables - 1:
        raise ValueError("Index cannot be larger than 'circuit.num_observables-1'.")

    new_circuit = stim.Circuit()
    observables = []
    # moving the definition of the observables messes with the rec[-i] definition
    # therefore I need to take care of how many measurements are between the definition
    # and the end of the circuit (where I am going to define the deterministic observables)
    measurements = []
    for i, instr in enumerate(circuit.flattened()):
        if instr.name == "OBSERVABLE_INCLUDE":
            observables.append(instr)
            measurements.append(circuit[i:].num_measurements)
        else:
            new_circuit.append(instr)

    for k, det_obs in enumerate(deterministic_obs):
        new_targets = []
        for obs_ind in det_obs:
            targets = observables[obs_ind].targets_copy()
            targets = [t.value - measurements[obs_ind] for t in targets]
            new_targets += targets
        new_targets = [stim.target_rec(t) for t in new_targets]
        new_obs = stim.CircuitInstruction("OBSERVABLE_INCLUDE", new_targets, [k])
        new_circuit.append(new_obs)

    return new_circuit
