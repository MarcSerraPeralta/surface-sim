from collections.abc import Collection

import stim


def remove_nondeterministic_observables(
    circuit: stim.Circuit, deterministic_obs: Collection[Collection[int]]
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
    if circuit.num_observables == 0:
        return circuit
    if not isinstance(deterministic_obs, Collection):
        raise TypeError(
            "'deterministic_obs' must be a Collection, "
            f"but {type(deterministic_obs)} was given."
        )
    if any(not isinstance(o, Collection) for o in deterministic_obs):
        raise TypeError("Elements in 'deterministic_obs' must be Collection.")
    indices = [i for o in deterministic_obs for i in o]
    if any(not isinstance(i, int) for i in indices):
        raise TypeError(
            "Elements inside each element in 'deterministic_obs' must be ints."
        )
    if max(indices) > circuit.num_observables - 1:
        raise ValueError("Index cannot be larger than 'circuit.num_observables-1'.")

    new_circuit = move_observables_to_end(circuit)
    observables: dict[int, set[int]] = {}
    for k, instr in enumerate(new_circuit[::-1]):
        if instr.name == "OBSERVABLE_INCLUDE":
            obs_ind = instr.gate_args_copy()[0]
            targets = [t.value for t in instr.targets_copy()]
            if obs_ind not in observables:
                observables[obs_ind] = set()
            observables[obs_ind].symmetric_difference_update(targets)
        else:
            break
    new_circuit = new_circuit[:-k]

    for k, det_obs in enumerate(deterministic_obs):
        new_targets: set[int] = set()
        for obs_ind in det_obs:
            targets = observables.get(obs_ind, [])
            new_targets.symmetric_difference_update(targets)
        new_obs = stim.CircuitInstruction(
            "OBSERVABLE_INCLUDE", [stim.target_rec(t) for t in new_targets], [k]
        )
        new_circuit.append(new_obs)

    return new_circuit


def move_observables_to_end(circuit: stim.Circuit) -> stim.Circuit:
    """Move the observable definitions to the end of the given circuit."""
    if not isinstance(circuit, stim.Circuit):
        raise TypeError(
            f"'circuit' must be a stim.Circuit, but {type(circuit)} was given."
        )

    # moving the definition of the observables messes with the rec[-i] definition
    # therefore I need to take care of how many measurements are between the definition
    # and the end of the circuit.

    new_circuit = stim.Circuit()
    observables: dict[int, set[int]] = {}
    for i, instr in enumerate(circuit.flattened()):
        if instr.name == "OBSERVABLE_INCLUDE":
            obs_ind = instr.gate_args_copy()[0]
            targets = instr.targets_copy()
            m = circuit[i:].num_measurements
            if obs_ind not in observables:
                observables[obs_ind] = set()
            observables[obs_ind].symmetric_difference_update(
                [t.value - m for t in targets]
            )
        else:
            new_circuit.append(instr)

    for obs_ind, new_targets in observables.items():
        new_obs = stim.CircuitInstruction(
            "OBSERVABLE_INCLUDE", [stim.target_rec(t) for t in new_targets], [obs_ind]
        )
        new_circuit.append(new_obs)

    return new_circuit
