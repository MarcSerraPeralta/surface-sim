import stim

from surface_sim.util import (
    move_observables_to_end,
    remove_nondeterministic_observables,
)


def test_remove_nondeterministic_observables():
    circuit = stim.Circuit(
        """
        R 0
        M 0 1
        OBSERVABLE_INCLUDE(0) rec[-2] rec[-1]
        M 0 
        OBSERVABLE_INCLUDE(100) rec[-1]
        M 0 1
        OBSERVABLE_INCLUDE(2) rec[-1]
        M 0
        """
    )

    new_circuit = remove_nondeterministic_observables(circuit, [[0, 1, 2], [1]])

    expected_circuit = stim.Circuit(
        """
        R 0
        M 0 1
        M 0
        M 0 1
        M 0
        OBSERVABLE_INCLUDE(0) rec[-6] rec[-5] rec[-4] rec[-2]
        OBSERVABLE_INCLUDE(1) rec[-4]
        """
    )

    assert new_circuit == expected_circuit


def test_move_observables_to_end():
    circuit = stim.Circuit(
        """
        R 0
        M 0 1
        OBSERVABLE_INCLUDE(0) rec[-2] rec[-1]
        M 0 
        OBSERVABLE_INCLUDE(100) rec[-1]
        M 0 1
        OBSERVABLE_INCLUDE(2) rec[-1]
        M 0
        """
    )

    new_circuit = move_observables_to_end(circuit)

    expected_circuit = stim.Circuit(
        """
        R 0
        M 0 1
        M 0
        M 0 1
        M 0
        OBSERVABLE_INCLUDE(0) rec[-6] rec[-5]
        OBSERVABLE_INCLUDE(100) rec[-4]
        OBSERVABLE_INCLUDE(2) rec[-2]
        """
    )

    assert new_circuit == expected_circuit
