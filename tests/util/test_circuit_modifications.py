import stim

from surface_sim.models import CircuitNoiseModel
from surface_sim.util import (
    add_missing_idling_to_circuit,
    add_noise_to_circuit,
    add_ticks_to_circuit,
    remove_idling_from_circuit,
)


def test_add_noise_to_circuit():
    noiseless_circuit = stim.Circuit(
        """
        R 0 1
        QUBIT_COORDS(9, 1) 0
        H 0
        CNOT 0 1
        RZ 0
        X 1
        M(0.1) 0
        M 0
        SQRT_X 1
        DETECTOR rec[-2]
        RY 0 1
        I 0 
        MY 1
        OBSERVABLE_INCLUDE(1) rec[-1]
        """
    )

    noise_model = CircuitNoiseModel({"a": 0, "b": 1})
    noise_model.setup.set_var_param("prob", 0.6)

    noisy_circuit = add_noise_to_circuit(noiseless_circuit, noise_model)

    expected_circuit = stim.Circuit(
        """
        R 0 1
        X_ERROR(0.6) 0 1
        QUBIT_COORDS(9, 1) 0
        H 0
        DEPOLARIZE1(0.6) 0
        CX 0 1
        DEPOLARIZE2(0.6) 0 1
        R 0
        X_ERROR(0.6) 0
        X 1
        DEPOLARIZE1(0.6) 1
        X_ERROR(0.6) 0 0
        M 0 0
        SQRT_X 1
        DEPOLARIZE1(0.6) 1
        DETECTOR rec[-2]
        RY 0 1
        X_ERROR(0.6) 0 1
        I 0
        DEPOLARIZE1(0.6) 0
        X_ERROR(0.6) 1
        MY 1
        OBSERVABLE_INCLUDE(1) rec[-1]
        """
    )

    assert noisy_circuit == expected_circuit

    return


def test_remove_idling_from_circuit():
    circuit = stim.Circuit(
        """
        R 0
        I 1 2 3
        M(0.1) 3 2
        DETECTOR rec[-1]
        CNOT 1 2
        DEPOLARIZE1(0.1) 0 1
        II 0 1
        """
    )

    output_circuit = remove_idling_from_circuit(circuit)

    expected_circuit = stim.Circuit(
        """
        R 0
        M(0.1) 3 2
        DETECTOR rec[-1]
        CNOT 1 2
        DEPOLARIZE1(0.1) 0 1
        """
    )

    assert output_circuit == expected_circuit

    return


def test_add_missing_idling_to_circuit():
    circuit = stim.Circuit(
        """
        R 0 1 2
        TICK
        I 1
        TICK
        CNOT 0 1
        S 2
        TICK
        M 0 1
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-2]
        TICK
        X 1
        """
    )

    output_circuit = add_missing_idling_to_circuit(circuit)

    expected_circuit = stim.Circuit(
        """
        R 0 1 2
        TICK
        I 1 0 2
        TICK
        CNOT 0 1
        S 2
        TICK
        M 0 1
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-2]
        I 2
        TICK
        X 1
        I 0 2
        """
    )

    assert output_circuit == expected_circuit

    output_circuit = add_missing_idling_to_circuit(circuit + stim.Circuit("TICK"))

    assert output_circuit == expected_circuit + stim.Circuit("TICK")

    return


def test_add_ticks_to_circuit():
    circuit = stim.Circuit(
        """
        R 0 1 2
        I 1
        CNOT 1 3
        CNOT 0 2
        CNOT 0 1 2 3
        M 0
        M 1
        DETECTOR rec[-1]
        MX 2
        OBSERVABLE_INCLUDE(0) rec[-2]
        X 0 1 2 3
        X 1
        TICK
        X 2
        """
    )

    output_circuit = add_ticks_to_circuit(circuit)

    expected_circuit = stim.Circuit(
        """
        R 0 1 2
        TICK
        I 1
        TICK
        CNOT 1 3
        CNOT 0 2
        TICK
        CNOT 0 1 2 3
        TICK
        M 0
        M 1
        DETECTOR rec[-1]
        MX 2
        OBSERVABLE_INCLUDE(0) rec[-2]
        TICK
        X 0 1 2 3
        TICK
        X 1
        TICK
        X 2
        """
    )

    assert output_circuit == expected_circuit

    return
