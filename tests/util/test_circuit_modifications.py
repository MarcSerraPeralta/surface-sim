import stim

from surface_sim.models import CircuitNoiseModel
from surface_sim.util import add_noise_to_circuit


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
