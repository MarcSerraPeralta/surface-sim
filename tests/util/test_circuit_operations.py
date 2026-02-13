import pytest
import stim

from surface_sim import Detectors
from surface_sim.circuit_blocks.unrot_surface_code_css import (
    init_qubits_z0_iterator,
    log_depolarize1_error_iterator,
    log_meas_x_iterator,
    log_meas_z_iterator,
    log_x_error_iterator,
    log_x_iterator,
    log_z_error_iterator,
    qec_round_iterator,
)
from surface_sim.layouts.library.unrot_surface_codes import unrot_surface_codes
from surface_sim.models import NoiselessModel
from surface_sim.util.circuit_operations import (
    group_logical_operations,
    merge_circuits,
    merge_logical_operations,
    merge_operation_layers,
)


def test_merge_operation_layers():
    blocks = [
        stim.Circuit("X 0\nS 0"),
        stim.Circuit("X 1\nS 1"),
        stim.Circuit("X 2\nS 2"),
    ]
    circuit = merge_operation_layers(*blocks)

    expected_circuit = stim.Circuit(
        """
        X 0 1 2
        S 0 1 2
        """
    )
    assert circuit == expected_circuit

    blocks = [stim.Circuit("X 0\nS 0\nX 0\nS 0")]
    circuit = merge_operation_layers(*blocks)

    assert circuit == blocks[0]

    blocks = [
        stim.Circuit("X_ERROR(0.1) 0 1\nM 0 1\nI 8 9"),
        stim.Circuit("X_ERROR(0.1) 2 3\nM 2 3\nI 6 7"),
    ]
    circuit = merge_operation_layers(*blocks)

    expected_circuit = stim.Circuit(
        """
        X_ERROR(0.1) 0 1 2 3
        M 0 1 2 3
        I 8 9 6 7
        """
    )
    assert circuit == expected_circuit

    blocks = [
        stim.Circuit("X_ERROR(0.1) 0 1\nM 0 1\nX_ERROR(0.1) 2 3\nM 2 3"),
        stim.Circuit("X_ERROR(0.1) 4 5\nM 4 5\nX_ERROR(0.1) 6 7\nM 6 7"),
    ]
    circuit = merge_operation_layers(*blocks)

    expected_circuit = sum(blocks, start=stim.Circuit())
    assert circuit == expected_circuit

    return


def test_merge_circuits():
    circuit_1 = stim.Circuit(
        """
        X 0
        TICK
        H 0
        """
    )
    circuit_2 = stim.Circuit(
        """
        X 1
        TICK
        H 1
        """
    )

    merged_circuit = merge_circuits(circuit_1, circuit_2)

    expected_circuit = stim.Circuit(
        """
        X 0 1
        TICK
        H 0 1
        """
    )

    assert merged_circuit == expected_circuit

    circuit_1 = stim.Circuit(
        """
        X 0
        TICK
        H 0
        TICK
        """
    )

    with pytest.raises(ValueError):
        _ = merge_circuits(circuit_1, circuit_2)

    circuit_1 = stim.Circuit(
        """
        M 0
        TICK
        X 1
        """
    )

    with pytest.raises(ValueError):
        _ = merge_circuits(circuit_1, circuit_2)

    with pytest.raises(ValueError):
        _ = merge_circuits(circuit_1, circuit_1)

    return


def test_merge_qec_rounds():
    layout, other_layout = unrot_surface_codes(2, distance=3)
    qubit_inds = layout.qubit_inds
    qubit_inds.update(other_layout.qubit_inds)
    anc_qubits = layout.anc_qubits + other_layout.anc_qubits
    anc_coords = layout.anc_coords
    anc_coords.update(other_layout.anc_coords)
    model = NoiselessModel(qubit_inds)
    detectors = Detectors(anc_qubits, frame="pre-gate", anc_coords=anc_coords)

    circuit = merge_logical_operations(
        [(qec_round_iterator, layout), (qec_round_iterator, other_layout)],
        model,
        detectors,
        anc_reset=True,
    )

    assert isinstance(circuit, stim.Circuit)

    return


def test_merge_logical_measurements():
    layout, other_layout = unrot_surface_codes(2, distance=3)
    qubit_inds = layout.qubit_inds
    qubit_inds.update(other_layout.qubit_inds)
    anc_qubits = layout.anc_qubits + other_layout.anc_qubits
    anc_coords = layout.anc_coords
    anc_coords.update(other_layout.anc_coords)
    model = NoiselessModel(qubit_inds)
    detectors = Detectors(anc_qubits, frame="pre-gate", anc_coords=anc_coords)

    # need to initialize the ancillas, because if not `detectors` complains
    # that they were inactive
    detectors.activate_detectors(anc_qubits, [])

    circuit = merge_logical_operations(
        [(log_meas_z_iterator, layout), (log_meas_x_iterator, other_layout)],
        model,
        detectors,
        init_log_obs_ind=1,
        anc_reset=True,
    )

    assert isinstance(circuit, stim.Circuit)

    return


def test_merge_logical_operations():
    layout, other_layout = unrot_surface_codes(2, distance=3)
    qubit_inds = layout.qubit_inds
    qubit_inds.update(other_layout.qubit_inds)
    anc_qubits = layout.anc_qubits + other_layout.anc_qubits
    anc_coords = layout.anc_coords
    anc_coords.update(other_layout.anc_coords)
    model = NoiselessModel(qubit_inds)
    detectors = Detectors(anc_qubits, frame="pre-gate", anc_coords=anc_coords)

    circuit = merge_logical_operations(
        [
            (init_qubits_z0_iterator, layout),
            (init_qubits_z0_iterator, other_layout),
        ],
        model,
        detectors,
        anc_detectors=["X1"],
    )

    assert isinstance(circuit, stim.Circuit)

    circuit = merge_logical_operations(
        [
            (log_meas_z_iterator, layout),
            (log_x_iterator, other_layout),
        ],
        model,
        detectors,
        init_log_obs_ind=0,
        anc_reset=True,
        anc_detectors=["X1"],
    )

    assert isinstance(circuit, stim.Circuit)

    with pytest.raises(ValueError):
        _ = merge_logical_operations(
            [
                (log_meas_z_iterator, layout),
                (log_x_iterator, layout),
            ],
            model,
            detectors,
            init_log_obs_ind=0,
            anc_reset=True,
            anc_detectors=["X1"],
        )

    return


def test_group_logical_operations():
    layout, other_layout = unrot_surface_codes(2, distance=3)

    pre, ops, post = group_logical_operations(
        [
            (log_x_error_iterator, layout),
            (init_qubits_z0_iterator, layout),
            (log_depolarize1_error_iterator, other_layout),
            (init_qubits_z0_iterator, other_layout),
            (log_z_error_iterator, layout),
        ],
    )

    assert pre == [
        (log_x_error_iterator, layout),
        (log_depolarize1_error_iterator, other_layout),
    ]
    assert ops == [
        (init_qubits_z0_iterator, layout),
        (init_qubits_z0_iterator, other_layout),
    ]
    assert post == [(log_z_error_iterator, layout)]

    return
