import pytest
import stim

from surface_sim.util.circuit_operations import (
    merge_circuits,
    merge_qec_rounds,
    merge_log_meas,
)
from surface_sim.circuit_blocks.unrot_surface_code_css import (
    qec_round_iterator,
    log_meas_iterator,
)
from surface_sim.models import NoiselessModel
from surface_sim.layouts.library.unrot_surface_codes import unrot_surface_code
from surface_sim import Detectors


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
        M 1
        """
    )

    with pytest.raises(ValueError):
        _ = merge_circuits(circuit_1, circuit_2)

    with pytest.raises(ValueError):
        _ = merge_circuits(circuit_1, circuit_1)

    return


def test_merge_qec_rounds():
    layout = unrot_surface_code(distance=3)
    model = NoiselessModel(layout.qubit_inds())
    detectors = Detectors(
        layout.get_qubits(role="anc"), frame="1", anc_coords=layout.anc_coords()
    )

    circuit = merge_qec_rounds(
        qec_round_iterator,
        model,
        [layout, layout],
        detectors,
    )

    assert isinstance(circuit, stim.Circuit)

    return


def test_merge_log_meas():
    layout = unrot_surface_code(distance=3, logical_qubit_label="L0")
    other_layout = unrot_surface_code(distance=3, logical_qubit_label="L1")
    model = NoiselessModel(layout.qubit_inds())
    detectors = Detectors(
        layout.get_qubits(role="anc"), frame="1", anc_coords=layout.anc_coords()
    )

    circuit = merge_log_meas(
        log_meas_iterator,
        model,
        [layout, other_layout],
        detectors,
        rot_bases=[{"L0": True}, {"L1": True}],
        anc_reset=True,
    )

    assert isinstance(circuit, stim.Circuit)

    return
