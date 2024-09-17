import numpy as np
import xarray as xr
import stim

from surface_sim.detectors import Detectors


def test_detectors_update():
    anc_qubits = ["X1", "Z1"]
    detectors = Detectors(anc_qubits=anc_qubits, frame="1")
    unitary_mat = xr.DataArray(
        data=[[1, 1], [0, 1]], coords=dict(new_stab_gen=anc_qubits, stab_gen=anc_qubits)
    )
    detectors.update(unitary_mat)

    new_gen = xr.DataArray(
        data=[[1, 1], [0, 1]],
        coords=dict(
            stab_gen=anc_qubits,
            basis=range(len(anc_qubits)),
        ),
    )

    assert (detectors.curr_gen == new_gen).all()

    detectors.update(unitary_mat)

    new_gen = xr.DataArray(
        data=np.identity(len(anc_qubits), dtype=np.int64),
        coords=dict(
            stab_gen=anc_qubits,
            basis=range(len(anc_qubits)),
        ),
    )

    assert (detectors.curr_gen == new_gen).all()

    return


def test_detectors_build_from_anc():
    anc_qubits = ["X1", "Z1"]
    unitary_mat = xr.DataArray(
        data=[[1, 1], [0, 1]], coords=dict(new_stab_gen=anc_qubits, stab_gen=anc_qubits)
    )
    meas_rec = lambda q, t: stim.target_rec(anc_qubits.index(q) - 2 + t * 100)

    detectors = Detectors(anc_qubits=anc_qubits, frame="1")
    detectors.num_rounds = 1
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_anc(meas_rec, meas_reset=True)
    detector_rec = [
        set([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert {-102, -101, -202} in detector_rec
    assert {-101, -201} in detector_rec

    detectors.num_rounds = 0
    detectors_stim = detectors.build_from_anc(meas_rec, meas_reset=True)
    detector_rec = [
        set([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert {-102, -101} in detector_rec
    assert {-101} in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="r")
    detectors.num_rounds = 1
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_anc(meas_rec, meas_reset=True)
    detector_rec = [
        set([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert {-102, -201, -202} in detector_rec
    assert {-101, -201} in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="1")
    detectors.num_rounds = 2
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_anc(meas_rec, meas_reset=False)
    detector_rec = [
        set([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert {-102, -101, -202, -201, -202, -302} in detector_rec
    assert {-101, -201, -201, -301} in detector_rec

    detectors.num_rounds = 1
    detectors_stim = detectors.build_from_anc(meas_rec, meas_reset=False)
    detector_rec = [
        set([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert {-102, -101, -202, -201, -202} in detector_rec
    assert {-101, -201, -201} in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="r")
    detectors.num_rounds = 2
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_anc(meas_rec, meas_reset=False)
    detector_rec = [
        set([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert {-102, -201, -202, -301, -202, -302} in detector_rec
    assert {-101, -201, -201, -301} in detector_rec

    return


def test_detectors_build_from_data():
    anc_qubits = ["X1", "Z1"]
    data_qubits = ["D1", "D2", "D3"]
    unitary_mat = xr.DataArray(
        data=[[1, 1], [0, 1]], coords=dict(new_stab_gen=anc_qubits, stab_gen=anc_qubits)
    )
    adj_matrix = xr.DataArray(
        data=[[1, 1, 0], [0, 0, 1]],
        coords=dict(from_qubit=anc_qubits, to_qubit=data_qubits),
    )

    def meas_rec(q, t):
        rec = t * 100
        if q in anc_qubits:
            rec += anc_qubits.index(q) - 2
            return stim.target_rec(rec)
        if q in data_qubits:
            rec += data_qubits.index(q) * 10 - 30
            return stim.target_rec(rec)

    detectors = Detectors(anc_qubits=anc_qubits, frame="1")
    detectors.num_rounds = 1
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_data(meas_rec, adj_matrix, meas_reset=True)
    detector_rec = [
        set([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert {-130, -120, -110, -202} in detector_rec
    assert {-110, -201} in detector_rec

    detectors.num_rounds = 0
    detectors_stim = detectors.build_from_data(meas_rec, adj_matrix, meas_reset=True)
    detector_rec = [
        set([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert {-130, -120, -110} in detector_rec
    assert {-110} in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="r")
    detectors.num_rounds = 1
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_data(meas_rec, adj_matrix, meas_reset=True)
    detector_rec = [
        set([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert {-130, -120, -201, -202} in detector_rec
    assert {-110, -201} in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="1")
    detectors.num_rounds = 2
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_data(meas_rec, adj_matrix, meas_reset=False)
    detector_rec = [
        set([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert {-130, -120, -110, -202, -201, -202, -302} in detector_rec
    assert {-110, -201, -201, -301} in detector_rec

    detectors.num_rounds = 1
    detectors_stim = detectors.build_from_data(meas_rec, adj_matrix, meas_reset=False)
    detector_rec = [
        set([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert {-130, -120, -110, -202, -201, -202} in detector_rec
    assert {-110, -201, -201} in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="r")
    detectors.num_rounds = 2
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_data(meas_rec, adj_matrix, meas_reset=False)
    detector_rec = [
        set([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert {-120, -130, -201, -202, -301, -202, -302} in detector_rec
    assert {-110, -201, -201, -301} in detector_rec

    return
