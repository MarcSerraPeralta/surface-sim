import numpy as np
import xarray as xr

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
    meas_rec = lambda x: anc_qubits.index(x[0]) - 2 + x[1] * 100

    detectors = Detectors(anc_qubits=anc_qubits, frame="1")
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_anc(meas_rec, meas_reset=True)
    detector_rec = [
        set([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert {-102, -101, -202} in detector_rec
    assert {-101, -201} in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="r")
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_anc(meas_rec, meas_reset=True)
    detector_rec = [
        set([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert {-102, -201, -202} in detector_rec
    assert {-101, -201} in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="1")
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_anc(meas_rec, meas_reset=False)
    detector_rec = [
        set([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert {-102, -101, -202, -201, -202, -302} in detector_rec
    assert {-101, -201, -201, -301} in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="r")
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_anc(meas_rec, meas_reset=False)
    detector_rec = [
        set([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert {-102, -201, -202, -301, -202, -302} in detector_rec
    assert {-101, -201, -201, -301} in detector_rec

    return
