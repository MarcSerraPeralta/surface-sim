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

    # check that the stabilizers are correctly updated in
    # `build_from_anc`.
    _ = detectors.build_from_anc(get_rec=lambda *_: stim.target_rec(-1), anc_reset=True)
    assert (detectors.curr_gen == new_gen).all()
    assert (detectors.prev_gen == new_gen).all()

    detectors.update(unitary_mat)

    new_gen = xr.DataArray(
        data=np.identity(len(anc_qubits), dtype=np.int64),
        coords=dict(
            stab_gen=anc_qubits,
            basis=range(len(anc_qubits)),
        ),
    )

    assert (detectors.curr_gen == new_gen).all()

    # this is required for testing that the stabilizers are correctly
    # updated in `build_from_data`.
    _ = detectors.build_from_anc(get_rec=lambda *_: stim.target_rec(-1), anc_reset=True)

    detectors.update(unitary_mat)

    new_gen = xr.DataArray(
        data=[[1, 1], [0, 1]],
        coords=dict(
            stab_gen=anc_qubits,
            basis=range(len(anc_qubits)),
        ),
    )

    _ = detectors.build_from_data(
        get_rec=lambda *_: stim.target_rec(-1),
        anc_reset=True,
        adjacency_matrix=xr.DataArray(
            data=np.identity(2),
            coords=dict(from_qubit=["X1", "Z1"], to_qubit=["D1", "D2"]),
        ),
        reconstructable_stabs=anc_qubits,
    )
    assert (detectors.curr_gen == new_gen).all()
    assert (detectors.prev_gen == new_gen).all()

    return


def test_detectors_update_from_dict():
    anc_qubits = ["X1", "Z1"]
    detectors = Detectors(anc_qubits=anc_qubits, frame="1")
    new_stabs = {"Z1": ["X1", "Z1"], "X1": ["X1"]}

    detectors.update_from_dict(new_stabs)

    new_gen = xr.DataArray(
        data=[[1, 1], [0, 1]],
        coords=dict(
            stab_gen=anc_qubits,
            basis=range(len(anc_qubits)),
        ),
    )

    assert (detectors.curr_gen == new_gen).all()

    return


def test_detectors_new_circuit():
    anc_qubits = ["X1", "Z1"]
    detectors = Detectors(anc_qubits=anc_qubits, frame="1")
    unitary_mat = xr.DataArray(
        data=[[1, 1], [0, 1]], coords=dict(new_stab_gen=anc_qubits, stab_gen=anc_qubits)
    )
    detectors.update(unitary_mat)
    detectors.new_circuit()

    init_gen = xr.DataArray(
        data=np.identity(len(anc_qubits), dtype=np.int64),
        coords=dict(
            stab_gen=anc_qubits,
            basis=range(len(anc_qubits)),
        ),
    )

    assert (detectors.curr_gen == init_gen).all()
    assert (detectors.prev_gen == init_gen).all()
    assert (detectors.init_gen == init_gen).all()
    assert detectors.num_rounds == 0

    return


def test_detectors_build_from_anc_frame_1():
    anc_qubits = ["X1", "Z1"]
    unitary_mat = xr.DataArray(
        data=[[1, 1], [0, 1]], coords=dict(new_stab_gen=anc_qubits, stab_gen=anc_qubits)
    )
    meas_rec = lambda q, t: stim.target_rec(anc_qubits.index(q) - 2 + t * 100)

    # FRAME '1' when reseting and not reseting the ancillas
    detectors = Detectors(anc_qubits=anc_qubits, frame="1")
    detectors.num_rounds = 0
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_anc(meas_rec, anc_reset=True)
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-102, -101]) in detector_rec
    assert sorted([-101]) in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="1")
    detectors.num_rounds = 1
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_anc(meas_rec, anc_reset=True)
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-102, -101, -202]) in detector_rec
    assert sorted([-101, -201]) in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="1")
    detectors.num_rounds = 1
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_anc(meas_rec, anc_reset=False)
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-102, -101, -202, -201, -202]) in detector_rec
    assert sorted([-101, -201, -201]) in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="1")
    detectors.num_rounds = 2
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_anc(meas_rec, anc_reset=False)
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-102, -101, -202, -201, -202, -302]) in detector_rec
    assert sorted([-101, -201, -201, -301]) in detector_rec

    return


def test_detectors_build_from_anc_frame_r():
    anc_qubits = ["X1", "Z1"]
    unitary_mat = xr.DataArray(
        data=[[1, 1], [0, 1]], coords=dict(new_stab_gen=anc_qubits, stab_gen=anc_qubits)
    )
    meas_rec = lambda q, t: stim.target_rec(anc_qubits.index(q) - 2 + t * 100)

    # FRAME 'r' when reseting and not reseting the ancillas
    detectors = Detectors(anc_qubits=anc_qubits, frame="r")
    detectors.num_rounds = 0
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_anc(meas_rec, anc_reset=True)
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-102]) in detector_rec
    assert sorted([-101]) in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="r")
    detectors.num_rounds = 1
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_anc(meas_rec, anc_reset=True)
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-102, -201, -202]) in detector_rec
    assert sorted([-101, -201]) in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="r")
    detectors.num_rounds = 1
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_anc(meas_rec, anc_reset=False)
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-102, -202, -201, -202]) in detector_rec
    assert sorted([-101, -201, -201]) in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="r")
    detectors.num_rounds = 2
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_anc(meas_rec, anc_reset=False)
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-102, -201, -202, -301, -202, -302]) in detector_rec
    assert sorted([-101, -201, -201, -301]) in detector_rec

    return


def test_detectors_build_from_anc_frame_r_1():
    anc_qubits = ["X1", "Z1"]
    unitary_mat = xr.DataArray(
        data=[[1, 1], [0, 1]], coords=dict(new_stab_gen=anc_qubits, stab_gen=anc_qubits)
    )
    meas_rec = lambda q, t: stim.target_rec(anc_qubits.index(q) - 2 + t * 100)

    # FRAME 'r-1' when reseting and not reseting the ancillas
    detectors = Detectors(anc_qubits=anc_qubits, frame="r-1")
    detectors.num_rounds = 0
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_anc(meas_rec, anc_reset=True)
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-102, -101]) in detector_rec
    assert sorted([-101]) in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="r-1")
    detectors.num_rounds = 1
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_anc(meas_rec, anc_reset=True)
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-102, -101, -202]) in detector_rec
    assert sorted([-101, -201]) in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="r-1")
    detectors.num_rounds = 1
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_anc(meas_rec, anc_reset=False)
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-102, -101, -202, -201, -202]) in detector_rec
    assert sorted([-101, -201, -201]) in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="r-1")
    detectors.num_rounds = 2
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_anc(meas_rec, anc_reset=False)
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-102, -101, -202, -201, -202, -302]) in detector_rec
    assert sorted([-101, -201, -201, -301]) in detector_rec

    return


def test_detectors_build_from_anc_frame_t():
    anc_qubits = ["X1", "Z1"]
    unitary_mat = xr.DataArray(
        data=[[1, 1], [0, 1]], coords=dict(new_stab_gen=anc_qubits, stab_gen=anc_qubits)
    )
    meas_rec = lambda q, t: stim.target_rec(anc_qubits.index(q) - 2 + t * 100)

    # FRAME 't' when reseting and not reseting the ancillas
    detectors = Detectors(anc_qubits=anc_qubits, frame="t")
    detectors.num_rounds = 0
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_anc(meas_rec, anc_reset=True)
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-102]) in detector_rec
    assert sorted([-101]) in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="t")
    detectors.num_rounds = 1
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_anc(meas_rec, anc_reset=True)
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-102, -202]) in detector_rec
    assert sorted([-101, -201]) in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="t")
    detectors.num_rounds = 1
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_anc(meas_rec, anc_reset=False)
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-102]) in detector_rec
    assert sorted([-101]) in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="t")
    detectors.num_rounds = 2
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_anc(meas_rec, anc_reset=False)
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-102, -302]) in detector_rec
    assert sorted([-101, -301]) in detector_rec

    return


def test_detectors_build_from_data_frame_1():
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
    detectors.num_rounds = 0
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_data(
        meas_rec, adj_matrix, anc_reset=True, reconstructable_stabs=anc_qubits
    )
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-130, -120, -110]) in detector_rec
    assert sorted([-110]) in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="1")
    detectors.num_rounds = 1
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_data(
        meas_rec, adj_matrix, anc_reset=True, reconstructable_stabs=anc_qubits
    )
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-130, -120, -110, -102]) in detector_rec
    assert sorted([-110, -101]) in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="1")
    detectors.num_rounds = 1
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_data(
        meas_rec, adj_matrix, anc_reset=False, reconstructable_stabs=anc_qubits
    )
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-130, -120, -110, -102]) in detector_rec
    assert sorted([-110, -101]) in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="1")
    detectors.num_rounds = 2
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_data(
        meas_rec, adj_matrix, anc_reset=False, reconstructable_stabs=anc_qubits
    )
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-130, -120, -110, -102, -202]) in detector_rec
    assert sorted([-110, -101, -201]) in detector_rec

    return


def test_detectors_build_from_data_frame_r():
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

    detectors = Detectors(anc_qubits=anc_qubits, frame="r")
    detectors.num_rounds = 0
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_data(
        meas_rec, adj_matrix, anc_reset=True, reconstructable_stabs=anc_qubits
    )
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-130, -120]) in detector_rec
    assert sorted([-110]) in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="r")
    detectors.num_rounds = 1
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_data(
        meas_rec, adj_matrix, anc_reset=True, reconstructable_stabs=anc_qubits
    )
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-130, -120, -101, -102]) in detector_rec
    assert sorted([-110, -101]) in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="r")
    detectors.num_rounds = 1
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_data(
        meas_rec, adj_matrix, anc_reset=False, reconstructable_stabs=anc_qubits
    )
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-130, -120, -101, -102]) in detector_rec
    assert sorted([-110, -101]) in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="r")
    detectors.num_rounds = 2
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_data(
        meas_rec, adj_matrix, anc_reset=False, reconstructable_stabs=anc_qubits
    )
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-120, -130, -101, -102, -201, -202]) in detector_rec
    assert sorted([-110, -101, -201]) in detector_rec

    return


def test_detectors_build_from_data_frame_r_1():
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

    detectors = Detectors(anc_qubits=anc_qubits, frame="r-1")
    detectors.num_rounds = 0
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_data(
        meas_rec, adj_matrix, anc_reset=True, reconstructable_stabs=anc_qubits
    )
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-130, -120, -110]) in detector_rec
    assert sorted([-110]) in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="r-1")
    detectors.num_rounds = 1
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_data(
        meas_rec, adj_matrix, anc_reset=True, reconstructable_stabs=anc_qubits
    )
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-130, -120, -110, -102]) in detector_rec
    assert sorted([-110, -101]) in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="r-1")
    detectors.num_rounds = 1
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_data(
        meas_rec, adj_matrix, anc_reset=False, reconstructable_stabs=anc_qubits
    )
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-130, -120, -110, -102]) in detector_rec
    assert sorted([-110, -101]) in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="r-1")
    detectors.num_rounds = 2
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_data(
        meas_rec, adj_matrix, anc_reset=False, reconstructable_stabs=anc_qubits
    )
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-120, -130, -110, -102, -202]) in detector_rec
    assert sorted([-110, -101, -201]) in detector_rec

    return


def test_detectors_build_from_data_frame_t():
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

    detectors = Detectors(anc_qubits=anc_qubits, frame="t")
    detectors.num_rounds = 0
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_data(
        meas_rec, adj_matrix, anc_reset=True, reconstructable_stabs=anc_qubits
    )
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-130, -120]) in detector_rec
    assert sorted([-110]) in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="t")
    detectors.num_rounds = 1
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_data(
        meas_rec, adj_matrix, anc_reset=True, reconstructable_stabs=anc_qubits
    )
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-130, -120, -102]) in detector_rec
    assert sorted([-110, -101]) in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="t")
    detectors.num_rounds = 1
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_data(
        meas_rec, adj_matrix, anc_reset=False, reconstructable_stabs=anc_qubits
    )
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-130, -120, -102]) in detector_rec
    assert sorted([-110, -101]) in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="t")
    detectors.num_rounds = 2
    detectors.update(unitary_mat)
    detectors_stim = detectors.build_from_data(
        meas_rec, adj_matrix, anc_reset=False, reconstructable_stabs=anc_qubits
    )
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-120, -130, -102, -202]) in detector_rec
    assert sorted([-110, -101, -201]) in detector_rec

    return
