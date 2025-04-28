import stim

from surface_sim.detectors import Detectors


def test_detectors_new_circuit():
    anc_qubits = ["X1", "Z1"]
    detectors = Detectors(anc_qubits=anc_qubits, frame="pre-gate")
    detectors.activate_detectors(anc_qubits, [])
    new_stabs = {"X1": set(["X1", "Z1"]), "Z1": set(["Z1"])}
    new_stabs_inv = {"X1": set(["X1", "Z1"]), "Z1": set(["Z1"])}
    detectors.update(new_stabs, new_stabs_inv)
    detectors.new_circuit()

    assert detectors.detectors == {}
    assert set(detectors.num_rounds.values()) == set([0])
    assert detectors.total_num_rounds == 0
    assert detectors.update_dict_list == []
    assert detectors.gauge_detectors == set()

    return


def test_detectors_gauge_dets():
    anc_qubits = ["X1", "Z1"]
    meas_rec = lambda q, t: stim.target_rec(anc_qubits.index(q) - 2 + t * 100)
    detectors = Detectors(
        anc_qubits=anc_qubits, frame="pre-gate", include_gauge_dets=False
    )
    detectors.activate_detectors(anc_qubits, gauge_dets=["X1", "Z1"])
    detectors_stim = detectors.build_from_anc(meas_rec, anc_reset=True)

    assert len(detectors_stim[0].targets_copy()) == 0
    assert len(detectors_stim[1].targets_copy()) == 0

    detectors.deactivate_detectors(anc_qubits)
    detectors.activate_detectors(anc_qubits, gauge_dets=["Z1"])
    detectors_stim = detectors.build_from_anc(meas_rec, anc_reset=True)

    assert len(detectors_stim[0].targets_copy()) != 0
    assert len(detectors_stim[1].targets_copy()) == 0

    detectors = Detectors(
        anc_qubits=anc_qubits, frame="pre-gate", include_gauge_dets=True
    )
    detectors.activate_detectors(anc_qubits, gauge_dets=["X1", "Z1"])
    detectors_stim = detectors.build_from_anc(meas_rec, anc_reset=True)

    assert len(detectors_stim[0].targets_copy()) != 0
    assert len(detectors_stim[1].targets_copy()) != 0
    return


def test_detectors_build_from_anc_frame_post_gate():
    anc_qubits = ["X1", "Z1"]
    new_stabs = {"X1": set(["X1", "Z1"]), "Z1": set(["Z1"])}
    new_stabs_inv = {"X1": set(["X1", "Z1"]), "Z1": set(["Z1"])}
    meas_rec = lambda q, t: stim.target_rec(anc_qubits.index(q) - 2 + t * 100)

    detectors = Detectors(anc_qubits=anc_qubits, frame="post-gate")
    detectors.activate_detectors(anc_qubits, [])
    detectors.num_rounds = {a: 0 for a in anc_qubits}
    detectors.update(new_stabs, new_stabs_inv)
    detectors_stim = detectors.build_from_anc(meas_rec, anc_reset=True)
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-102]) in detector_rec
    assert sorted([-101]) in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="post-gate")
    detectors.activate_detectors(anc_qubits, [])
    detectors.num_rounds = {a: 1 for a in anc_qubits}
    detectors.update(new_stabs, new_stabs_inv)
    detectors_stim = detectors.build_from_anc(meas_rec, anc_reset=True)
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-102, -201, -202]) in detector_rec
    assert sorted([-101, -201]) in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="post-gate")
    detectors.activate_detectors(anc_qubits, [])
    detectors.num_rounds = {a: 1 for a in anc_qubits}
    detectors.update(new_stabs, new_stabs_inv)
    detectors_stim = detectors.build_from_anc(meas_rec, anc_reset=False)
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-102, -201]) in detector_rec
    assert sorted([-101]) in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="post-gate")
    detectors.activate_detectors(anc_qubits, [])
    detectors.num_rounds = {a: 2 for a in anc_qubits}
    detectors.update(new_stabs, new_stabs_inv)
    detectors_stim = detectors.build_from_anc(meas_rec, anc_reset=False)
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-102, -201, -301, -302]) in detector_rec
    assert sorted([-101, -301]) in detector_rec

    return


def test_detectors_build_from_anc_frame_pre_gate():
    anc_qubits = ["X1", "Z1"]
    new_stabs = {"X1": set(["X1", "Z1"]), "Z1": set(["Z1"])}
    new_stabs_inv = {"X1": set(["X1", "Z1"]), "Z1": set(["Z1"])}
    meas_rec = lambda q, t: stim.target_rec(anc_qubits.index(q) - 2 + t * 100)

    detectors = Detectors(anc_qubits=anc_qubits, frame="pre-gate")
    detectors.activate_detectors(anc_qubits, [])
    detectors.num_rounds = {a: 0 for a in anc_qubits}
    detectors.update(new_stabs, new_stabs_inv)
    detectors_stim = detectors.build_from_anc(meas_rec, anc_reset=True)
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-102, -101]) in detector_rec
    assert sorted([-101]) in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="pre-gate")
    detectors.activate_detectors(anc_qubits, [])
    detectors.num_rounds = {a: 1 for a in anc_qubits}
    detectors.update(new_stabs, new_stabs_inv)
    detectors_stim = detectors.build_from_anc(meas_rec, anc_reset=True)
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-102, -101, -202]) in detector_rec
    assert sorted([-101, -201]) in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="pre-gate")
    detectors.activate_detectors(anc_qubits, [])
    detectors.num_rounds = {a: 1 for a in anc_qubits}
    detectors.update(new_stabs, new_stabs_inv)
    detectors_stim = detectors.build_from_anc(meas_rec, anc_reset=False)
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-102, -101, -201]) in detector_rec
    assert sorted([-101]) in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="pre-gate")
    detectors.activate_detectors(anc_qubits, [])
    detectors.num_rounds = {a: 2 for a in anc_qubits}
    detectors.update(new_stabs, new_stabs_inv)
    detectors_stim = detectors.build_from_anc(meas_rec, anc_reset=False)
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-102, -101, -201, -302]) in detector_rec
    assert sorted([-101, -301]) in detector_rec

    return


def test_detectors_build_from_anc_frame_gate_independent():
    anc_qubits = ["X1", "Z1"]
    new_stabs = {"X1": set(["X1", "Z1"]), "Z1": set(["Z1"])}
    new_stabs_inv = {"X1": set(["X1", "Z1"]), "Z1": set(["Z1"])}
    meas_rec = lambda q, t: stim.target_rec(anc_qubits.index(q) - 2 + t * 100)

    detectors = Detectors(anc_qubits=anc_qubits, frame="gate-independent")
    detectors.activate_detectors(anc_qubits, [])
    detectors.num_rounds = {a: 0 for a in anc_qubits}
    detectors.update(new_stabs, new_stabs_inv)
    detectors_stim = detectors.build_from_anc(meas_rec, anc_reset=True)
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-102]) in detector_rec
    assert sorted([-101]) in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="gate-independent")
    detectors.activate_detectors(anc_qubits, [])
    detectors.num_rounds = {a: 1 for a in anc_qubits}
    detectors.update(new_stabs, new_stabs_inv)
    detectors_stim = detectors.build_from_anc(meas_rec, anc_reset=True)
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-102, -202]) in detector_rec
    assert sorted([-101, -201]) in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="gate-independent")
    detectors.activate_detectors(anc_qubits, [])
    detectors.num_rounds = {a: 1 for a in anc_qubits}
    detectors.update(new_stabs, new_stabs_inv)
    detectors_stim = detectors.build_from_anc(meas_rec, anc_reset=False)
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-102]) in detector_rec
    assert sorted([-101]) in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="gate-independent")
    detectors.activate_detectors(anc_qubits, [])
    detectors.num_rounds = {a: 2 for a in anc_qubits}
    detectors.update(new_stabs, new_stabs_inv)
    detectors_stim = detectors.build_from_anc(meas_rec, anc_reset=False)
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-102, -302]) in detector_rec
    assert sorted([-101, -301]) in detector_rec

    return


def test_detectors_build_from_data_frames():
    anc_qubits = ["X1", "Z1"]
    data_qubits = ["D1", "D2", "D3"]
    new_stabs = {"X1": set(["X1", "Z1"]), "Z1": set(["Z1"])}
    new_stabs_inv = {"X1": set(["X1", "Z1"]), "Z1": set(["Z1"])}
    anc_support = {"X1": ["D1", "D2"], "Z1": ["D3"]}

    def meas_rec(q, t):
        rec = t * 100
        if q in anc_qubits:
            rec += anc_qubits.index(q) - 2
            return stim.target_rec(rec)
        if q in data_qubits:
            rec += data_qubits.index(q) * 10 - 30
            return stim.target_rec(rec)

    for frame in ["post-gate", "pre-gate"]:
        detectors = Detectors(anc_qubits=anc_qubits, frame=frame)
        detectors.activate_detectors(anc_qubits, [])
        detectors.num_rounds = {a: 0 for a in anc_qubits}
        detectors.update(new_stabs, new_stabs_inv)
        detectors_stim = detectors.build_from_data(
            meas_rec, anc_support, anc_reset=True, reconstructable_stabs=anc_qubits
        )
        detector_rec = [
            sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
        ]
        assert sorted([-130, -120]) in detector_rec
        assert sorted([-110]) in detector_rec

        detectors = Detectors(anc_qubits=anc_qubits, frame=frame)
        detectors.activate_detectors(anc_qubits, [])
        detectors.num_rounds = {a: 1 for a in anc_qubits}
        detectors.update(new_stabs, new_stabs_inv)
        detectors_stim = detectors.build_from_data(
            meas_rec, anc_support, anc_reset=True, reconstructable_stabs=anc_qubits
        )
        detector_rec = [
            sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
        ]
        assert sorted([-130, -120, -101, -102]) in detector_rec
        assert sorted([-110, -101]) in detector_rec

        detectors = Detectors(anc_qubits=anc_qubits, frame=frame)
        detectors.activate_detectors(anc_qubits, [])
        detectors.num_rounds = {a: 1 for a in anc_qubits}
        detectors.update(new_stabs, new_stabs_inv)
        detectors_stim = detectors.build_from_data(
            meas_rec, anc_support, anc_reset=False, reconstructable_stabs=anc_qubits
        )
        detector_rec = [
            sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
        ]
        assert sorted([-130, -120, -101, -102]) in detector_rec
        assert sorted([-110, -101]) in detector_rec

        detectors = Detectors(anc_qubits=anc_qubits, frame=frame)
        detectors.activate_detectors(anc_qubits, [])
        detectors.num_rounds = {a: 2 for a in anc_qubits}
        detectors.update(new_stabs, new_stabs_inv)
        detectors_stim = detectors.build_from_data(
            meas_rec, anc_support, anc_reset=False, reconstructable_stabs=anc_qubits
        )
        detector_rec = [
            sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
        ]
        assert sorted([-120, -130, -101, -102, -201, -202]) in detector_rec
        assert sorted([-110, -101, -201]) in detector_rec

    return


def test_detectors_build_from_data_frame_gate_independent():
    anc_qubits = ["X1", "Z1"]
    data_qubits = ["D1", "D2", "D3"]
    new_stabs = {"X1": set(["X1", "Z1"]), "Z1": set(["Z1"])}
    new_stabs_inv = {"X1": set(["X1", "Z1"]), "Z1": set(["Z1"])}
    anc_support = {"X1": ["D1", "D2"], "Z1": ["D3"]}

    def meas_rec(q, t):
        rec = t * 100
        if q in anc_qubits:
            rec += anc_qubits.index(q) - 2
            return stim.target_rec(rec)
        if q in data_qubits:
            rec += data_qubits.index(q) * 10 - 30
            return stim.target_rec(rec)

    detectors = Detectors(anc_qubits=anc_qubits, frame="gate-independent")
    detectors.activate_detectors(anc_qubits, [])
    detectors.num_rounds = {a: 0 for a in anc_qubits}
    detectors.update(new_stabs, new_stabs_inv)
    detectors_stim = detectors.build_from_data(
        meas_rec, anc_support, anc_reset=True, reconstructable_stabs=anc_qubits
    )
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-130, -120]) in detector_rec
    assert sorted([-110]) in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="gate-independent")
    detectors.activate_detectors(anc_qubits, [])
    detectors.num_rounds = {a: 1 for a in anc_qubits}
    detectors.update(new_stabs, new_stabs_inv)
    detectors_stim = detectors.build_from_data(
        meas_rec, anc_support, anc_reset=True, reconstructable_stabs=anc_qubits
    )
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-130, -120, -102]) in detector_rec
    assert sorted([-110, -101]) in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="gate-independent")
    detectors.activate_detectors(anc_qubits, [])
    detectors.num_rounds = {a: 1 for a in anc_qubits}
    detectors.update(new_stabs, new_stabs_inv)
    detectors_stim = detectors.build_from_data(
        meas_rec, anc_support, anc_reset=False, reconstructable_stabs=anc_qubits
    )
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-130, -120, -102]) in detector_rec
    assert sorted([-110, -101]) in detector_rec

    detectors = Detectors(anc_qubits=anc_qubits, frame="gate-independent")
    detectors.activate_detectors(anc_qubits, [])
    detectors.num_rounds = {a: 2 for a in anc_qubits}
    detectors.update(new_stabs, new_stabs_inv)
    detectors_stim = detectors.build_from_data(
        meas_rec, anc_support, anc_reset=False, reconstructable_stabs=anc_qubits
    )
    detector_rec = [
        sorted([t.value for t in instr.targets_copy()]) for instr in detectors_stim
    ]
    assert sorted([-120, -130, -102, -202]) in detector_rec
    assert sorted([-110, -101, -201]) in detector_rec

    return
