import stim

from surface_sim.layouts import rot_surface_code, rot_surface_code_rectangle

from surface_sim.experiments.rot_surface_code_css import (
    memory_experiment,
    repeated_s_experiment,
    repeated_cnot_experiment,
)
from surface_sim.models import NoiselessModel
from surface_sim import Detectors
from surface_sim.log_gates.rot_surface_code_css import set_trans_s, set_trans_cnot


def test_memory_experiment():
    layout = rot_surface_code(distance=3)
    qubit_ids = {q: i for i, q in enumerate(layout.get_qubits())}
    anc_coords = {q: layout.get_coords([q])[0] for q in layout.get_qubits(role="anc")}
    model = NoiselessModel(qubit_ids)
    detectors = Detectors(
        layout.get_qubits(role="anc"), frame="1", anc_coords=anc_coords
    )

    for rot_basis in [True, False]:
        circuit = memory_experiment(
            model=model,
            layout=layout,
            detectors=detectors,
            num_rounds=10,
            anc_reset=False,
            data_init={q: 0 for q in layout.get_qubits(role="data")},
            rot_basis=rot_basis,
        )

        assert isinstance(circuit, stim.Circuit)

        # check that the detectors and logicals fulfill their
        # conditions by building the stim diagram
        dem = circuit.detector_error_model(allow_gauge_detectors=True)

        num_coords = 0
        anc_coords = {k: list(map(float, v)) for k, v in anc_coords.items()}
        for dem_instr in dem:
            if dem_instr.type == "detector":
                assert dem_instr.args_copy()[:-1] in anc_coords.values()
                num_coords += 1

        assert num_coords == dem.num_detectors

    return


def test_repeated_s_experiment():
    layout = rot_surface_code_rectangle(distance_z=4, distance_x=3)
    set_trans_s(layout, "D1")
    qubit_ids = {q: i for i, q in enumerate(layout.get_qubits())}
    anc_coords = {q: layout.get_coords([q])[0] for q in layout.get_qubits(role="anc")}
    model = NoiselessModel(qubit_ids)
    detectors = Detectors(
        layout.get_qubits(role="anc"), frame="1", anc_coords=anc_coords
    )

    for rot_basis in [True, False]:
        circuit = repeated_s_experiment(
            model=model,
            layout=layout,
            detectors=detectors,
            num_s_gates=4,
            num_rounds_per_gate=2,
            anc_reset=False,
            data_init={q: 0 for q in layout.get_qubits(role="data")},
            rot_basis=rot_basis,
        )

        assert isinstance(circuit, stim.Circuit)

        # check that the detectors and logicals fulfill their
        # conditions by building the stim diagram
        dem = circuit.detector_error_model(allow_gauge_detectors=True)

        num_coords = 0
        anc_coords = {k: list(map(float, v)) for k, v in anc_coords.items()}
        for dem_instr in dem:
            if dem_instr.type == "detector":
                assert dem_instr.args_copy()[:-1] in anc_coords.values()
                num_coords += 1

        assert num_coords == dem.num_detectors

    return


def test_repeated_cnot_experiment():
    layout_c = rot_surface_code(distance=3)
    layout_t = rot_surface_code(
        distance=3,
        logical_qubit_label="L1",
        init_point=(20, 20),
        init_data_qubit_id=20,
        init_zanc_qubit_id=9,
        init_xanc_qubit_id=9,
        init_ind=layout_c.get_max_ind() + 1,
    )
    set_trans_cnot(layout_c, layout_t)
    set_trans_cnot(layout_t, layout_c)
    qubit_inds = {q: layout_c.get_inds([q])[0] for q in layout_c.get_qubits()}
    qubit_inds.update({q: layout_t.get_inds([q])[0] for q in layout_t.get_qubits()})
    anc_coords = {
        q: layout_c.get_coords([q])[0] for q in layout_c.get_qubits(role="anc")
    }
    anc_coords.update(
        {q: layout_t.get_coords([q])[0] for q in layout_t.get_qubits(role="anc")}
    )
    model = NoiselessModel(qubit_inds)
    detectors = Detectors(
        layout_c.get_qubits(role="anc") + layout_t.get_qubits(role="anc"),
        frame="r",
        anc_coords=anc_coords,
    )

    for rot_basis in [True, False]:
        circuit = repeated_cnot_experiment(
            model=model,
            layout_c=layout_c,
            layout_t=layout_t,
            detectors=detectors,
            num_cnot_gates=4,
            num_rounds_per_gate=2,
            cnot_orientation="alternating",
            anc_reset=False,
            data_init={
                q: 0
                for q in layout_c.get_qubits(role="data")
                + layout_t.get_qubits(role="data")
            },
            rot_basis=rot_basis,
        )

        assert isinstance(circuit, stim.Circuit)

        # check that the detectors and logicals fulfill their
        # conditions by building the stim diagram
        dem = circuit.detector_error_model(allow_gauge_detectors=True)

        num_coords = 0
        anc_coords = {k: list(map(float, v)) for k, v in anc_coords.items()}
        for dem_instr in dem:
            if dem_instr.type == "detector":
                assert dem_instr.args_copy()[:-1] in anc_coords.values()
                num_coords += 1

        assert num_coords == dem.num_detectors

    return


def test_memory_experiment_anc_detectors():
    layout = rot_surface_code(distance=3)
    qubit_ids = {q: i for i, q in enumerate(layout.get_qubits())}
    model = NoiselessModel(qubit_ids)
    detectors = Detectors(layout.get_qubits(role="anc"), frame="1")
    circuit = memory_experiment(
        model=model,
        layout=layout,
        detectors=detectors,
        num_rounds=10,
        anc_reset=False,
        anc_detectors=["X1"],
        data_init={q: 0 for q in layout.get_qubits(role="data")},
        rot_basis=True,
    )

    num_anc = len(layout.get_qubits(role="anc"))
    num_anc_x = len(layout.get_qubits(role="anc", stab_type="x_type"))
    assert circuit.num_detectors == 10 * num_anc + num_anc_x

    non_zero_dets = []
    for instr in circuit.flattened():
        if instr.name == "DETECTOR" and len(instr.targets_copy()) != 0:
            non_zero_dets.append(instr)

    assert len(non_zero_dets) == 10 + 1

    return


def test_repeated_s_experiment_anc_detectors():
    layout = rot_surface_code_rectangle(distance_z=4, distance_x=3)
    set_trans_s(layout, "D1")
    qubit_ids = {q: i for i, q in enumerate(layout.get_qubits())}
    model = NoiselessModel(qubit_ids)
    detectors = Detectors(layout.get_qubits(role="anc"), frame="1")
    circuit = repeated_s_experiment(
        model=model,
        layout=layout,
        detectors=detectors,
        num_s_gates=4,
        num_rounds_per_gate=2,
        anc_reset=False,
        anc_detectors=["X1"],
        data_init={q: 0 for q in layout.get_qubits(role="data")},
        rot_basis=True,
    )

    num_anc = len(layout.get_qubits(role="anc"))
    num_anc_x = len(layout.get_qubits(role="anc", stab_type="x_type"))
    assert circuit.num_detectors == (1 + 4 * 2) * num_anc + num_anc_x

    non_zero_dets = []
    for instr in circuit.flattened():
        if instr.name == "DETECTOR" and len(instr.targets_copy()) != 0:
            non_zero_dets.append(instr)

    assert len(non_zero_dets) == 1 + 4 * 2 + 1

    return


def test_repeated_cnot_experiment_anc_detectors():
    layout_c = rot_surface_code(distance=3)
    layout_t = rot_surface_code(
        distance=3,
        logical_qubit_label="L1",
        init_point=(20, 20),
        init_data_qubit_id=20,
        init_zanc_qubit_id=9,
        init_xanc_qubit_id=9,
        init_ind=layout_c.get_max_ind() + 1,
    )
    set_trans_cnot(layout_c, layout_t)
    set_trans_cnot(layout_t, layout_c)
    qubit_inds = {q: layout_c.get_inds([q])[0] for q in layout_c.get_qubits()}
    qubit_inds.update({q: layout_t.get_inds([q])[0] for q in layout_t.get_qubits()})
    anc_coords = {
        q: layout_c.get_coords([q])[0] for q in layout_c.get_qubits(role="anc")
    }
    anc_coords.update(
        {q: layout_t.get_coords([q])[0] for q in layout_t.get_qubits(role="anc")}
    )
    model = NoiselessModel(qubit_inds)
    detectors = Detectors(
        layout_c.get_qubits(role="anc") + layout_t.get_qubits(role="anc"),
        frame="r",
        anc_coords=anc_coords,
    )
    circuit = repeated_cnot_experiment(
        model=model,
        layout_c=layout_c,
        layout_t=layout_t,
        detectors=detectors,
        num_cnot_gates=4,
        num_rounds_per_gate=2,
        cnot_orientation="alternating",
        anc_reset=False,
        data_init={
            q: 0
            for q in layout_c.get_qubits(role="data") + layout_t.get_qubits(role="data")
        },
        rot_basis=True,
        anc_detectors=["X1"],
    )

    num_anc = len(layout_c.get_qubits(role="anc")) + len(
        layout_t.get_qubits(role="anc")
    )
    num_anc_x = len(layout_c.get_qubits(role="anc", stab_type="x_type")) + len(
        layout_t.get_qubits(role="anc", stab_type="x_type")
    )
    assert circuit.num_detectors == (1 + 4 * 2) * num_anc + num_anc_x

    non_zero_dets = []
    for instr in circuit.flattened():
        if instr.name == "DETECTOR" and len(instr.targets_copy()) != 0:
            non_zero_dets.append(instr)

    assert len(non_zero_dets) == 1 + 4 * 2 + 1

    return


def test_memory_experiment_gauge_detectors():
    layout = rot_surface_code(distance=3)
    qubit_ids = {q: i for i, q in enumerate(layout.get_qubits())}
    model = NoiselessModel(qubit_ids)
    detectors = Detectors(layout.get_qubits(role="anc"), frame="1")
    circuit = memory_experiment(
        model=model,
        layout=layout,
        detectors=detectors,
        num_rounds=10,
        anc_reset=False,
        data_init={q: 0 for q in layout.get_qubits(role="data")},
        rot_basis=True,
        gauge_detectors=False,
    )

    num_anc = len(layout.get_qubits(role="anc"))
    num_anc_x = len(layout.get_qubits(role="anc", stab_type="x_type"))
    assert circuit.num_detectors == 10 * num_anc + num_anc_x

    non_zero_dets = []
    for instr in circuit.flattened():
        if instr.name == "DETECTOR" and len(instr.targets_copy()) != 0:
            non_zero_dets.append(instr)

    assert len(non_zero_dets) == num_anc_x + 9 * num_anc + num_anc_x

    return


def test_repeated_s_experiment_gauge_detectors():
    layout = rot_surface_code_rectangle(distance_z=4, distance_x=3)
    set_trans_s(layout, "D1")
    qubit_ids = {q: i for i, q in enumerate(layout.get_qubits())}
    model = NoiselessModel(qubit_ids)
    detectors = Detectors(layout.get_qubits(role="anc"), frame="1")
    circuit = repeated_s_experiment(
        model=model,
        layout=layout,
        detectors=detectors,
        num_s_gates=4,
        num_rounds_per_gate=2,
        anc_reset=False,
        data_init={q: 0 for q in layout.get_qubits(role="data")},
        rot_basis=True,
        gauge_detectors=False,
    )

    num_anc = len(layout.get_qubits(role="anc"))
    num_anc_x = len(layout.get_qubits(role="anc", stab_type="x_type"))
    assert circuit.num_detectors == (1 + 4 * 2) * num_anc + num_anc_x

    non_zero_dets = []
    for instr in circuit.flattened():
        if instr.name == "DETECTOR" and len(instr.targets_copy()) != 0:
            non_zero_dets.append(instr)

    assert len(non_zero_dets) == num_anc_x + 4 * 2 * num_anc + num_anc_x

    return


def test_repeated_cnot_experiment_gauge_detectors():
    layout_c = rot_surface_code(distance=3)
    layout_t = rot_surface_code(
        distance=3,
        logical_qubit_label="L1",
        init_point=(20, 20),
        init_data_qubit_id=20,
        init_zanc_qubit_id=9,
        init_xanc_qubit_id=9,
        init_ind=layout_c.get_max_ind() + 1,
    )
    set_trans_cnot(layout_c, layout_t)
    set_trans_cnot(layout_t, layout_c)
    qubit_inds = {q: layout_c.get_inds([q])[0] for q in layout_c.get_qubits()}
    qubit_inds.update({q: layout_t.get_inds([q])[0] for q in layout_t.get_qubits()})
    anc_coords = {
        q: layout_c.get_coords([q])[0] for q in layout_c.get_qubits(role="anc")
    }
    anc_coords.update(
        {q: layout_t.get_coords([q])[0] for q in layout_t.get_qubits(role="anc")}
    )
    model = NoiselessModel(qubit_inds)
    detectors = Detectors(
        layout_c.get_qubits(role="anc") + layout_t.get_qubits(role="anc"),
        frame="r",
        anc_coords=anc_coords,
    )
    circuit = repeated_cnot_experiment(
        model=model,
        layout_c=layout_c,
        layout_t=layout_t,
        detectors=detectors,
        num_cnot_gates=4,
        num_rounds_per_gate=2,
        cnot_orientation="alternating",
        anc_reset=False,
        data_init={
            q: 0
            for q in layout_c.get_qubits(role="data") + layout_t.get_qubits(role="data")
        },
        rot_basis=True,
        gauge_detectors=False,
    )

    num_anc = len(layout_c.get_qubits(role="anc")) + len(
        layout_t.get_qubits(role="anc")
    )
    num_anc_x = len(layout_c.get_qubits(role="anc", stab_type="x_type")) + len(
        layout_t.get_qubits(role="anc", stab_type="x_type")
    )
    assert circuit.num_detectors == (1 + 4 * 2) * num_anc + num_anc_x

    non_zero_dets = []
    for instr in circuit.flattened():
        if instr.name == "DETECTOR" and len(instr.targets_copy()) != 0:
            non_zero_dets.append(instr)

    assert len(non_zero_dets) == num_anc_x + 4 * 2 * num_anc + num_anc_x

    return
