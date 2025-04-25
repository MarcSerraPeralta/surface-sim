import stim

from surface_sim.layouts import unrot_surface_code

from surface_sim.experiments.unrot_surface_code_css import (
    memory_experiment,
    repeated_s_experiment,
    repeated_sqrt_x_experiment,
    repeated_h_experiment,
    repeated_cnot_experiment,
)
from surface_sim.models import NoiselessModel
from surface_sim import Detectors
from surface_sim.log_gates.unrot_surface_code_css import (
    set_fold_trans_s,
    set_fold_trans_sqrt_x,
    set_trans_cnot,
    set_fold_trans_h,
)


def test_memory_experiment():
    layout = unrot_surface_code(distance=3)
    model = NoiselessModel(layout.qubit_inds)
    detectors = Detectors(
        layout.anc_qubits, frame="post-gate", anc_coords=layout.anc_coords
    )

    for rot_basis in [True, False]:
        circuit = memory_experiment(
            model=model,
            layout=layout,
            detectors=detectors,
            num_rounds=10,
            anc_reset=False,
            data_init={q: 0 for q in layout.data_qubits},
            rot_basis=rot_basis,
        )

        assert isinstance(circuit, stim.Circuit)

        # check that the detectors and logicals fulfill their
        # conditions by building the stim diagram
        dem = circuit.detector_error_model(allow_gauge_detectors=True)

        num_coords = 0
        anc_coords = {k: list(map(float, v)) for k, v in layout.anc_coords.items()}
        for dem_instr in dem:
            if dem_instr.type == "detector":
                assert dem_instr.args_copy()[:-1] in anc_coords.values()
                num_coords += 1

        assert num_coords == dem.num_detectors

    return


def test_repeated_s_experiment():
    layout = unrot_surface_code(distance=3)
    set_fold_trans_s(layout, "D1")
    model = NoiselessModel(layout.qubit_inds)
    detectors = Detectors(
        layout.anc_qubits, frame="post-gate", anc_coords=layout.anc_coords
    )

    for rot_basis in [True, False]:
        circuit = repeated_s_experiment(
            model=model,
            layout=layout,
            detectors=detectors,
            num_s_gates=4,
            num_rounds_per_gate=2,
            anc_reset=False,
            data_init={q: 0 for q in layout.data_qubits},
            rot_basis=rot_basis,
        )

        assert isinstance(circuit, stim.Circuit)

        # check that the detectors and logicals fulfill their
        # conditions by building the stim diagram
        dem = circuit.detector_error_model(allow_gauge_detectors=True)

        num_coords = 0
        anc_coords = {k: list(map(float, v)) for k, v in layout.anc_coords.items()}
        for dem_instr in dem:
            if dem_instr.type == "detector":
                assert dem_instr.args_copy()[:-1] in anc_coords.values()
                num_coords += 1

        assert num_coords == dem.num_detectors

    return


def test_repeated_sqrt_x_experiment():
    layout = unrot_surface_code(distance=3)
    set_fold_trans_sqrt_x(layout, "D1")
    model = NoiselessModel(layout.qubit_inds)
    detectors = Detectors(
        layout.anc_qubits, frame="post-gate", anc_coords=layout.anc_coords
    )

    for rot_basis in [True, False]:
        circuit = repeated_sqrt_x_experiment(
            model=model,
            layout=layout,
            detectors=detectors,
            num_sqrt_x_gates=4,
            num_rounds_per_gate=2,
            anc_reset=False,
            data_init={q: 0 for q in layout.data_qubits},
            rot_basis=rot_basis,
        )

        assert isinstance(circuit, stim.Circuit)

        # check that the detectors and logicals fulfill their
        # conditions by building the stim diagram
        dem = circuit.detector_error_model(allow_gauge_detectors=True)

        num_coords = 0
        anc_coords = {k: list(map(float, v)) for k, v in layout.anc_coords.items()}
        for dem_instr in dem:
            if dem_instr.type == "detector":
                assert dem_instr.args_copy()[:-1] in anc_coords.values()
                num_coords += 1

        assert num_coords == dem.num_detectors

    return


def test_repeated_h_experiment():
    layout = unrot_surface_code(distance=3)
    set_fold_trans_h(layout, "D1")
    model = NoiselessModel(layout.qubit_inds)
    detectors = Detectors(
        layout.anc_qubits, frame="post-gate", anc_coords=layout.anc_coords
    )

    for rot_basis in [True, False]:
        circuit = repeated_h_experiment(
            model=model,
            layout=layout,
            detectors=detectors,
            num_h_gates=5,
            num_rounds_per_gate=2,
            anc_reset=False,
            data_init={q: 0 for q in layout.data_qubits},
            rot_basis=rot_basis,
        )

        assert isinstance(circuit, stim.Circuit)

        # check that the detectors and logicals fulfill their
        # conditions by building the stim diagram
        dem = circuit.detector_error_model(allow_gauge_detectors=True)

        num_coords = 0
        anc_coords = {k: list(map(float, v)) for k, v in layout.anc_coords.items()}
        for dem_instr in dem:
            if dem_instr.type == "detector":
                assert dem_instr.args_copy()[:-1] in anc_coords.values()
                num_coords += 1

        assert num_coords == dem.num_detectors

    return


def test_repeated_cnot_experiment():
    layout_c = unrot_surface_code(distance=3)
    layout_t = unrot_surface_code(
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
    qubit_inds = layout_c.qubit_inds
    qubit_inds.update(layout_t.qubit_inds)
    anc_coords = layout_c.anc_coords
    anc_coords.update(layout_t.anc_coords)
    model = NoiselessModel(qubit_inds)
    detectors = Detectors(
        layout_c.anc_qubits + layout_t.anc_qubits,
        frame="post-gate",
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
            data_init={q: 0 for q in layout_c.data_qubits + layout_t.data_qubits},
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
