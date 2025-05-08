import stim

from surface_sim.layouts import unrot_surface_codes, rot_surface_code_rectangles

from surface_sim.experiments import (
    unrot_surface_code_css,
    rot_surface_code_css,
    rot_surface_code_xzzx,
)
import surface_sim.circuit_blocks as cb
from surface_sim.models import NoiselessModel
from surface_sim import Detectors


def test_memory_experiments():
    TESTS = [
        (
            unrot_surface_codes(1, 3)[0],
            unrot_surface_code_css.memory_experiment,
        ),
        (
            rot_surface_code_rectangles(1, 3)[0],
            rot_surface_code_css.memory_experiment,
        ),
        (
            rot_surface_code_rectangles(1, 3)[0],
            rot_surface_code_xzzx.memory_experiment,
        ),
        (
            rot_surface_code_rectangles(1, 3)[0],
            lambda *args, **kargs: rot_surface_code_css.memory_experiment(
                *args,
                gate_to_iterator=cb.rot_surface_code_css.gate_to_iterator_pipelined,
                **kargs,
            ),
        ),
        (
            rot_surface_code_rectangles(1, 3)[0],
            lambda *args, **kargs: rot_surface_code_xzzx.memory_experiment(
                *args,
                gate_to_iterator=cb.rot_surface_code_xzzx.gate_to_iterator_pipelined,
                **kargs,
            ),
        ),
    ]

    for layout, memory_experiment in TESTS:
        model = NoiselessModel(layout.qubit_inds)
        detectors = Detectors(
            layout.anc_qubits, frame="post-gate", anc_coords=layout.anc_coords
        )

        # standard experiment in both basis
        for rot_basis in [True, False]:
            circuit = memory_experiment(
                model=model,
                layout=layout,
                detectors=detectors,
                num_rounds=5,
                anc_reset=False,
                data_init={q: 0 for q in layout.data_qubits},
                rot_basis=rot_basis,
            )

            assert isinstance(circuit, stim.Circuit)

            # check that the detectors and logicals fulfill their
            # conditions by building the stim diagram
            dem = circuit.detector_error_model(allow_gauge_detectors=False)

            num_coords = 0
            anc_coords = {k: list(map(float, v)) for k, v in layout.anc_coords.items()}
            for dem_instr in dem:
                if dem_instr.type == "detector":
                    assert dem_instr.args_copy()[:-1] in anc_coords.values()
                    num_coords += 1

            assert num_coords == dem.num_detectors

        # build for some specific detectors
        detectors = Detectors(
            layout.anc_qubits, frame="post-gate", include_gauge_dets=True
        )
        circuit = memory_experiment(
            model=model,
            layout=layout,
            detectors=detectors,
            num_rounds=3,
            anc_reset=False,
            anc_detectors=["X1"],
            data_init={q: 0 for q in layout.data_qubits},
            rot_basis=True,
        )

        num_anc = len(layout.anc_qubits)
        num_anc_x = len(layout.get_qubits(role="anc", stab_type="x_type"))
        assert circuit.num_detectors == 3 * num_anc + num_anc_x

        non_zero_dets = []
        for instr in circuit.flattened():
            if instr.name == "DETECTOR" and len(instr.targets_copy()) != 0:
                non_zero_dets.append(instr)

        assert len(non_zero_dets) == 3 + 1

        # without gauge detectors
        detectors = Detectors(
            layout.anc_qubits, frame="post-gate", include_gauge_dets=False
        )
        circuit = memory_experiment(
            model=model,
            layout=layout,
            detectors=detectors,
            num_rounds=3,
            anc_reset=False,
            data_init={q: 0 for q in layout.data_qubits},
            rot_basis=True,
        )
        _ = circuit.detector_error_model(allow_gauge_detectors=False)

        num_anc = len(layout.anc_qubits)
        num_anc_x = len(layout.get_qubits(role="anc", stab_type="x_type"))
        assert circuit.num_detectors == 3 * num_anc + num_anc_x

        non_zero_dets = []
        for instr in circuit.flattened():
            if instr.name == "DETECTOR" and len(instr.targets_copy()) != 0:
                non_zero_dets.append(instr)

        assert len(non_zero_dets) == num_anc_x + 2 * num_anc + num_anc_x

    return


def test_repeated_s_experiments():
    TESTS = [
        (
            unrot_surface_codes(1, 3)[0],
            unrot_surface_code_css.repeated_s_experiment,
        ),
        (
            rot_surface_code_rectangles(1, 3)[0],
            rot_surface_code_css.repeated_s_experiment,
        ),
        (
            rot_surface_code_rectangles(1, 3)[0],
            lambda *args, **kargs: rot_surface_code_css.repeated_s_experiment(
                *args,
                gate_to_iterator=cb.rot_surface_code_css.gate_to_iterator_pipelined,
                **kargs,
            ),
        ),
    ]

    for layout, repeated_s_experiment in TESTS:
        model = NoiselessModel(layout.qubit_inds)
        detectors = Detectors(
            layout.anc_qubits, frame="post-gate", anc_coords=layout.anc_coords
        )

        # standard experiment in both basis
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
            dem = circuit.detector_error_model(allow_gauge_detectors=False)

            num_coords = 0
            anc_coords = {k: list(map(float, v)) for k, v in layout.anc_coords.items()}
            for dem_instr in dem:
                if dem_instr.type == "detector":
                    assert dem_instr.args_copy()[:-1] in anc_coords.values()
                    num_coords += 1

            assert num_coords == dem.num_detectors

        # build for some specific detectors
        detectors = Detectors(
            layout.anc_qubits, frame="post-gate", include_gauge_dets=True
        )
        circuit = repeated_s_experiment(
            model=model,
            layout=layout,
            detectors=detectors,
            num_s_gates=4,
            num_rounds_per_gate=2,
            anc_reset=False,
            anc_detectors=["X1"],
            data_init={q: 0 for q in layout.data_qubits},
            rot_basis=True,
        )

        num_anc = len(layout.anc_qubits)
        num_anc_x = len(layout.get_qubits(role="anc", stab_type="x_type"))
        assert circuit.num_detectors == (1 + 4 * 2) * num_anc + num_anc_x

        non_zero_dets = []
        for instr in circuit.flattened():
            if instr.name == "DETECTOR" and len(instr.targets_copy()) != 0:
                non_zero_dets.append(instr)

        assert len(non_zero_dets) == 1 + 4 * 2 + 1

        # without gauge detectors
        detectors = Detectors(
            layout.anc_qubits, frame="post-gate", include_gauge_dets=False
        )
        circuit = repeated_s_experiment(
            model=model,
            layout=layout,
            detectors=detectors,
            num_s_gates=4,
            num_rounds_per_gate=2,
            anc_reset=False,
            data_init={q: 0 for q in layout.data_qubits},
            rot_basis=True,
        )
        _ = circuit.detector_error_model(allow_gauge_detectors=False)

        num_anc = len(layout.anc_qubits)
        num_anc_x = len(layout.get_qubits(role="anc", stab_type="x_type"))
        assert circuit.num_detectors == (1 + 4 * 2) * num_anc + num_anc_x

        non_zero_dets = []
        for instr in circuit.flattened():
            if instr.name == "DETECTOR" and len(instr.targets_copy()) != 0:
                non_zero_dets.append(instr)

        assert len(non_zero_dets) == num_anc_x + 4 * 2 * num_anc + num_anc_x

    return


def test_repeated_h_experiments():
    TESTS = [
        (
            unrot_surface_codes(1, 3)[0],
            unrot_surface_code_css.repeated_h_experiment,
        ),
    ]

    for layout, repeated_h_experiment in TESTS:
        model = NoiselessModel(layout.qubit_inds)
        detectors = Detectors(
            layout.anc_qubits, frame="post-gate", anc_coords=layout.anc_coords
        )

        # standard experiment in both basis
        for rot_basis in [True, False]:
            circuit = repeated_h_experiment(
                model=model,
                layout=layout,
                detectors=detectors,
                num_h_gates=4,
                num_rounds_per_gate=2,
                anc_reset=False,
                data_init={q: 0 for q in layout.data_qubits},
                rot_basis=rot_basis,
            )

            assert isinstance(circuit, stim.Circuit)

            # check that the detectors and logicals fulfill their
            # conditions by building the stim diagram
            dem = circuit.detector_error_model(allow_gauge_detectors=False)

            num_coords = 0
            anc_coords = {k: list(map(float, v)) for k, v in layout.anc_coords.items()}
            for dem_instr in dem:
                if dem_instr.type == "detector":
                    assert dem_instr.args_copy()[:-1] in anc_coords.values()
                    num_coords += 1

            assert num_coords == dem.num_detectors

        # build for some specific detectors
        detectors = Detectors(
            layout.anc_qubits, frame="post-gate", include_gauge_dets=True
        )
        circuit = repeated_h_experiment(
            model=model,
            layout=layout,
            detectors=detectors,
            num_h_gates=4,
            num_rounds_per_gate=2,
            anc_reset=False,
            anc_detectors=["X1"],
            data_init={q: 0 for q in layout.data_qubits},
            rot_basis=True,
        )

        num_anc = len(layout.anc_qubits)
        num_anc_x = len(layout.get_qubits(role="anc", stab_type="x_type"))
        assert circuit.num_detectors == (1 + 4 * 2) * num_anc + num_anc_x

        non_zero_dets = []
        for instr in circuit.flattened():
            if instr.name == "DETECTOR" and len(instr.targets_copy()) != 0:
                non_zero_dets.append(instr)

        assert len(non_zero_dets) == 1 + 4 * 2 + 1

        # without gauge detectors
        detectors = Detectors(
            layout.anc_qubits, frame="post-gate", include_gauge_dets=False
        )
        circuit = repeated_h_experiment(
            model=model,
            layout=layout,
            detectors=detectors,
            num_h_gates=4,
            num_rounds_per_gate=2,
            anc_reset=False,
            data_init={q: 0 for q in layout.data_qubits},
            rot_basis=True,
        )
        _ = circuit.detector_error_model(allow_gauge_detectors=False)

        num_anc = len(layout.anc_qubits)
        num_anc_x = len(layout.get_qubits(role="anc", stab_type="x_type"))
        assert circuit.num_detectors == (1 + 4 * 2) * num_anc + num_anc_x

        non_zero_dets = []
        for instr in circuit.flattened():
            if instr.name == "DETECTOR" and len(instr.targets_copy()) != 0:
                non_zero_dets.append(instr)

        assert len(non_zero_dets) == num_anc_x + 4 * 2 * num_anc + num_anc_x

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
    model = NoiselessModel.from_layouts(layout_c, layout_t)
    detectors = Detectors.from_layouts("post-gate", layout_c, layout_t)

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
        dem = circuit.detector_error_model(allow_gauge_detectors=False)

        num_coords = 0
        anc_coords = layout_c.anc_coords
        anc_coords |= layout_t.anc_coords
        anc_coords = {k: list(map(float, v)) for k, v in anc_coords.items()}
        for dem_instr in dem:
            if dem_instr.type == "detector":
                assert dem_instr.args_copy()[:-1] in anc_coords.values()
                num_coords += 1

        assert num_coords == dem.num_detectors

    return


def test_repeated_s_injection_experiment():
    layout, layout_anc = unrot_surface_codes(2, distance=3)
    model = NoiselessModel.from_layouts(layout, layout_anc)
    detectors = Detectors.from_layouts("post-gate", layout, layout_anc)

    for rot_basis in [True, False]:
        circuit = repeated_s_injection_experiment(
            model=model,
            layout=layout,
            layout_anc=layout_anc,
            detectors=detectors,
            num_s_injections=2,
            num_rounds_per_gate=1,
            anc_reset=True,
            data_init={q: 0 for q in layout.data_qubits + layout_anc.data_qubits},
            rot_basis=rot_basis,
        )

        assert isinstance(circuit, stim.Circuit)

        # check that the detectors and logicals fulfill their
        # conditions by building the stim diagram
        dem = circuit.detector_error_model(allow_gauge_detectors=False)

        num_coords = 0
        anc_coords = layout.anc_coords
        anc_coords |= layout_anc.anc_coords
        anc_coords = {k: list(map(float, v)) for k, v in anc_coords.items()}
        for dem_instr in dem:
            if dem_instr.type == "detector":
                assert dem_instr.args_copy()[:-1] in anc_coords.values()
                num_coords += 1

        assert num_coords == dem.num_detectors

    return
