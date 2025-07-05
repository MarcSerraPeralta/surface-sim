import pytest
import stim

from surface_sim.layouts import (
    unrot_surface_codes,
    rot_surface_code_rectangles,
    rot_surface_stability_rectangle,
    ssd_code,
)

from surface_sim.experiments import (
    unrot_surface_code_css,
    rot_surface_code_css,
    rot_surface_code_xzzx,
    small_stellated_dodecahedron_code,
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
        (
            ssd_code(),
            small_stellated_dodecahedron_code.memory_experiment,
        ),
        (
            rot_surface_code_rectangles(1, 3)[0],
            rot_surface_code_xzzx.memory_experiment_google,
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
        (
            ssd_code(),
            small_stellated_dodecahedron_code.repeated_s_like_experiment,
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
        (
            ssd_code(),
            small_stellated_dodecahedron_code.repeated_h_like_experiment,
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

        num_anc = len(layout.anc_qubits)
        num_anc_x = len(layout.get_qubits(role="anc", stab_type="x_type"))
        assert circuit.num_detectors == (1 + 4 * 2) * num_anc + num_anc_x

        non_zero_dets = []
        for instr in circuit.flattened():
            if instr.name == "DETECTOR" and len(instr.targets_copy()) != 0:
                non_zero_dets.append(instr)

        assert len(non_zero_dets) == num_anc_x + 4 * 2 * num_anc + num_anc_x

    return


def test_repeated_swap_experiments():
    TESTS = [
        (
            ssd_code(),
            small_stellated_dodecahedron_code.repeated_swap_r_like_experiment,
        ),
        (
            ssd_code(),
            small_stellated_dodecahedron_code.repeated_swap_s_like_experiment,
        ),
        (
            ssd_code(),
            small_stellated_dodecahedron_code.repeated_swap_a_like_experiment,
        ),
        (
            ssd_code(),
            small_stellated_dodecahedron_code.repeated_swap_b_like_experiment,
        ),
        (
            ssd_code(),
            small_stellated_dodecahedron_code.repeated_swap_c_like_experiment,
        ),
    ]

    for layout, repeated_swap_experiment in TESTS:
        model = NoiselessModel(layout.qubit_inds)
        detectors = Detectors(
            layout.anc_qubits, frame="post-gate", anc_coords=layout.anc_coords
        )

        # standard experiment in both basis
        for rot_basis in [True, False]:
            circuit = repeated_swap_experiment(
                model=model,
                layout=layout,
                detectors=detectors,
                num_swap_gates=3,
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
        circuit = repeated_swap_experiment(
            model=model,
            layout=layout,
            detectors=detectors,
            num_swap_gates=4,
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
        circuit = repeated_swap_experiment(
            model=model,
            layout=layout,
            detectors=detectors,
            num_swap_gates=4,
            num_rounds_per_gate=2,
            anc_reset=False,
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

        assert len(non_zero_dets) == num_anc_x + 4 * 2 * num_anc + num_anc_x

    return


def test_repeated_cnot_experiments():
    TESTS = [
        (
            unrot_surface_codes(2, 3),
            unrot_surface_code_css.repeated_cnot_experiment,
        ),
        (
            rot_surface_code_rectangles(2, 3),
            rot_surface_code_css.repeated_cnot_experiment,
        ),
        (
            rot_surface_code_rectangles(2, 3),
            lambda *args, **kargs: rot_surface_code_css.repeated_cnot_experiment(
                *args,
                gate_to_iterator=cb.rot_surface_code_css.gate_to_iterator_pipelined,
                **kargs,
            ),
        ),
    ]

    for layouts, repeated_cnot_experiment in TESTS:
        layout_c, layout_t = layouts
        model = NoiselessModel.from_layouts(*layouts)
        detectors = Detectors.from_layouts("post-gate", *layouts)

        # standard experiment in both basis
        for rot_basis in [True, False]:
            circuit = repeated_cnot_experiment(
                model=model,
                layout_c=layout_c,
                layout_t=layout_t,
                detectors=detectors,
                num_cnot_gates=4,
                num_rounds_per_gate=2,
                anc_reset=False,
                data_init={q: 0 for q in layout_c.data_qubits + layout_t.data_qubits},
                rot_basis=rot_basis,
            )

            assert isinstance(circuit, stim.Circuit)

            # check that the detectors and logicals fulfill their
            # conditions by building the stim diagram
            dem = circuit.detector_error_model(allow_gauge_detectors=False)

            num_coords = 0
            anc_coords = {
                k: list(map(float, v)) for k, v in layout_c.anc_coords.items()
            }
            anc_coords |= {
                k: list(map(float, v)) for k, v in layout_t.anc_coords.items()
            }
            for dem_instr in dem:
                if dem_instr.type == "detector":
                    assert dem_instr.args_copy()[:-1] in anc_coords.values()
                    num_coords += 1

            assert num_coords == dem.num_detectors

        # build for some specific detectors
        detectors = Detectors.from_layouts(
            "post-gate", *layouts, include_gauge_dets=True
        )
        circuit = repeated_cnot_experiment(
            model=model,
            layout_c=layout_c,
            layout_t=layout_t,
            detectors=detectors,
            num_cnot_gates=4,
            num_rounds_per_gate=2,
            anc_reset=False,
            anc_detectors=["X1"],
            data_init={q: 0 for q in layout_c.data_qubits + layout_t.data_qubits},
            rot_basis=True,
        )

        num_anc = len(layout_c.anc_qubits) + len(layout_t.anc_qubits)
        num_anc_x = len(layout_c.get_qubits(role="anc", stab_type="x_type")) + len(
            layout_t.get_qubits(role="anc", stab_type="x_type")
        )
        assert circuit.num_detectors == (1 + 4 * 2) * num_anc + num_anc_x

        non_zero_dets = []
        for instr in circuit.flattened():
            if instr.name == "DETECTOR" and len(instr.targets_copy()) != 0:
                non_zero_dets.append(instr)

        assert len(non_zero_dets) == 1 + 4 * 2 + 1

        # without gauge detectors
        detectors = Detectors.from_layouts(
            "post-gate", *layouts, include_gauge_dets=False
        )
        circuit = repeated_cnot_experiment(
            model=model,
            layout_c=layout_c,
            layout_t=layout_t,
            detectors=detectors,
            num_cnot_gates=4,
            num_rounds_per_gate=2,
            anc_reset=False,
            data_init={q: 0 for q in layout_c.data_qubits + layout_t.data_qubits},
            rot_basis=True,
        )

        num_anc = len(layout_c.anc_qubits) + len(layout_t.anc_qubits)
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


def test_repeated_s_injection_experiment():
    TESTS = [
        (
            unrot_surface_codes(2, 3),
            unrot_surface_code_css.repeated_s_injection_experiment,
        ),
        (
            rot_surface_code_rectangles(2, 3),
            rot_surface_code_css.repeated_s_injection_experiment,
        ),
        (
            rot_surface_code_rectangles(2, 3),
            lambda *args, **kargs: rot_surface_code_css.repeated_s_injection_experiment(
                *args,
                gate_to_iterator=cb.rot_surface_code_css.gate_to_iterator_pipelined,
                **kargs,
            ),
        ),
    ]

    for layouts, repeated_s_injection_experiment in TESTS:
        layout, layout_anc = layouts
        model = NoiselessModel.from_layouts(*layouts)
        detectors = Detectors.from_layouts("post-gate", *layouts)

        # standard experiment in both basis
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
            anc_coords = {k: list(map(float, v)) for k, v in layout.anc_coords.items()}
            anc_coords |= {
                k: list(map(float, v)) for k, v in layout_anc.anc_coords.items()
            }
            for dem_instr in dem:
                if dem_instr.type == "detector":
                    assert dem_instr.args_copy()[:-1] in anc_coords.values()
                    num_coords += 1

            assert num_coords == dem.num_detectors

        # build for some specific detectors
        detectors = Detectors.from_layouts(
            "post-gate", *layouts, include_gauge_dets=True
        )
        circuit = repeated_s_injection_experiment(
            model=model,
            layout=layout,
            layout_anc=layout_anc,
            detectors=detectors,
            num_s_injections=2,
            num_rounds_per_gate=1,
            anc_reset=False,
            anc_detectors=["X1"],
            data_init={q: 0 for q in layout.data_qubits + layout_anc.data_qubits},
            rot_basis=True,
        )

        num_anc = len(layout.anc_qubits) + len(layout_anc.anc_qubits)
        num_anc_x = len(layout.get_qubits(role="anc", stab_type="x_type")) + len(
            layout_anc.get_qubits(role="anc", stab_type="x_type")
        )
        num_anc_z = len(layout.get_qubits(role="anc", stab_type="z_type")) + len(
            layout_anc.get_qubits(role="anc", stab_type="z_type")
        )

        assert (
            circuit.num_detectors
            == num_anc + 2 * (4 * num_anc + num_anc_z // 2) + num_anc_x // 2
        )

        non_zero_dets = []
        for instr in circuit.flattened():
            if instr.name == "DETECTOR" and len(instr.targets_copy()) != 0:
                non_zero_dets.append(instr)

        assert len(non_zero_dets) == 1 + 2 * 4 * 1 + 1

        # without gauge detectors
        detectors = Detectors.from_layouts(
            "post-gate", *layouts, include_gauge_dets=False
        )
        circuit = repeated_s_injection_experiment(
            model=model,
            layout=layout,
            layout_anc=layout_anc,
            detectors=detectors,
            num_s_injections=2,
            num_rounds_per_gate=1,
            anc_reset=True,
            data_init={q: 0 for q in layout.data_qubits + layout_anc.data_qubits},
            rot_basis=True,
        )

        num_anc = len(layout.anc_qubits) + len(layout_anc.anc_qubits)
        num_anc_x = len(layout.get_qubits(role="anc", stab_type="x_type")) + len(
            layout_anc.get_qubits(role="anc", stab_type="x_type")
        )
        num_anc_z = len(layout.get_qubits(role="anc", stab_type="z_type")) + len(
            layout_anc.get_qubits(role="anc", stab_type="z_type")
        )

        assert (
            circuit.num_detectors
            == num_anc + 2 * (4 * num_anc + num_anc_z // 2) + num_anc_x // 2
        )

        non_zero_dets = []
        for instr in circuit.flattened():
            if instr.name == "DETECTOR" and len(instr.targets_copy()) != 0:
                non_zero_dets.append(instr)

        assert (
            len(non_zero_dets)
            == num_anc_x // 2
            + 2
            * (
                num_anc_x // 2
                + num_anc // 2
                + 2 * num_anc
                + num_anc_z // 2
                + num_anc // 2
            )
            + num_anc_x // 2
        )

    return


def test_stability_experiments():
    TESTS = [
        (
            rot_surface_stability_rectangle("z_type", 3, 4),
            rot_surface_code_css.stability_experiment,
            "x_type",
        ),
        (
            rot_surface_stability_rectangle("x_type", 3, 4),
            rot_surface_code_css.stability_experiment,
            "z_type",
        ),
        (
            rot_surface_stability_rectangle("z_type", 3, 4),
            rot_surface_code_xzzx.stability_experiment,
            "x_type",
        ),
        (
            rot_surface_stability_rectangle("x_type", 3, 4),
            rot_surface_code_xzzx.stability_experiment,
            "z_type",
        ),
    ]

    for layout, stability_experiment, other_stab_type in TESTS:
        model = NoiselessModel(layout.qubit_inds)
        detectors = Detectors(
            layout.anc_qubits, frame="post-gate", anc_coords=layout.anc_coords
        )

        # standard experiment (basis is determined from layout)
        circuit = stability_experiment(
            model=model,
            layout=layout,
            detectors=detectors,
            num_rounds=5,
            data_init={q: 0 for q in layout.data_qubits},
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
        with pytest.raises(ValueError):
            circuit = stability_experiment(
                model=model,
                layout=layout,
                detectors=detectors,
                num_rounds=3,
                anc_detectors=["X1"],
                data_init={q: 0 for q in layout.data_qubits},
            )

        # try to build with just one QEC cycle
        detectors = Detectors(
            layout.anc_qubits, frame="post-gate", include_gauge_dets=False
        )
        with pytest.raises(ValueError):
            circuit = stability_experiment(
                model=model,
                layout=layout,
                detectors=detectors,
                num_rounds=1,
                anc_detectors=["X1"],
                data_init={q: 0 for q in layout.data_qubits},
            )

        # without gauge detectors
        detectors = Detectors(
            layout.anc_qubits, frame="post-gate", include_gauge_dets=False
        )
        circuit = stability_experiment(
            model=model,
            layout=layout,
            detectors=detectors,
            num_rounds=3,
            anc_detectors=["X1"],
            data_init={q: 0 for q in layout.data_qubits},
        )

        num_anc = len(layout.anc_qubits)
        num_anc_o = len(layout.get_qubits(role="anc", stab_type=other_stab_type))
        assert circuit.num_detectors == 3 * num_anc + num_anc_o

        non_zero_dets = []
        for instr in circuit.flattened():
            if instr.name == "DETECTOR" and len(instr.targets_copy()) != 0:
                non_zero_dets.append(instr)

        assert len(non_zero_dets) == 2 + (other_stab_type == "x_type") * 2

    return
