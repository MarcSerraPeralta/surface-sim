import pytest
import stim

from surface_sim.layouts import (
    rot_surface_codes,
    rot_surface_stability_rectangle,
)

import surface_sim.experiments as exp
from surface_sim.models import NoiselessModel
from surface_sim import Detectors


def test_memory_experiments():
    TESTS = [
        (
            rot_surface_codes(1, 3)[0],
            exp.rot_surface_code_css.memory_experiment_mid_cycle,
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


def test_repeated_cnot_experiments():
    TESTS = [
        (
            rot_surface_codes(2, 3),
            exp.rot_surface_code_css.repeated_cnot_experiment_mid_cycle,
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
                anc_reset=True,
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
            anc_reset=True,
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
            anc_reset=True,
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


def test_stability_experiments():
    TESTS = [
        (
            rot_surface_stability_rectangle("z_type", 3, 3),
            exp.rot_surface_code_css.stability_experiment_mid_cycle,
            "x_type",
        ),
        (
            rot_surface_stability_rectangle("x_type", 3, 3),
            exp.rot_surface_code_css.stability_experiment_mid_cycle,
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

        if layout.code != "repetition_stability":
            assert len(non_zero_dets) == 2 + (other_stab_type == "x_type") * 2
        else:
            # repetition codes do not have stabilizers of the other type, thus
            # they won't be defined for the logical measurement or initialization
            if other_stab_type == "z_type":
                assert len(non_zero_dets) == 2
            else:
                assert len(non_zero_dets) == 0

    return
