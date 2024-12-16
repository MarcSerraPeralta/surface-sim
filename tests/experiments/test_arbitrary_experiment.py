import stim

from surface_sim import Layout
from surface_sim.models import NoiselessModel
from surface_sim import Detectors
from surface_sim.experiments import schedule_from_circuit, experiment_from_schedule
from surface_sim.circuit_blocks.unrot_surface_code_css import gate_to_iterator
from surface_sim.layouts import unrot_surface_codes


def test_schedule_from_circuit():
    layouts = unrot_surface_codes(4, distance=3)
    circuit = stim.Circuit(
        """
        R 0 1 2
        TICK
        X 0
        M 1
        TICK
        CX 0 1 2 3
        """
    )

    schedule = schedule_from_circuit(circuit, layouts, gate_to_iterator)

    assert len(schedule) == 9

    list_num_layouts = [1, 1, 1, 0, 1, 1, 0, 2, 2]
    for op, num_layouts in zip(schedule, list_num_layouts):
        assert len(op) == num_layouts + 1
        assert all(isinstance(l, Layout) for l in op[1:])
        if num_layouts != 0:
            assert op[0].log_op_type != "qec_cycle"

    return


def test_experiment_from_schedule():
    layouts = unrot_surface_codes(3, distance=3)
    qubit_inds = {}
    anc_coords = []
    anc_qubits = []
    for layout in layouts:
        qubit_inds.update(layout.qubit_inds())
        anc_qubits += layout.get_qubits(role="anc")
        anc_coords += layout.anc_coords()

    circuit = stim.Circuit(
        """
        R 0 1
        TICK
        X 1
        I 0
        TICK
        M 0
        I 1
        TICK
        """
    )
    model = NoiselessModel(qubit_inds=qubit_inds)
    detectors = Detectors(anc_qubits, frame="pre-gate")

    schedule = schedule_from_circuit(circuit, layouts, gate_to_iterator)
    experiment = experiment_from_schedule(
        schedule, model, detectors, anc_reset=True, anc_detectors=None
    )

    assert isinstance(experiment, stim.Circuit)

    # check that the detectors and logicals fulfill their
    # conditions by building the stim diagram
    dem = circuit.detector_error_model(allow_gauge_detectors=True)

    num_coords = 0
    anc_coords = {k: list(map(float, v)) for k, v in layout.anc_coords().items()}
    for dem_instr in dem:
        if dem_instr.type == "detector":
            assert dem_instr.args_copy()[:-1] in anc_coords.values()
            num_coords += 1

    assert num_coords == dem.num_detectors

    return
