from copy import deepcopy
from stim import Circuit

from ..circuit_blocks.unrot_surface_code_css import (
    init_qubits_iterator,
    log_meas_iterator,
    gate_to_iterator,
)
from .arbitrary_experiment import experiment_from_schedule, schedule_from_circuit
from ..layouts.layout import Layout
from ..models import Model
from ..detectors import Detectors
from ..circuit_blocks.decorators import (
    qubit_init_z,
    qubit_init_x,
    logical_measurement_z,
    logical_measurement_x,
)


def memory_experiment(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    num_rounds: int,
    data_init: dict[str, int] | list[int],
    rot_basis: bool = False,
    anc_reset: bool = True,
    anc_detectors: list[str] | None = None,
) -> Circuit:
    """Returns the circuit for running a memory experiment.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout.
    detectors
        Detector definitions to use.
    num_rounds
        Number of QEC cycle to run in the memory experiment.
    data_init
        Bitstring for initializing the data qubits.
    rot_basis
        If ``True``, the memory experiment is performed in the X basis.
        If ``False``, the memory experiment is performed in the Z basis.
        By deafult ``False``.
    anc_reset
        If ``True``, ancillas are reset at the beginning of the QEC cycle.
        By default ``True``.
    anc_detectors
        List of ancilla qubits for which to define the detectors.
        If ``None``, adds all detectors.
        By default ``None``.
    """
    if not isinstance(num_rounds, int):
        raise ValueError(
            f"'num_rounds' expected as int, got {type(num_rounds)} instead."
        )
    if num_rounds < 0:
        raise ValueError("'num_rounds' needs to be a positive integer.")
    if not isinstance(data_init, dict):
        raise TypeError(f"'data_init' must be a dict, but {type(data_init)} was given.")

    reset = qubit_init_x if rot_basis else qubit_init_z
    meas = logical_measurement_x if rot_basis else logical_measurement_z

    @reset
    def custom_reset_iterator(m: Model, l: Layout):
        return init_qubits_iterator(m, l, data_init=data_init, rot_basis=rot_basis)

    @meas
    def custom_measurement_iterator(m: Model, l: Layout):
        return log_meas_iterator(m, l, rot_basis=rot_basis)

    custom_gate_to_iterator = deepcopy(gate_to_iterator)
    custom_gate_to_iterator["R"] = custom_reset_iterator
    custom_gate_to_iterator["M"] = custom_measurement_iterator

    unencoded_circuit = Circuit("R 0" + "\nTICK" * num_rounds + "\nM 0")
    schedule = schedule_from_circuit(
        unencoded_circuit, layouts=[layout], gate_to_iterator=custom_gate_to_iterator
    )
    experiment = experiment_from_schedule(
        schedule, model, detectors, anc_reset=anc_reset, anc_detectors=anc_detectors
    )

    return experiment


def repeated_s_experiment(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    num_s_gates: int,
    num_rounds_per_gate: int,
    data_init: dict[str, int] | list[int],
    rot_basis: bool = False,
    anc_reset: bool = True,
    anc_detectors: list[str] | None = None,
) -> Circuit:
    """Returns the circuit for running a repeated-S experiment.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout.
    detectors
        Detector definitions to use.
    num_s_gates
        Number of logical (transversal) S gates to run in the experiment.
    num_rounds_per_gate
        Number of QEC cycles to be run after each logical S gate.
    data_init
        Bitstring for initializing the data qubits.
    rot_basis
        If ``True``, the repeated-S experiment is performed in the X basis.
        If ``False``, the repeated-S experiment is performed in the Z basis.
        By deafult ``False``.
    anc_reset
        If ``True``, ancillas are reset at the beginning of the QEC cycle.
        By default ``True``.
    anc_detectors
        List of ancilla qubits for which to define the detectors.
        If ``None``, adds all detectors.
        By default ``None``.
    """
    if not isinstance(num_rounds_per_gate, int):
        raise ValueError(
            f"'num_rounds_per_gate' expected as int, got {type(num_rounds_per_gate)} instead."
        )
    if num_rounds_per_gate < 0:
        raise ValueError("'num_rounds_per_gate' needs to be a positive integer.")

    if not isinstance(num_s_gates, int):
        raise ValueError(
            f"'num_s_gates' expected as int, got {type(num_s_gates)} instead."
        )
    if (num_s_gates < 0) or (num_s_gates % 2 == 1):
        raise ValueError("'num_s_gates' needs to be an even positive integer.")

    if not isinstance(data_init, dict):
        raise TypeError(f"'data_init' must be a dict, but {type(data_init)} was given.")

    if not isinstance(layout, Layout):
        raise TypeError(f"'layout' must be a layout, but {type(layout)} was given.")

    reset = qubit_init_x if rot_basis else qubit_init_z
    meas = logical_measurement_x if rot_basis else logical_measurement_z

    @reset
    def custom_reset_iterator(m: Model, l: Layout):
        return init_qubits_iterator(m, l, data_init=data_init, rot_basis=rot_basis)

    @meas
    def custom_measurement_iterator(m: Model, l: Layout):
        return log_meas_iterator(m, l, rot_basis=rot_basis)

    custom_gate_to_iterator = deepcopy(gate_to_iterator)
    custom_gate_to_iterator["R"] = custom_reset_iterator
    custom_gate_to_iterator["M"] = custom_measurement_iterator

    unencoded_circuit = Circuit(
        "R 0\nTICK" + ("\nS 0" + "\nTICK" * num_rounds_per_gate) * num_s_gates + "\nM 0"
    )
    schedule = schedule_from_circuit(
        unencoded_circuit, layouts=[layout], gate_to_iterator=custom_gate_to_iterator
    )
    experiment = experiment_from_schedule(
        schedule, model, detectors, anc_reset=anc_reset, anc_detectors=anc_detectors
    )

    return experiment


def repeated_h_experiment(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    num_h_gates: int,
    num_rounds_per_gate: int,
    data_init: dict[str, int] | list[int],
    rot_basis: bool = False,
    anc_reset: bool = True,
    anc_detectors: list[str] | None = None,
) -> Circuit:
    """Returns the circuit for running a repeated-H experiment.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout.
    detectors
        Detector definitions to use.
    num_h_gates
        Number of logical (transversal) H gates to run in the experiment.
    num_rounds_per_gate
        Number of QEC cycles to be run after each logical H gate.
    data_init
        Bitstring for initializing the data qubits.
    rot_basis
        If ``True``, the logical qubit is initialized in the X basis.
        If ``False``, the logical qubit is initialized in the Z basis.
        By deafult ``False``.
    anc_reset
        If ``True``, ancillas are reset at the beginning of the QEC cycle.
        By default ``True``.
    anc_detectors
        List of ancilla qubits for which to define the detectors.
        If ``None``, adds all detectors.
        By default ``None``.
    """
    if not isinstance(num_rounds_per_gate, int):
        raise ValueError(
            f"'num_rounds_per_gate' expected as int, got {type(num_rounds_per_gate)} instead."
        )
    if num_rounds_per_gate < 0:
        raise ValueError("'num_rounds_per_gate' needs to be a positive integer.")

    if not isinstance(num_h_gates, int):
        raise ValueError(
            f"'num_h_gates' expected as int, got {type(num_h_gates)} instead."
        )
    if (num_h_gates < 0) or (num_h_gates % 2 == 1):
        raise ValueError("'num_h_gates' needs to be an even positive integer.")

    if not isinstance(data_init, dict):
        raise TypeError(f"'data_init' must be a dict, but {type(data_init)} was given.")

    if not isinstance(layout, Layout):
        raise TypeError(f"'layout' must be a layout, but {type(layout)} was given.")

    reset = qubit_init_x if rot_basis else qubit_init_z
    meas = logical_measurement_x if rot_basis else logical_measurement_z

    @reset
    def custom_reset_iterator(m: Model, l: Layout):
        return init_qubits_iterator(m, l, data_init=data_init, rot_basis=rot_basis)

    @meas
    def custom_measurement_iterator(m: Model, l: Layout):
        return log_meas_iterator(m, l, rot_basis=rot_basis)

    custom_gate_to_iterator = deepcopy(gate_to_iterator)
    custom_gate_to_iterator["R"] = custom_reset_iterator
    custom_gate_to_iterator["M"] = custom_measurement_iterator

    unencoded_circuit = Circuit(
        "R 0\nTICK" + ("\nH 0" + "\nTICK" * num_rounds_per_gate) * num_h_gates + "\nM 0"
    )
    schedule = schedule_from_circuit(
        unencoded_circuit, layouts=[layout], gate_to_iterator=custom_gate_to_iterator
    )
    experiment = experiment_from_schedule(
        schedule, model, detectors, anc_reset=anc_reset, anc_detectors=anc_detectors
    )

    return experiment


def repeated_cnot_experiment(
    model: Model,
    layout_c: Layout,
    layout_t: Layout,
    detectors: Detectors,
    num_cnot_gates: int,
    num_rounds_per_gate: int,
    data_init: dict[str, int] | list[int],
    cnot_orientation: str = "alternating",
    rot_basis: bool = False,
    anc_reset: bool = True,
    anc_detectors: list[str] | None = None,
) -> Circuit:
    """Returns the circuit for running a repeated-CNOT experiment.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout_c
        Code layout for the control of the transversal CNOT, unless
        ``cnot_orientation = "alternating"``.
    layout_t
        Code layout for the target of the transversal CNOT, unless
        ``cnot_orientation = "alternating"``.
    detectors
        Detector definitions to use.
    num_cnot_gates
        Number of logical (transversal) CNOT gates to run in the experiment.
    num_rounds_per_gate
        Number of QEC cycles to be run after each logical CNOT gate.
    data_init
        Bitstring for initializing the data qubits.
    cnot_orientation
        Determines the orientation of the CNOTS in this circuit.
        The options are ``"constant"`` and ``"alternating"``.
        By default ``"alternating"``.
    rot_basis
        If ``True``, the repeated-CNOT experiment is performed in the X basis.
        If ``False``, the repeated-CNOT experiment is performed in the Z basis.
        By deafult ``False``.
    anc_reset
        If ``True``, ancillas are reset at the beginning of the QEC cycle.
        By default ``True``.
    anc_detectors
        List of ancilla qubits for which to define the detectors.
        If ``None``, adds all detectors.
        By default ``None``.
    """
    if not isinstance(num_rounds_per_gate, int):
        raise ValueError(
            f"'num_rounds_per_gate' expected as int, got {type(num_rounds_per_gate)} instead."
        )
    if num_rounds_per_gate < 0:
        raise ValueError("'num_rounds_per_gate' needs to be a positive integer.")

    if not isinstance(num_cnot_gates, int):
        raise ValueError(
            f"'num_s_gates' expected as int, got {type(num_cnot_gates)} instead."
        )
    if num_cnot_gates < 0:
        raise ValueError("'num_cnot_gates' needs to be a positive integer.")

    if not isinstance(data_init, dict):
        raise TypeError(f"'data_init' must be a dict, but {type(data_init)} was given.")

    if not isinstance(layout_c, Layout):
        raise TypeError(f"'layout_c' must be a Layout, but {type(layout_c)} was given.")
    if not isinstance(layout_t, Layout):
        raise TypeError(f"'layout_t' must be a Layout, but {type(layout_t)} was given.")

    if cnot_orientation not in ["constant", "alternating"]:
        raise TypeError(
            "'cnot_orientation' must be 'constant' or 'alternating'"
            f" but {cnot_orientation} was given."
        )

    reset = qubit_init_x if rot_basis else qubit_init_z
    meas = logical_measurement_x if rot_basis else logical_measurement_z

    @reset
    def custom_reset_iterator(m: Model, l: Layout):
        return init_qubits_iterator(m, l, data_init=data_init, rot_basis=rot_basis)

    @meas
    def custom_measurement_iterator(m: Model, l: Layout):
        return log_meas_iterator(m, l, rot_basis=rot_basis)

    custom_gate_to_iterator = deepcopy(gate_to_iterator)
    custom_gate_to_iterator["R"] = custom_reset_iterator
    custom_gate_to_iterator["M"] = custom_measurement_iterator

    unencoded_circuit_str = "R 0 1\nTICK"
    for r in range(num_cnot_gates):
        if (r % 2 == 1) and (cnot_orientation == "alternating"):
            unencoded_circuit_str += "\nCX 1 0" + "\nTICK" * num_rounds_per_gate
        else:
            unencoded_circuit_str += "\nCX 0 1" + "\nTICK" * num_rounds_per_gate
    unencoded_circuit_str += "\nM 0 1"
    unencoded_circuit = Circuit(unencoded_circuit_str)

    schedule = schedule_from_circuit(
        unencoded_circuit,
        layouts=[layout_c, layout_t],
        gate_to_iterator=custom_gate_to_iterator,
    )
    experiment = experiment_from_schedule(
        schedule, model, detectors, anc_reset=anc_reset, anc_detectors=anc_detectors
    )

    return experiment
