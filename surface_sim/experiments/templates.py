from collections.abc import Callable
from copy import deepcopy
from stim import Circuit

from .arbitrary_experiment import experiment_from_schedule, schedule_from_circuit
from ..layouts.layout import Layout
from ..models import Model
from ..detectors import Detectors
from ..circuit_blocks.decorators import (
    qubit_init_z,
    qubit_init_x,
    LogOpCallable,
)
from ..util.observables import remove_nondeterministic_observables


def memory_experiment(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    num_rounds: int,
    gate_to_iterator: dict[str, LogOpCallable],
    init_qubits_iterator: Callable | None = None,
    data_init: dict[str, int] | None = None,
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
        Number of QEC round to run in the memory experiment.
    gate_to_iterator
        Dictonary mapping stim.CircuitInstuction names to ``LogOpCallable`` functions
        that return a generator with the physical implementation of the logical
        operation. By default, uses the default ``gate_to_iterator`` from
        ``surface_sim.circuit_blocks.rot_surface_code_css``.
    init_qubits_iterator
        If ``data_init`` is not ``None``, the reset iterator is built from
        this specified function. It should have the following inputs:
        ``(model, layout, data_init, rot_basis)`` and return a valid
        generator for the initialization of the data qubits. By default, ``None``.
    data_init
        Bitstring for initializing the data qubits. By default ``None`` mearning
        that it initializes the qubits using the reset given by ``gate_to_iterator``.
    rot_basis
        If ``True``, the memory experiment is performed in the X basis.
        If ``False``, the memory experiment is performed in the Z basis.
        By deafult ``False``.
    anc_reset
        If ``True``, ancillas are reset at the beginning of the QEC round.
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

    b = "X" if rot_basis else "Z"
    if data_init is not None:
        if init_qubits_iterator is None:
            raise TypeError(
                "As 'data_init' is not None, 'init_qubits_iterator' must not be None."
            )

        reset = qubit_init_x if rot_basis else qubit_init_z

        @reset
        def custom_reset_iterator(m: Model, l: Layout):
            return init_qubits_iterator(m, l, data_init=data_init, rot_basis=rot_basis)

        gate_to_iterator = deepcopy(gate_to_iterator)
        gate_to_iterator[f"R{b}"] = custom_reset_iterator

    unencoded_circuit = Circuit(f"R{b} 0" + "\nTICK" * num_rounds + f"\nM{b} 0")
    schedule = schedule_from_circuit(
        unencoded_circuit, layouts=[layout], gate_to_iterator=gate_to_iterator
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
    gate_to_iterator: dict[str, LogOpCallable],
    init_qubits_iterator: Callable | None = None,
    data_init: dict[str, int] | None = None,
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
        Number of QEC round to be run after each logical S gate.
    gate_to_iterator
        Dictonary mapping stim.CircuitInstuction names to ``LogOpCallable`` functions
        that return a generator with the physical implementation of the logical
        operation.
    init_qubits_iterator
        If ``data_init`` is not ``None``, the reset iterator is built from
        this specified function. It should have the following inputs:
        ``(model, layout, data_init, rot_basis)`` and return a valid
        generator for the initialization of the data qubits. By default, ``None``.
    data_init
        Bitstring for initializing the data qubits. By default ``None`` mearning
        that it initializes the qubits using the reset given by ``gate_to_iterator``.
    rot_basis
        If ``True``, the repeated-S experiment is performed in the X basis.
        If ``False``, the repeated-S experiment is performed in the Z basis.
        By deafult ``False``.
    anc_reset
        If ``True``, ancillas are reset at the beginning of the QEC round.
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

    b = "X" if rot_basis else "Z"
    if data_init is None:
        if init_qubits_iterator is None:
            raise TypeError(
                "As 'data_init' is not None, 'init_qubits_iterator' must not be None."
            )

        reset = qubit_init_x if rot_basis else qubit_init_z

        @reset
        def custom_reset_iterator(m: Model, l: Layout):
            return init_qubits_iterator(m, l, data_init=data_init, rot_basis=rot_basis)

        gate_to_iterator = deepcopy(gate_to_iterator)
        gate_to_iterator[f"R{b}"] = custom_reset_iterator

    unencoded_circuit = Circuit(
        f"R{b} 0\nTICK"
        + ("\nS 0" + "\nTICK" * num_rounds_per_gate) * num_s_gates
        + f"\nM{b} 0"
    )
    schedule = schedule_from_circuit(
        unencoded_circuit, layouts=[layout], gate_to_iterator=gate_to_iterator
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
    gate_to_iterator: dict[str, LogOpCallable],
    init_qubits_iterator: Callable | None = None,
    data_init: dict[str, int] | None = None,
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
        Number of QEC rounds to be run after each logical H gate.
    gate_to_iterator
        Dictonary mapping stim.CircuitInstuction names to ``LogOpCallable`` functions
        that return a generator with the physical implementation of the logical
        operation.
    init_qubits_iterator
        If ``data_init`` is not ``None``, the reset iterator is built from
        this specified function. It should have the following inputs:
        ``(model, layout, data_init, rot_basis)`` and return a valid
        generator for the initialization of the data qubits. By default, ``None``.
    data_init
        Bitstring for initializing the data qubits. By default ``None`` mearning
        that it initializes the qubits using the reset given by ``gate_to_iterator``.
    rot_basis
        If ``True``, the repeated-S experiment is performed in the X basis.
        If ``False``, the repeated-S experiment is performed in the Z basis.
        By deafult ``False``.
    anc_reset
        If ``True``, ancillas are reset at the beginning of the QEC round.
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

    b = "X" if rot_basis else "Z"
    if data_init is None:
        if init_qubits_iterator is None:
            raise TypeError(
                "As 'data_init' is not None, 'init_qubits_iterator' must not be None."
            )

        reset = qubit_init_x if rot_basis else qubit_init_z

        @reset
        def custom_reset_iterator(m: Model, l: Layout):
            return init_qubits_iterator(m, l, data_init=data_init, rot_basis=rot_basis)

        gate_to_iterator = deepcopy(gate_to_iterator)
        gate_to_iterator[f"R{b}"] = custom_reset_iterator

    unencoded_circuit = Circuit(
        f"R{b} 0\nTICK"
        + ("\nH 0" + "\nTICK" * num_rounds_per_gate) * num_h_gates
        + f"\nM{b} 0"
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
    gate_to_iterator: dict[str, LogOpCallable],
    init_qubits_iterator: Callable | None = None,
    data_init: dict[str, int] | None = None,
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
        Number of QEC rounds to be run after each logical CNOT gate.
    gate_to_iterator
        Dictonary mapping stim.CircuitInstuction names to ``LogOpCallable`` functions
        that return a generator with the physical implementation of the logical
        operation.
    init_qubits_iterator
        If ``data_init`` is not ``None``, the reset iterator is built from
        this specified function. It should have the following inputs:
        ``(model, layout, data_init, rot_basis)`` and return a valid
        generator for the initialization of the data qubits. By default, ``None``.
    data_init
        Bitstring for initializing the data qubits. By default ``None`` mearning
        that it initializes the qubits using the reset given by ``gate_to_iterator``.
    cnot_orientation
        Determines the orientation of the CNOTS in this circuit.
        The options are ``"constant"`` and ``"alternating"``.
        By default ``"alternating"``.
    rot_basis
        If ``True``, the repeated-CNOT experiment is performed in the X basis.
        If ``False``, the repeated-CNOT experiment is performed in the Z basis.
        By deafult ``False``.
    anc_reset
        If ``True``, ancillas are reset at the beginning of the QEC round.
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

    if cnot_orientation not in ["constant", "alternating"]:
        raise TypeError(
            "'cnot_orientation' must be 'constant' or 'alternating'"
            f" but {cnot_orientation} was given."
        )

    b = "X" if rot_basis else "Z"
    if data_init is None:
        if init_qubits_iterator is None:
            raise TypeError(
                "As 'data_init' is not None, 'init_qubits_iterator' must not be None."
            )

        reset = qubit_init_x if rot_basis else qubit_init_z

        @reset
        def custom_reset_iterator(m: Model, l: Layout):
            return init_qubits_iterator(m, l, data_init=data_init, rot_basis=rot_basis)

        gate_to_iterator = deepcopy(gate_to_iterator)
        gate_to_iterator[f"R{b}"] = custom_reset_iterator

    unencoded_circuit_str = f"R{b} 0 1\nTICK"
    for r in range(num_cnot_gates):
        if (r % 2 == 1) and (cnot_orientation == "alternating"):
            unencoded_circuit_str += "\nCX 1 0" + "\nTICK" * num_rounds_per_gate
        else:
            unencoded_circuit_str += "\nCX 0 1" + "\nTICK" * num_rounds_per_gate
    unencoded_circuit_str += f"\nM{b} 0 1"
    unencoded_circuit = Circuit(unencoded_circuit_str)

    schedule = schedule_from_circuit(
        unencoded_circuit,
        layouts=[layout_c, layout_t],
        gate_to_iterator=gate_to_iterator,
    )
    experiment = experiment_from_schedule(
        schedule, model, detectors, anc_reset=anc_reset, anc_detectors=anc_detectors
    )

    return experiment


def repeated_s_injection_experiment(
    model: Model,
    layout: Layout,
    layout_anc: Layout,
    detectors: Detectors,
    num_s_injections: int,
    num_rounds_per_gate: int,
    gate_to_iterator: dict[str, LogOpCallable],
    init_qubits_iterator: Callable | None = None,
    data_init: dict[str, int] | None = None,
    rot_basis: bool = False,
    anc_reset: bool = True,
    anc_detectors: list[str] | None = None,
) -> Circuit:
    """Returns the circuit for running a repeated-S-injection experiment.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout for the logical qubit to which the logical S gate is injected.
    layout_anc
        Code layout for the logical (ancilla) qubit used to inject the logical S gate.
    detectors
        Detector definitions to use.
    num_s_injections
        Number of logical (transversal) CNOT gates to run in the experiment.
    num_rounds_per_gate
        Number of QEC rounds to be run after each logical CNOT gate.
    gate_to_iterator
        Dictonary mapping stim.CircuitInstuction names to ``LogOpCallable`` functions
        that return a generator with the physical implementation of the logical
        operation.
    init_qubits_iterator
        If ``data_init`` is not ``None``, the reset iterator is built from
        this specified function. It should have the following inputs:
        ``(model, layout, data_init, rot_basis)`` and return a valid
        generator for the initialization of the data qubits. By default, ``None``.
    data_init
        Bitstring for initializing the data qubits. By default ``None`` mearning
        that it initializes the qubits using the reset given by ``gate_to_iterator``.
    rot_basis
        If ``True``, the repeated-CNOT experiment is performed in the X basis.
        If ``False``, the repeated-CNOT experiment is performed in the Z basis.
        By deafult ``False``.
    anc_reset
        If ``True``, ancillas are reset at the beginning of the QEC round.
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

    if not isinstance(num_s_injections, int):
        raise ValueError(
            f"'num_s_injections' expected as int, got {type(num_s_injections)} instead."
        )
    if (num_s_injections < 0) or (num_s_injections % 2 == 1):
        raise ValueError("'num_s_injections' needs to be a positive even integer.")

    b = "X" if rot_basis else "Z"
    if data_init is None:
        if init_qubits_iterator is None:
            raise TypeError(
                "As 'data_init' is not None, 'init_qubits_iterator' must not be None."
            )

        reset = qubit_init_x if rot_basis else qubit_init_z

        @reset
        def custom_reset_iterator(m: Model, l: Layout):
            return init_qubits_iterator(m, l, data_init=data_init, rot_basis=rot_basis)

        @qubit_init_x
        def custom_reset_x_iterator(m: Model, l: Layout):
            return init_qubits_iterator(m, l, data_init=data_init, rot_basis=True)

        gate_to_iterator = deepcopy(gate_to_iterator)
        gate_to_iterator[f"R{b}"] = custom_reset_iterator
        gate_to_iterator["RX"] = custom_reset_x_iterator

    unencoded_circuit_str = f"R{b} 0\nTICK"
    s_injection_circuit_str = "\nRX 1" + "\nTICK" * num_rounds_per_gate
    s_injection_circuit_str += "\nS 1" + "\nTICK" * num_rounds_per_gate
    s_injection_circuit_str += "\nCX 0 1" + "\nTICK" * num_rounds_per_gate
    s_injection_circuit_str += "\nMZ 1" + "\nTICK" * num_rounds_per_gate

    for _ in range(num_s_injections):
        unencoded_circuit_str += s_injection_circuit_str
    unencoded_circuit_str += f"\nM{b} 0"
    unencoded_circuit = Circuit(unencoded_circuit_str)

    schedule = schedule_from_circuit(
        unencoded_circuit,
        layouts=[layout, layout_anc],
        gate_to_iterator=gate_to_iterator,
    )
    experiment = experiment_from_schedule(
        schedule, model, detectors, anc_reset=anc_reset, anc_detectors=anc_detectors
    )

    # keep only deterministic observables
    if rot_basis:
        experiment = remove_nondeterministic_observables(
            experiment, [list(range(num_s_injections + 1))]
        )
    else:
        experiment = remove_nondeterministic_observables(
            experiment, [[num_s_injections]]
        )

    return experiment
