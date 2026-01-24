from copy import deepcopy

from stim import Circuit

from ..circuit_blocks.decorators import (
    LogOpCallable,
    qubit_init_x,
    qubit_init_z,
)
from ..circuit_blocks.small_stellated_dodecahedron_code import (
    gate_to_iterator,
    init_qubits_iterator,
    log_fold_trans_h_iterator,
    log_fold_trans_s_iterator,
    log_fold_trans_swap_a_iterator,
    log_fold_trans_swap_b_iterator,
    log_fold_trans_swap_c_iterator,
    log_fold_trans_swap_r_iterator,
    log_fold_trans_swap_s_iterator,
)
from ..detectors.detectors import Detectors
from ..layouts.layout import Layout
from ..models.model import Model
from . import templates
from .arbitrary_experiment import Schedule, experiment_from_schedule


def memory_experiment(
    *args,
    gate_to_iterator: dict[str, LogOpCallable] = gate_to_iterator,
    init_qubits_iterator: LogOpCallable | None = init_qubits_iterator,
    **kargs,
) -> Circuit:
    """For information, see ``surface_sim.experiments.templates.memory_experiment``."""
    return templates.memory_experiment(
        *args,
        gate_to_iterator=gate_to_iterator,
        init_qubits_iterator=init_qubits_iterator,
        **kargs,
    )


def repeated_s_like_experiment(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    num_s_gates: int,
    num_rounds_per_gate: int,
    gate_to_iterator: dict[str, LogOpCallable] = gate_to_iterator,
    init_qubits_iterator: LogOpCallable | None = init_qubits_iterator,
    data_init: dict[str, int] | None = None,
    rot_basis: bool = False,
    anc_reset: bool = True,
    anc_detectors: list[str] | None = None,
) -> Circuit:
    """Returns the circuit for running a repeated-(S-like) experiment.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout.
    detectors
        Detector definitions to use.
    num_s_gates
        Number of logical (transversal) S-like gates to run in the experiment.
    num_rounds_per_gate
        Number of QEC round to be run after each logical S-like gate.
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

    schedule: Schedule = [
        [(gate_to_iterator[f"R{b}"], layout)],
        [(gate_to_iterator["TICK"], layout)],
    ]
    for _ in range(num_s_gates):
        schedule.append([(log_fold_trans_s_iterator, layout)])
        for _ in range(num_rounds_per_gate):
            schedule.append([(gate_to_iterator["TICK"], layout)])

    schedule.append([(gate_to_iterator[f"M{b}"], layout)])

    experiment = experiment_from_schedule(
        schedule, model, detectors, anc_reset=anc_reset, anc_detectors=anc_detectors
    )

    return experiment


def repeated_h_like_experiment(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    num_h_gates: int,
    num_rounds_per_gate: int,
    gate_to_iterator: dict[str, LogOpCallable] = gate_to_iterator,
    init_qubits_iterator: LogOpCallable | None = init_qubits_iterator,
    data_init: dict[str, int] | None = None,
    rot_basis: bool = False,
    anc_reset: bool = True,
    anc_detectors: list[str] | None = None,
) -> Circuit:
    """Returns the circuit for running a repeated-(H-like) experiment.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout.
    detectors
        Detector definitions to use.
    num_h_gates
        Number of logical (transversal) H-like gates to run in the experiment.
    num_rounds_per_gate
        Number of QEC round to be run after each logical H-like gate.
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

    schedule: Schedule = [
        [(gate_to_iterator[f"R{b}"], layout)],
        [(gate_to_iterator["TICK"], layout)],
    ]
    for _ in range(num_h_gates):
        schedule.append([(log_fold_trans_h_iterator, layout)])
        for _ in range(num_rounds_per_gate):
            schedule.append([(gate_to_iterator["TICK"], layout)])

    schedule.append([(gate_to_iterator[f"M{b}"], layout)])

    experiment = experiment_from_schedule(
        schedule, model, detectors, anc_reset=anc_reset, anc_detectors=anc_detectors
    )

    return experiment


def repeated_swap_r_like_experiment(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    num_swap_gates: int,
    num_rounds_per_gate: int,
    gate_to_iterator: dict[str, LogOpCallable] = gate_to_iterator,
    init_qubits_iterator: LogOpCallable | None = init_qubits_iterator,
    data_init: dict[str, int] | None = None,
    rot_basis: bool = False,
    anc_reset: bool = True,
    anc_detectors: list[str] | None = None,
) -> Circuit:
    """Returns the circuit for running a repeated-(SWAP-like) experiment,
    in particular the SWAP from :math:`\\sigma_r`.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout.
    detectors
        Detector definitions to use.
    num_swap_gates
        Number of logical (transversal) SWAP-like gates to run in the experiment.
    num_rounds_per_gate
        Number of QEC round to be run after each logical SWAP-like gate.
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

    if not isinstance(num_swap_gates, int):
        raise ValueError(
            f"'num_swap_gates' expected as int, got {type(num_swap_gates)} instead."
        )
    if num_swap_gates < 0:
        raise ValueError("'num_swap_gates' needs to be a positive integer.")

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

    schedule: Schedule = [
        [(gate_to_iterator[f"R{b}"], layout)],
        [(gate_to_iterator["TICK"], layout)],
    ]
    for _ in range(num_swap_gates):
        schedule.append([(log_fold_trans_swap_r_iterator, layout)])
        for _ in range(num_rounds_per_gate):
            schedule.append([(gate_to_iterator["TICK"], layout)])

    schedule.append([(gate_to_iterator[f"M{b}"], layout)])

    experiment = experiment_from_schedule(
        schedule, model, detectors, anc_reset=anc_reset, anc_detectors=anc_detectors
    )

    return experiment


def repeated_swap_s_like_experiment(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    num_swap_gates: int,
    num_rounds_per_gate: int,
    gate_to_iterator: dict[str, LogOpCallable] = gate_to_iterator,
    init_qubits_iterator: LogOpCallable | None = init_qubits_iterator,
    data_init: dict[str, int] | None = None,
    rot_basis: bool = False,
    anc_reset: bool = True,
    anc_detectors: list[str] | None = None,
) -> Circuit:
    """Returns the circuit for running a repeated-(SWAP-like) experiment,
    in particular the SWAP from :math:`\\sigma_s`.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout.
    detectors
        Detector definitions to use.
    num_swap_gates
        Number of logical (transversal) SWAP-like gates to run in the experiment.
    num_rounds_per_gate
        Number of QEC round to be run after each logical SWAP-like gate.
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

    if not isinstance(num_swap_gates, int):
        raise ValueError(
            f"'num_swap_gates' expected as int, got {type(num_swap_gates)} instead."
        )
    if num_swap_gates < 0:
        raise ValueError("'num_swap_gates' needs to be a positive integer.")

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

    schedule: Schedule = [
        [(gate_to_iterator[f"R{b}"], layout)],
        [(gate_to_iterator["TICK"], layout)],
    ]
    for _ in range(num_swap_gates):
        schedule.append([(log_fold_trans_swap_s_iterator, layout)])
        for _ in range(num_rounds_per_gate):
            schedule.append([(gate_to_iterator["TICK"], layout)])

    schedule.append([(gate_to_iterator[f"M{b}"], layout)])

    experiment = experiment_from_schedule(
        schedule, model, detectors, anc_reset=anc_reset, anc_detectors=anc_detectors
    )

    return experiment


def repeated_swap_a_like_experiment(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    num_swap_gates: int,
    num_rounds_per_gate: int,
    gate_to_iterator: dict[str, LogOpCallable] = gate_to_iterator,
    init_qubits_iterator: LogOpCallable | None = init_qubits_iterator,
    data_init: dict[str, int] | None = None,
    rot_basis: bool = False,
    anc_reset: bool = True,
    anc_detectors: list[str] | None = None,
) -> Circuit:
    """Returns the circuit for running a repeated-(SWAP-like) experiment.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout.
    detectors
        Detector definitions to use.
    num_swap_gates
        Number of logical (transversal) SWAP-like gates to run in the experiment.
    num_rounds_per_gate
        Number of QEC round to be run after each logical SWAP-like gate.
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
        If ``True``, the experiment is performed in the X basis.
        If ``False``, the experiment is performed in the Z basis.
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

    if not isinstance(num_swap_gates, int):
        raise ValueError(
            f"'num_swap_gates' expected as int, got {type(num_swap_gates)} instead."
        )
    if num_swap_gates < 0:
        raise ValueError("'num_swap_gates' needs to be a positive integer.")

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

    schedule: Schedule = [
        [(gate_to_iterator[f"R{b}"], layout)],
        [(gate_to_iterator["TICK"], layout)],
    ]
    for _ in range(num_swap_gates):
        schedule.append([(log_fold_trans_swap_a_iterator, layout)])
        for _ in range(num_rounds_per_gate):
            schedule.append([(gate_to_iterator["TICK"], layout)])

    schedule.append([(gate_to_iterator[f"M{b}"], layout)])

    experiment = experiment_from_schedule(
        schedule, model, detectors, anc_reset=anc_reset, anc_detectors=anc_detectors
    )

    return experiment


def repeated_swap_b_like_experiment(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    num_swap_gates: int,
    num_rounds_per_gate: int,
    gate_to_iterator: dict[str, LogOpCallable] = gate_to_iterator,
    init_qubits_iterator: LogOpCallable | None = init_qubits_iterator,
    data_init: dict[str, int] | None = None,
    rot_basis: bool = False,
    anc_reset: bool = True,
    anc_detectors: list[str] | None = None,
) -> Circuit:
    """Returns the circuit for running a repeated-(SWAP-like) experiment.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout.
    detectors
        Detector definitions to use.
    num_swap_gates
        Number of logical (transversal) SWAP-like gates to run in the experiment.
    num_rounds_per_gate
        Number of QEC round to be run after each logical SWAP-like gate.
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
        If ``True``, the experiment is performed in the X basis.
        If ``False``, the experiment is performed in the Z basis.
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

    if not isinstance(num_swap_gates, int):
        raise ValueError(
            f"'num_swap_gates' expected as int, got {type(num_swap_gates)} instead."
        )
    if num_swap_gates < 0:
        raise ValueError("'num_swap_gates' needs to be a positive integer.")

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

    schedule: Schedule = [
        [(gate_to_iterator[f"R{b}"], layout)],
        [(gate_to_iterator["TICK"], layout)],
    ]
    for _ in range(num_swap_gates):
        schedule.append([(log_fold_trans_swap_b_iterator, layout)])
        for _ in range(num_rounds_per_gate):
            schedule.append([(gate_to_iterator["TICK"], layout)])

    schedule.append([(gate_to_iterator[f"M{b}"], layout)])

    experiment = experiment_from_schedule(
        schedule, model, detectors, anc_reset=anc_reset, anc_detectors=anc_detectors
    )

    return experiment


def repeated_swap_c_like_experiment(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    num_swap_gates: int,
    num_rounds_per_gate: int,
    gate_to_iterator: dict[str, LogOpCallable] = gate_to_iterator,
    init_qubits_iterator: LogOpCallable | None = init_qubits_iterator,
    data_init: dict[str, int] | None = None,
    rot_basis: bool = False,
    anc_reset: bool = True,
    anc_detectors: list[str] | None = None,
) -> Circuit:
    """Returns the circuit for running a repeated-(SWAP-like) experiment.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout.
    detectors
        Detector definitions to use.
    num_swap_gates
        Number of logical (transversal) SWAP-like gates to run in the experiment.
    num_rounds_per_gate
        Number of QEC round to be run after each logical SWAP-like gate.
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
        If ``True``, the experiment is performed in the X basis.
        If ``False``, the experiment is performed in the Z basis.
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

    if not isinstance(num_swap_gates, int):
        raise ValueError(
            f"'num_swap_gates' expected as int, got {type(num_swap_gates)} instead."
        )
    if num_swap_gates < 0:
        raise ValueError("'num_swap_gates' needs to be a positive integer.")

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

    schedule: Schedule = [
        [(gate_to_iterator[f"R{b}"], layout)],
        [(gate_to_iterator["TICK"], layout)],
    ]
    for _ in range(num_swap_gates):
        schedule.append([(log_fold_trans_swap_c_iterator, layout)])
        for _ in range(num_rounds_per_gate):
            schedule.append([(gate_to_iterator["TICK"], layout)])

    schedule.append([(gate_to_iterator[f"M{b}"], layout)])

    experiment = experiment_from_schedule(
        schedule, model, detectors, anc_reset=anc_reset, anc_detectors=anc_detectors
    )

    return experiment
