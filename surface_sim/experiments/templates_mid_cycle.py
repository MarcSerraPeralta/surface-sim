from collections.abc import Collection, Sequence
from copy import deepcopy
from stim import Circuit, CircuitInstruction, GateTarget

from .arbitrary_experiment import (
    experiment_from_schedule,
    schedule_from_mid_cycle_circuit,
)
from ..layouts.layout import Layout
from ..models import Model
from ..detectors import Detectors
from ..circuit_blocks.decorators import (
    qubit_init_z,
    qubit_init_x,
    LogOpCallable,
)


def memory_experiment(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    num_rounds: int,
    gate_to_iterator: dict[str, LogOpCallable],
    tick_iterators: Sequence[LogOpCallable],
    init_qubits_iterator: LogOpCallable | None = None,
    data_init: dict[str, int] | None = None,
    rot_basis: bool = False,
    anc_reset: bool = True,
    anc_detectors: Collection[str] | None = None,
) -> Circuit:
    """Returns the circuit for running a memory experiment for a mid-cycle code.

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
        Dictionary mapping the names of stim circuit instructions used in ``circuit``
        to the functions that generate the equivalent logical circuit.
        The value for ``TICK`` will be ommited as they should be specified in
        ``tick_iterators``.
    tick_iterators
        Sequence of (half) QEC round iterators that are going to be used in cyclic
        order when encountering ``TICK`` instructions. Each iterator is applied
        to all the active layouts.
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

    Notes
    -----
    One round is considered to be going from the end-cycle state, to the mid-cycle state, and
    coming back (to the end-cycle state).
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

    unencoded_circuit = Circuit(f"R{b} 0" + "\nTICK" * 2 * num_rounds + f"\nM{b} 0")
    schedule = schedule_from_mid_cycle_circuit(
        unencoded_circuit,
        layouts=[layout],
        gate_to_iterator=gate_to_iterator,
        tick_iterators=tick_iterators,
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
    tick_iterators: Sequence[LogOpCallable],
    init_qubits_iterator: LogOpCallable | None = None,
    data_init: dict[str, int] | None = None,
    cnot_orientation: str = "alternating",
    rot_basis: bool = False,
    anc_reset: bool = True,
    anc_detectors: Collection[str] | None = None,
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
        Dictionary mapping the names of stim circuit instructions used in ``circuit``
        to the functions that generate the equivalent logical circuit.
        The value for ``TICK`` will be ommited as they should be specified in
        ``tick_iterators``.
    tick_iterators
        Sequence of (half) QEC round iterators that are going to be used in cyclic
        order when encountering ``TICK`` instructions. Each iterator is applied
        to all the active layouts.
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

    Notes
    -----
    One round is considered to be going from the end-cycle state, to the mid-cycle state, and
    coming back (to the end-cycle state).
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

    unencoded_circuit_str = f"R{b} 0 1\nTICK"
    for r in range(num_cnot_gates):
        if (r % 2 == 1) and (cnot_orientation == "alternating"):
            unencoded_circuit_str += "\nCX 1 0" + "\nTICK\nTICK" * num_rounds_per_gate
        else:
            unencoded_circuit_str += "\nCX 0 1" + "\nTICK\nTICK" * num_rounds_per_gate
    unencoded_circuit_str += f"\nTICK\nM{b} 0 1"
    unencoded_circuit = Circuit(unencoded_circuit_str)

    schedule = schedule_from_mid_cycle_circuit(
        unencoded_circuit,
        layouts=[layout_c, layout_t],
        gate_to_iterator=gate_to_iterator,
        tick_iterators=tick_iterators,
    )
    experiment = experiment_from_schedule(
        schedule, model, detectors, anc_reset=anc_reset, anc_detectors=anc_detectors
    )

    return experiment


def stability_experiment(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    num_rounds: int,
    gate_to_iterator: dict[str, LogOpCallable],
    tick_iterators: Sequence[LogOpCallable],
    init_qubits_iterator: LogOpCallable | None = None,
    data_init: dict[str, int] | None = None,
    anc_reset: bool = True,
    anc_detectors: Collection[str] | None = None,
    obs_def_rounds: Collection[int] = (1,),
) -> Circuit:
    """Returns the circuit for running a memory experiment.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout. It must contain one observable.
    detectors
        Detector definitions to use.
    num_rounds
        Number of QEC round to run in the memory experiment.
    gate_to_iterator
        Dictionary mapping the names of stim circuit instructions used in ``circuit``
        to the functions that generate the equivalent logical circuit.
        The value for ``TICK`` will be ommited as they should be specified in
        ``tick_iterators``.
    tick_iterators
        Sequence of (half) QEC round iterators that are going to be used in cyclic
        order when encountering ``TICK`` instructions. Each iterator is applied
        to all the active layouts.
    init_qubits_iterator
        If ``data_init`` is not ``None``, the reset iterator is built from
        this specified function. It should have the following inputs:
        ``(model, layout, data_init, rot_basis)`` and return a valid
        generator for the initialization of the data qubits. By default, ``None``.
    data_init
        Bitstring for initializing the data qubits. By default ``None`` mearning
        that it initializes the qubits using the reset given by ``gate_to_iterator``.
    anc_reset
        If ``True``, ancillas are reset at the beginning of the QEC round.
        By default ``True``.
    anc_detectors
        List of ancilla qubits for which to define the detectors.
        If ``None``, adds all detectors.
        By default ``None``.
    obs_def_rounds
        Rounds involved in the observable definition. By default ``[1]``, which
        means that the observable for the stability experiment is defined as the XOR
        of the outcomes of the ancilla qubits in ``layout.observables[0]`` from the
        first QEC round. The default definition is valid for any stabilitiy experiment.
        This function does not check if the provided rounds generate a correct
        observable definition. For example, ``obs_def_rounds = [2]`` if ``anc_reset=False``
        is not a valid observable definition as it corresponds to a product of detectors.

    Notes
    -----
    In a stability experiment, the data qubits are initialized in the opposite basis
    as the stabilizer type that defines the observable. Gauge detectors must not be
    included as they reveal the observable.

    One round is considered to be going from the end-cycle state, to the mid-cycle state, and
    coming back (to the end-cycle state).
    """
    if not isinstance(num_rounds, int):
        raise ValueError(
            f"'num_rounds' expected as int, got {type(num_rounds)} instead."
        )
    if num_rounds < 2:
        raise ValueError("'num_rounds' needs to be a positive integer larger than 1.")
    if not isinstance(obs_def_rounds, Collection):
        raise TypeError(
            f"'obs_def_rounds' must be a Collection, but {type(obs_def_rounds)} was given."
        )
    if any(not isinstance(r, int) for r in obs_def_rounds):
        raise TypeError("The elements of 'obs_def_rounds' must be integers.")
    if not isinstance(detectors, Detectors):
        raise TypeError(
            f"'detectors' must be a Detectors, but {type(detectors)} was given."
        )
    if detectors.include_gauge_dets:
        raise ValueError(
            "In stability experiments, detectors.include_gauge_dets must be False."
        )
    if not isinstance(layout, Layout):
        raise TypeError(f"'layout' must be a Layout, but {type(layout)} was given.")
    if layout.num_observables != 1:
        raise ValueError(
            f"'layout' must contain one observable, but it contains {layout.num_observables}."
        )
    observable = layout.observable_definition(layout.observables[0])
    stab_types = [layout.param("stab_type", q) for q in observable]
    if set(stab_types) not in (set(["x_type"]), set(["z_type"])):
        raise ValueError(
            f"The layout observable contains more than one type of stabilizers: {stab_types}."
        )
    rot_basis = stab_types[0] == "z_type"
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

    unencoded_circuit = Circuit(f"R{b} 0" + "\nTICK" * 2 * num_rounds + f"\nM{b} 0")
    schedule = schedule_from_mid_cycle_circuit(
        unencoded_circuit,
        layouts=[layout],
        gate_to_iterator=gate_to_iterator,
        tick_iterators=tick_iterators,
    )
    experiment = experiment_from_schedule(
        schedule, model, detectors, anc_reset=anc_reset, anc_detectors=anc_detectors
    )

    # layout does not contain any logical qubit and thus there is no OBSERVABLE defined
    targets: list[GateTarget] = []
    for r in obs_def_rounds:
        for q in observable:
            targets.append(model.meas_target(q, -num_rounds - 1 + r))
    obs_instr = CircuitInstruction(
        name="OBSERVABLE_INCLUDE", gate_args=[0], targets=targets
    )
    experiment.append(obs_instr)

    return experiment
