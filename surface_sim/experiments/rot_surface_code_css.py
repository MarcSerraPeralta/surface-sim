from copy import deepcopy
from stim import Circuit

from ..layouts import Layout
from ..circuit_blocks.rot_surface_code_css import (
    init_qubits,
    log_meas,
    log_meas_iterator,
    qec_round,
    qec_round_iterator,
    qubit_coords,
    log_trans_s,
    log_trans_cnot,
)
from ..models import Model
from ..detectors import Detectors
from ..util import merge_circuits, merge_qec_rounds, merge_log_meas


def memory_experiment(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    num_rounds: int,
    data_init: dict[str, int] | list[int],
    rot_basis: bool = False,
    anc_reset: bool = True,
    anc_detectors: list[str] | None = None,
    gauge_detectors: bool = True,
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
    gauge_detectors
        If ``True``, adds gauge detectors (coming from the first QEC cycle).
        If ``False``, the resulting circuit does not have gauge detectors.
        By default ``True``.
    """
    if not isinstance(num_rounds, int):
        raise ValueError(
            f"'num_rounds' expected as int, got {type(num_rounds)} instead."
        )
    if num_rounds < 0:
        raise ValueError("'num_rounds' needs to be a positive integer.")
    if not isinstance(data_init, dict):
        raise TypeError(f"'data_init' must be a dict, but {type(data_init)} was given.")
    if not isinstance(layout, Layout):
        raise TypeError(f"'layout' must be a layout, but {type(layout)} was given.")
    if anc_detectors is None:
        anc_detectors = layout.get_qubits(role="anc")

    model.new_circuit()
    detectors.new_circuit()

    experiment = Circuit()
    experiment += qubit_coords(model, layout)
    experiment += init_qubits(model, layout, detectors, data_init, rot_basis)

    for r in range(num_rounds):
        if r == 0 and (not gauge_detectors):
            stab_type = "x_type" if rot_basis else "z_type"
            stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)
            first_dets = set(anc_detectors).intersection(stab_qubits)
            experiment += qec_round(model, layout, detectors, anc_reset, first_dets)
            continue

        experiment += qec_round(model, layout, detectors, anc_reset, anc_detectors)

    experiment += log_meas(
        model, layout, detectors, rot_basis, anc_reset, anc_detectors
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
    gauge_detectors: bool = True,
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
    gauge_detectors
        If ``True``, adds gauge detectors (coming from the first QEC cycle).
        If ``False``, the resulting circuit does not have gauge detectors.
        By default ``True``.
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
    if anc_detectors is None:
        anc_detectors = layout.get_qubits(role="anc")

    model.new_circuit()
    detectors.new_circuit()

    experiment = Circuit()
    experiment += qubit_coords(model, layout)
    experiment += init_qubits(model, layout, detectors, data_init, rot_basis)

    first_dets = deepcopy(anc_detectors)
    if not gauge_detectors:
        stab_type = "x_type" if rot_basis else "z_type"
        stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)
        first_dets = set(anc_detectors).intersection(stab_qubits)

    experiment += qec_round(model, layout, detectors, anc_reset, first_dets)

    for _ in range(num_s_gates):
        experiment += log_trans_s(model, layout, detectors)
        for _ in range(num_rounds_per_gate):
            experiment += qec_round(model, layout, detectors, anc_reset, anc_detectors)
    experiment += log_meas(
        model, layout, detectors, rot_basis, anc_reset, anc_detectors
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
    gauge_detectors: bool = True,
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
    gauge_detectors
        If ``True``, adds gauge detectors (coming from the first QEC cycle).
        If ``False``, the resulting circuit does not have gauge detectors.
        By default ``True``.
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
    if anc_detectors is None:
        anc_detectors = layout_c.get_qubits(role="anc")
        anc_detectors += layout_t.get_qubits(role="anc")

    if cnot_orientation not in ["constant", "alternating"]:
        raise TypeError(
            "'cnot_orientation' must be 'constant' or 'alternating'"
            f" but {cnot_orientation} was given."
        )

    data_init_c = {k: v for k, v in data_init.items() if k in layout_c.get_qubits()}
    data_init_t = {k: v for k, v in data_init.items() if k in layout_t.get_qubits()}

    model.new_circuit()
    detectors.new_circuit()

    experiment = Circuit()
    experiment += merge_circuits(
        qubit_coords(model, layout_c),
        qubit_coords(model, layout_t),
    )
    experiment += merge_circuits(
        init_qubits(
            model, layout_c, detectors, data_init=data_init_c, rot_basis=rot_basis
        ),
        init_qubits(
            model, layout_t, detectors, data_init=data_init_t, rot_basis=rot_basis
        ),
    )

    first_dets = deepcopy(anc_detectors)
    if not gauge_detectors:
        stab_type = "x_type" if rot_basis else "z_type"
        stab_qubits = layout_c.get_qubits(role="anc", stab_type=stab_type)
        stab_qubits += layout_t.get_qubits(role="anc", stab_type=stab_type)
        first_dets = set(anc_detectors).intersection(stab_qubits)

    experiment += merge_qec_rounds(
        qec_round_iterator,
        model,
        [layout_c, layout_t],
        detectors,
        anc_reset=anc_reset,
        anc_detectors=first_dets,
    )

    for k in range(num_cnot_gates):
        if (cnot_orientation == "alternating") and (k % 2 == 1):
            # change control and target of t-CNOT
            layout_c, layout_t = layout_t, layout_c

        experiment += log_trans_cnot(model, layout_c, layout_t, detectors)

        for _ in range(num_rounds_per_gate):
            experiment += merge_qec_rounds(
                qec_round_iterator,
                model,
                [layout_c, layout_t],
                detectors,
                anc_reset=anc_reset,
                anc_detectors=anc_detectors,
            )

    experiment += merge_log_meas(
        log_meas_iterator,
        model,
        [layout_c, layout_t],
        detectors,
        rot_bases=[rot_basis, rot_basis],
        anc_reset=anc_reset,
        anc_detectors=anc_detectors,
    )

    return experiment
