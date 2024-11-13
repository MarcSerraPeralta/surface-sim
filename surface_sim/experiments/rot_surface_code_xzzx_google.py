from copy import deepcopy
from stim import Circuit

from ..layouts import Layout
from ..circuit_blocks.rot_surface_code_xzzx_google import (
    init_qubits,
    qec_round_with_log_meas,
    qec_round,
    qubit_coords,
)
from ..models import Model
from ..detectors import Detectors


def memory_experiment(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    num_rounds: int,
    data_init: dict[str, int] | list[int],
    anc_detectors: list[str] | None = None,
    rot_basis: bool = False,
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
        raise ValueError(f"num_rounds expected as int, got {type(num_rounds)} instead.")
    if num_rounds <= 0:
        raise ValueError("num_rounds needs to be a (strickly) positive integer.")
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

    if num_rounds == 1:
        first_dets = deepcopy(anc_detectors)
        if not gauge_detectors:
            stab_type = "x_type" if rot_basis else "z_type"
            stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)
            first_dets = set(anc_detectors).intersection(stab_qubits)

        experiment += qec_round_with_log_meas(
            model, layout, detectors, first_dets, rot_basis
        )
        return experiment

    for r in range(num_rounds - 1):
        if r == 0 and (not gauge_detectors):
            stab_type = "x_type" if rot_basis else "z_type"
            stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)
            first_dets = set(anc_detectors).intersection(stab_qubits)
            experiment += qec_round(model, layout, detectors, first_dets)
            continue

        experiment += qec_round(model, layout, detectors, anc_detectors)

    experiment += qec_round_with_log_meas(
        model, layout, detectors, anc_detectors, rot_basis
    )

    return experiment
