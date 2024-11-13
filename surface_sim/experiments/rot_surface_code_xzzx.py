from stim import Circuit

from ..layouts import Layout
from ..circuit_blocks.rot_surface_code_xzzx import (
    init_qubits,
    log_meas,
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
        raise ValueError(f"num_rounds expected as int, got {type(num_rounds)} instead.")
    if num_rounds < 0:
        raise ValueError("num_rounds needs to be a positive integer.")
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
