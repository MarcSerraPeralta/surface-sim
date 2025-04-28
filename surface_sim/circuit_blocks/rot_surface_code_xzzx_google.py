"""
The circuits are based on the following paper by Google AI:
https://doi.org/10.1038/s41586-022-05434-1
https://doi.org/10.48550/arXiv.2207.06431
"""

from itertools import chain

from stim import Circuit

from ..layouts.layout import Layout
from ..models import Model
from ..detectors import Detectors

# methods to have in this script
from .util import qubit_coords
from .util import log_x_xzzx as log_x
from .util import log_z_xzzx as log_z
from .util import init_qubits_xzzx as init_qubits

__all__ = [
    "qubit_coords",
    "qec_round_with_log_meas",
    "log_x",
    "log_z",
    "qec_round",
    "init_qubits",
]


def qec_round_with_log_meas(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    anc_detectors: list[str] | None = None,
    rot_basis: bool = False,
) -> Circuit:
    """
    Returns stim circuit corresponding to a QEC cycle
    that includes the logical measurement
    of the given model. It defines the observables for
    all logical qubits in the layout.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout.
    detectors
        Detector definitions to use.
    anc_detectors
        List of ancilla qubits for which to define the detectors.
        If ``None``, adds all detectors.
        By default ``None``.
    rot_basis
        If ``True``, the memory experiment is performed in the X basis.
        If ``False``, the memory experiment is performed in the Z basis.
        By deafult ``False``.
    """
    if layout.code != "rotated_surface_code":
        raise TypeError(
            "The given layout is not a rotated surface code, " f"but a {layout.code}"
        )
    if anc_detectors is None:
        anc_detectors = layout.anc_qubits
    if set(anc_detectors) > set(layout.anc_qubits):
        raise ValueError("Some of the given 'anc_qubits' are not ancilla qubits.")

    anc_qubits = layout.anc_qubits
    data_qubits = layout.data_qubits
    qubits = set(layout.qubits)

    # a-h
    circuit = coherent_qec_part(model=model, layout=layout)

    # i (for logical measurement)
    stab_type = "x_type" if rot_basis else "z_type"
    stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)

    rot_qubits = set(anc_qubits)
    for direction in ("north_west", "south_east"):
        neighbors = layout.get_neighbors(stab_qubits, direction=direction)
        rot_qubits.update(neighbors)

    circuit += model.hadamard(rot_qubits)

    idle_qubits = qubits - rot_qubits
    circuit += model.idle(idle_qubits)
    circuit += model.tick()

    # j (for logical measurement)
    circuit += model.measure(anc_qubits)
    circuit += model.measure(data_qubits)

    # detectors and logical observables
    detectors_stim = detectors.build_from_anc(
        model.meas_target, anc_reset=True, anc_qubits=anc_detectors
    )
    circuit += detectors_stim

    stab_type = "x_type" if rot_basis else "z_type"
    stabs = layout.get_qubits(role="anc", stab_type=stab_type)
    anc_support = layout.get_support(stabs)
    detectors_stim = detectors.build_from_data(
        model.meas_target,
        anc_support,
        anc_reset=True,
        reconstructable_stabs=stabs,
        anc_qubits=anc_detectors,
    )
    circuit += detectors_stim

    log_op = "log_x" if rot_basis else "log_z"
    for logical_qubit in layout.logical_qubits:
        log_data_qubits = layout.logical_param(log_op, logical_qubit)
        targets = [model.meas_target(qubit, -1) for qubit in log_data_qubits]
        circuit.append("OBSERVABLE_INCLUDE", targets, 0)

    detectors.deactivate_detectors(layout.anc_qubits)

    return circuit


def coherent_qec_part(model: Model, layout: Layout) -> Circuit:
    """
    Returns stim circuit corresponding to the steps "a" to "h" from the QEC cycle
    described in Google's paper for the given model.
    """
    data_qubits = layout.data_qubits
    x_anc = layout.get_qubits(role="anc", stab_type="x_type")
    z_anc = layout.get_qubits(role="anc", stab_type="z_type")
    anc_qubits = x_anc + z_anc
    qubits = set(layout.qubits)

    circuit = Circuit()

    circuit += model.incoming_noise(data_qubits)
    circuit += model.tick()

    # a
    circuit += model.hadamard(anc_qubits)
    circuit += model.x_gate(data_qubits)
    circuit += model.tick()

    # b
    int_pairs = layout.get_neighbors(anc_qubits, direction="north_east", as_pairs=True)
    int_qubits = list(chain.from_iterable(int_pairs))

    circuit += model.cphase(int_qubits)

    idle_qubits = qubits - set(int_qubits)
    circuit += model.idle(idle_qubits)
    circuit += model.tick()

    # c
    circuit += model.hadamard(data_qubits)
    circuit += model.x_gate(anc_qubits)
    circuit += model.tick()

    # d
    x_pairs = layout.get_neighbors(x_anc, direction="north_west", as_pairs=True)
    z_pairs = layout.get_neighbors(z_anc, direction="south_east", as_pairs=True)
    int_pairs = chain(x_pairs, z_pairs)
    int_qubits = list(chain.from_iterable(int_pairs))

    circuit += model.cphase(int_qubits)

    idle_qubits = qubits - set(int_qubits)
    circuit += model.idle(idle_qubits)
    circuit += model.tick()

    # e
    circuit += model.x_gate(qubits)
    circuit += model.tick()

    # f
    x_pairs = layout.get_neighbors(x_anc, direction="south_east", as_pairs=True)
    z_pairs = layout.get_neighbors(z_anc, direction="north_west", as_pairs=True)
    int_pairs = chain(x_pairs, z_pairs)
    int_qubits = list(chain.from_iterable(int_pairs))

    circuit += model.cphase(int_qubits)

    idle_qubits = qubits - set(int_qubits)
    circuit += model.idle(idle_qubits)
    circuit += model.tick()

    # g
    circuit += model.hadamard(data_qubits)
    circuit += model.x_gate(anc_qubits)
    circuit += model.tick()

    # h
    int_pairs = layout.get_neighbors(anc_qubits, direction="south_west", as_pairs=True)
    int_qubits = list(chain.from_iterable(int_pairs))

    circuit += model.cphase(int_qubits)

    idle_qubits = qubits - set(int_qubits)
    circuit += model.idle(idle_qubits)
    circuit += model.tick()

    return circuit


def qec_round(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    anc_detectors: list[str] | None = None,
) -> Circuit:
    """
    Returns stim circuit corresponding to a QEC cycle
    of the given model.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout.
    detectors
        Detector definitions to use.
    anc_detectors
        List of ancilla qubits for which to define the detectors.
        If ``None``, adds all detectors.
        By default ``None``.
    """
    if layout.code != "rotated_surface_code":
        raise TypeError(
            "The given layout is not a rotated surface code, " f"but a {layout.code}"
        )

    data_qubits = layout.data_qubits
    anc_qubits = layout.anc_qubits

    # a-h
    circuit = coherent_qec_part(model=model, layout=layout)

    # i
    circuit += model.hadamard(anc_qubits)
    circuit += model.x_gate(data_qubits)
    circuit += model.tick()

    # j
    circuit += model.measure(anc_qubits)

    circuit += model.idle_meas(data_qubits)
    circuit += model.tick()

    circuit += model.reset(anc_qubits)
    circuit += model.idle_reset(data_qubits)
    circuit += model.tick()

    # add detectors
    detectors_stim = detectors.build_from_anc(
        model.meas_target, anc_reset=True, anc_qubits=anc_detectors
    )
    circuit += detectors_stim

    return circuit
