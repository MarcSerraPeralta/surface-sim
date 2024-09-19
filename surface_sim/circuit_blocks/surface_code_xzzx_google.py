"""
The circuits are based on the following paper by Google AI:
https://doi.org/10.1038/s41586-022-05434-1
https://doi.org/10.48550/arXiv.2207.06431 
"""

from itertools import chain
import warnings

from stim import Circuit

from ..layouts import Layout
from ..models import Model
from ..detectors import Detectors

# methods to have in this script
from .util import qubit_coords, log_x, log_z
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
    rot_basis: bool = False,
    meas_reset: bool = False,
) -> Circuit:
    """
    Returns stim circuit corresponding to a QEC cycle
    that includes the logical measurement
    of the given model.

    Params
    -------
    rot_basis
        By default, the logical measurement is in the Z basis.
        If rot_basis, the logical measurement is in the X basis.
    """
    anc_qubits = layout.get_qubits(role="anc")
    data_qubits = layout.get_qubits(role="data")
    qubits = set(data_qubits + anc_qubits)

    # a-h
    circuit = coherent_qec_part(model=model, layout=layout)

    # i (for logical measurement)
    stab_type = "x_type" if rot_basis else "z_type"
    stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)

    rot_qubits = set(anc_qubits)
    for direction in ("north_west", "south_east"):
        neighbors = layout.get_neighbors(stab_qubits, direction=direction)
        rot_qubits.update(neighbors)

    for instruction in model.hadamard(rot_qubits):
        circuit.append(instruction)

    idle_qubits = qubits - rot_qubits
    for instruction in model.idle(idle_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    # j (for logical measurement)
    # with detectors ordered as in the measurements
    for instruction in model.measure(anc_qubits):
        circuit.append(instruction)
    for instruction in model.measure(data_qubits):
        circuit.append(instruction)

    # detectors and logical observables
    stab_type = "x_type" if rot_basis else "z_type"
    stabs = layout.get_qubits(role="anc", stab_type=stab_type)
    detectors_stim = detectors.build_from_data(
        model.meas_target, layout.adjacency_matrix(), meas_reset, anc_qubits=stabs
    )
    circuit += detectors_stim

    log_op = "log_x" if rot_basis else "log_z"
    if log_op not in dir(layout):
        warnings.warn(
            "Deprecation warning: specify log_x and log_z in your layout.",
            DeprecationWarning,
        )
        targets = [model.meas_target(qubit, -1) for qubit in data_qubits]
        circuit.append("OBSERVABLE_INCLUDE", targets, 0)
    else:
        log_data_qubits = getattr(layout, log_op)
        targets = [model.meas_target(qubit, -1) for qubit in log_data_qubits]
        circuit.append("OBSERVABLE_INCLUDE", targets, 0)

    return circuit


def coherent_qec_part(model: Model, layout: Layout) -> Circuit:
    """
    Returns stim circuit corresponding to the steps "a" to "h" from the QEC cycle
    described in Google's paper for the given model.
    """
    data_qubits = layout.get_qubits(role="data")
    x_anc = layout.get_qubits(role="anc", stab_type="x_type")
    z_anc = layout.get_qubits(role="anc", stab_type="z_type")
    anc_qubits = x_anc + z_anc
    qubits = set(data_qubits + anc_qubits)

    circuit = Circuit()

    # a
    rot_qubits = set(anc_qubits)
    for instruction in model.hadamard(rot_qubits):
        circuit.append(instruction)

    x_qubits = set(data_qubits)
    for instruction in model.x_gate(x_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    # b
    int_pairs = layout.get_neighbors(anc_qubits, direction="north_east", as_pairs=True)
    int_qubits = list(chain.from_iterable(int_pairs))

    for instruction in model.cphase(int_qubits):
        circuit.append(instruction)

    idle_qubits = qubits - set(int_qubits)
    for instruction in model.idle(idle_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    # c
    rot_qubits = set(data_qubits)
    for instruction in model.hadamard(rot_qubits):
        circuit.append(instruction)

    x_qubits = set(anc_qubits)
    for instruction in model.x_gate(x_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    # d
    x_pairs = layout.get_neighbors(x_anc, direction="north_west", as_pairs=True)
    z_pairs = layout.get_neighbors(z_anc, direction="south_east", as_pairs=True)
    int_pairs = chain(x_pairs, z_pairs)
    int_qubits = list(chain.from_iterable(int_pairs))

    for instruction in model.cphase(int_qubits):
        circuit.append(instruction)

    idle_qubits = qubits - set(int_qubits)
    for instruction in model.idle(idle_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    # e
    x_qubits = qubits
    for instruction in model.x_gate(x_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    # f
    x_pairs = layout.get_neighbors(x_anc, direction="south_east", as_pairs=True)
    z_pairs = layout.get_neighbors(z_anc, direction="north_west", as_pairs=True)
    int_pairs = chain(x_pairs, z_pairs)
    int_qubits = list(chain.from_iterable(int_pairs))

    for instruction in model.cphase(int_qubits):
        circuit.append(instruction)

    idle_qubits = qubits - set(int_qubits)
    for instruction in model.idle(idle_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    # g
    rot_qubits = set(data_qubits)
    for instruction in model.hadamard(rot_qubits):
        circuit.append(instruction)

    x_qubits = set(anc_qubits)
    for instruction in model.x_gate(x_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    # h
    int_pairs = layout.get_neighbors(anc_qubits, direction="south_west", as_pairs=True)
    int_qubits = list(chain.from_iterable(int_pairs))

    for instruction in model.cphase(int_qubits):
        circuit.append(instruction)

    idle_qubits = qubits - set(int_qubits)
    for instruction in model.idle(idle_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    return circuit


def qec_round(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    meas_reset: bool = False,
) -> Circuit:
    """
    Returns stim circuit corresponding to a QEC cycle
    of the given model.
    """
    data_qubits = layout.get_qubits(role="data")
    anc_qubits = layout.get_qubits(role="anc")

    # a-h
    circuit = coherent_qec_part(model=model, layout=layout)

    # i
    rot_qubits = set(anc_qubits)
    for instruction in model.hadamard(rot_qubits):
        circuit.append(instruction)

    for instruction in model.x_gate(data_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    # j
    for instruction in model.measure(anc_qubits):
        circuit.append(instruction)

    for instruction in model.idle(data_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    if meas_reset:
        for instruction in model.reset(anc_qubits):
            circuit.append(instruction)

        for instruction in model.idle(data_qubits):
            circuit.append(instruction)

        circuit.append("TICK")

    # add detectors
    detectors_stim = detectors.build_from_anc(model.meas_target, meas_reset)
    circuit += detectors_stim

    return circuit
