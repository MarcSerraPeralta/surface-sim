import stim
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF


def get_circuit_distance(circuit: stim.Circuit) -> int:
    """Returns the circuit distance of the given circuit.

    Note that the SAT solver can take some time.

    Parameters
    ----------
    circuit
        Stim circuit.

    Returns
    -------
    d_circ
        Circuit distance of the given circuit.
    """
    if not isinstance(circuit, stim.Circuit):
        raise ValueError(
            "'circuit' must be a 'stim.Circuit', " f"but {type(circuit)} was given."
        )

    # remove gauge detectors from experiment (if not, it doesn't work)
    dem = circuit.detector_error_model(allow_gauge_detectors=True)
    gauge_dets = []
    for line in dem.flattened():
        if line.type == "error" and line.args_copy()[0] == 0.5:
            gauge_dets += line.targets_copy()
    gauge_dets = [d.val for d in gauge_dets]

    new_circuit = stim.Circuit()
    det_counter = -1
    for line in circuit.flattened():
        if line.name == "DETECTOR":
            det_counter += 1
            if det_counter in gauge_dets:
                continue

        new_circuit.append(line)

    # solve SAT problem
    wcnf_string = new_circuit.shortest_error_sat_problem()
    wcnf = WCNF(from_string=wcnf_string)
    with RC2(wcnf) as rc2:
        rc2.compute()
        d_circ = rc2.cost

    return d_circ
