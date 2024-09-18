from typing import List, Callable, Dict, Tuple, Optional
from copy import deepcopy

import numpy as np
import xarray as xr
import galois
import stim


GF2 = galois.GF(2)


class Detectors:
    def __init__(self, anc_qubits: List[str], frame: str):
        generators = xr.DataArray(
            data=np.identity(len(anc_qubits), dtype=np.int64),
            coords=dict(
                stab_gen=anc_qubits,
                basis=range(len(anc_qubits)),
            ),
        )

        self.prev_gen = deepcopy(generators)
        self.curr_gen = deepcopy(generators)
        self.init_gen = deepcopy(generators)
        self.frame = frame
        self.num_rounds = 0

        return

    def update(self, unitary_mat: xr.DataArray):
        """Update the current stabilizer generators with the unitary matrix
        descriving the effect of the logical gate.

        Parameters
        ----------
        unitary_mat
            Unitary matrix descriving the change of the stabilizers
            generators (mod 2). It must have coordinates 'stab_gen' and
            'new_stab_gen' whose values correspond to the ancilla qubit labels.
            An entry ``(stab_gen="X1", new_stab_gen="Z1")`` being 1, indicates
            that the new stabilizer generator that would be measured in ancilla
            qubit ``"Z1"`` by a QEC cycle is a product of at least the
            stabilizer generator that would be measured in ancilla qubit
            ``"X1"`` by a QEC cycle (before the logical gate).

        Notes
        -----
        The ``unitary_mat`` matrix can be computed by calculating

        .. math::

            S'_i = U_L^\\dagger S_i U_L

        with :math:`U_L` the logical gate and :math:`S_i` (:math:`S'_i`) the
        stabilizer generator :math:`i` before (after) the logical gate.
        From `this reference <https://arthurpesah.me/blog/2023-03-16-stabilizer-formalism-2/>`_.
        """
        if not isinstance(unitary_mat, xr.DataArray):
            raise TypeError(
                "'unitary_mat' must be an xr.DataArray, "
                f"but {type(unitary_mat)} was given."
            )
        if set(unitary_mat.coords.dims) != set(["stab_gen", "new_stab_gen"]):
            raise ValueError(
                "The coordinates of 'unitary_mat' must be 'stab_gen' and 'new_stab_gen', "
                f"but {unitary_mat.coords.dims} were given."
            )
        if not (
            set(unitary_mat.stab_gen.values)
            == set(unitary_mat.new_stab_gen.values)
            == set(self.init_gen.stab_gen.values)
        ):
            raise ValueError(
                "The coordinate values of 'unitary_mat' must match "
                "the ones from 'self.init_gen'"
            )

        # check that the matrix is invertible (mod 2)
        matrix = GF2(unitary_mat.to_numpy())
        if np.linalg.det(matrix) == 0:
            raise ValueError("'unitary_mat' is not invertible.")

        self.curr_gen = (unitary_mat @ self.curr_gen) % 2
        self.curr_gen = self.curr_gen.rename({"new_stab_gen": "stab_gen"})

        return

    def build_from_anc(
        self,
        get_rec: Callable,
        meas_reset: bool,
        anc_qubits: Optional[List[str]] = None,
    ) -> stim.Circuit:
        """Returns the stim circuit with the corresponding detectors
        given that the ancilla qubits have been measured.

        Parameters
        ----------
        get_rec
            Function that given ``qubit_label, rel_meas_id`` returns the
            ``target_rec`` integer. The intention is to give the
            ``Model.meas_target`` method.
        meas_reset
            Flag for if the ancillas are being reset after being measured.
        anc_qubits
            List of the ancilla qubits for which to build the detectors.
            By default, builds all the detectors.

        Returns
        -------
        detectors_stim
            Detectors defined in a ``stim`` circuit.
        """
        if self.frame == "1":
            basis = self.init_gen
        elif self.frame == "r":
            basis = self.curr_gen
        else:
            raise ValueError(f"'frame' must be '1' or 'r', but {self.frame} was given.")

        self.num_rounds += 1

        detectors = _get_ancilla_meas_for_detectors(
            self.curr_gen,
            self.prev_gen,
            basis=basis,
            num_rounds=self.num_rounds,
            meas_reset=meas_reset,
        )
        if anc_qubits is not None:
            detectors = {anc: d for anc, d in detectors.items() if anc in anc_qubits}

        # build the stim circuit
        detectors_stim = stim.Circuit()
        for targets in detectors.values():
            detectors_rec = [get_rec(*t) for t in targets]
            detectors_stim.append("DETECTOR", detectors_rec, [])

        # update generators
        self.previous_gen = deepcopy(self.curr_gen)

        return detectors_stim

    def build_from_data(
        self,
        get_rec: Callable,
        adjacency_matrix: xr.DataArray,
        meas_reset: bool,
        anc_qubits: Optional[List[str]] = None,
    ) -> stim.Circuit:
        """Returns the stim circuit with the corresponding detectors
        given that the data qubits have been measured.

        Parameters
        ----------
        get_rec
            Function that given ``qubit_label, rel_meas_id`` returns the
            ``target_rec`` integer. The intention is to give the
            ``Model.meas_target`` method.
        adjacency_matrix
            Matrix descriving the data qubit support on the stabilizers.
            Its coordinates are ``from_qubit`` and ``to_qubit``.
            See ``qec_util.Layout.adjacency_matrix`` for more information.
        meas_reset
            Flag for if the ancillas are being reset after being measured.
        anc_qubits
            List of the ancilla qubits for which to build the detectors.
            By default, builds all the detectors.

        Returns
        -------
        detectors_stim
            Detectors defined in a ``stim`` circuit.
        """
        if self.frame == "1":
            basis = self.init_gen
        elif self.frame == "r":
            basis = self.curr_gen
        else:
            raise ValueError(f"'frame' must be '1' or 'r', but {self.frame} was given.")

        self.num_rounds += 1

        anc_detectors = _get_ancilla_meas_for_detectors(
            self.curr_gen,
            self.prev_gen,
            basis=basis,
            num_rounds=self.num_rounds,
            meas_reset=meas_reset,
        )
        if anc_qubits is not None:
            anc_detectors = {
                anc: d for anc, d in anc_detectors.items() if anc in anc_qubits
            }

        # udpate the (anc, -1) to a the corresponding set of (data, -1)
        detectors = {}
        for anc_qubit, dets in anc_detectors.items():
            new_dets = []
            for det in dets:
                if det[1] != -1:
                    new_dets.append(det)
                    continue

                support = adjacency_matrix.sel(from_qubit=det[0])
                data_qubits = [
                    q for q, sup in zip(support.to_qubit.values, support) if sup
                ]
                new_dets += [(q, -1) for q in data_qubits]
            detectors[anc_qubit] = new_dets

        # build the stim circuit
        detectors_stim = stim.Circuit()
        for targets in detectors.values():
            detectors_rec = [get_rec(*t) for t in targets]
            detectors_stim.append("DETECTOR", detectors_rec, [])

        # update generators
        self.previous_gen = deepcopy(self.curr_gen)

        return detectors_stim


def _get_ancilla_meas_for_detectors(
    curr_gen: xr.DataArray,
    prev_gen: xr.DataArray,
    basis: xr.DataArray,
    num_rounds: int,
    meas_reset: bool,
) -> Dict[str, List[Tuple[str, int]]]:
    """Returns the ancilla measurements as ``(anc_qubit, rel_meas_ind)``
    required to build the detectors in the given frame.

    Parameters
    ----------
    curr_gen
        Current stabilizer generators.
    prev_gen
        Stabilizer generators measured in the previous round.
        If no stabilizers have been measured, it is ``None``.
    basis
        Basis in which to represent the detectors.
    num_rounds
        Number of QEC cycles performed (including the current one).
    meas_reset
        Flag for if the ancillas are being reset after being measured.

    Returns
    -------
    detectors
        Dictionary of the ancilla qubits and their corresponding detectors
        expressed as a list of ``(anc_qubit, -meas_rel_id)``.
    """
    # matrix inversion is not possible in xarray,
    # thus go to np.ndarrays with correct order of columns and rows.
    anc_qubits = curr_gen.stab_gen.values
    curr_gen_arr = curr_gen.sel(stab_gen=anc_qubits).values
    basis_arr = basis.sel(stab_gen=anc_qubits).values
    prev_gen_arr = prev_gen.sel(stab_gen=anc_qubits).values

    # convert self.prev_gen and self.curr_gen to the frame basis
    curr_gen_arr = curr_gen_arr @ np.linalg.inv(basis_arr)
    prev_gen_arr = prev_gen_arr @ np.linalg.inv(basis_arr)

    # get all outcomes that need to be XORed
    detectors = {}
    for anc_qubit, c_gen, p_gen in zip(anc_qubits, curr_gen_arr, prev_gen_arr):
        c_gen_inds = np.where(c_gen)[0]
        p_gen_inds = np.where(p_gen)[0]

        targets = [(anc_qubits[ind], -1) for ind in c_gen_inds]
        if num_rounds >= 2:
            targets += [(anc_qubits[ind], -2) for ind in p_gen_inds]

        if not meas_reset:
            if num_rounds >= 2:
                targets += [(anc_qubits[ind], -2) for ind in c_gen_inds]
            if num_rounds >= 3:
                targets += [(anc_qubits[ind], -3) for ind in p_gen_inds]

        detectors[anc_qubit] = targets

    return detectors
