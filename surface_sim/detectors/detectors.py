from typing import List, Callable
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
        if unitary_mat.coords.dims != ["stab_gen", "new_stab_gen"]:
            raise ValueError(
                "The coordinates of 'unitary_mat' must be 'stab_gen' and 'new_stab_gen', "
                f"but {unitary_mat.coords} were given."
            )
        if not (
            unitary_mat.stab_gen.values()
            == unitary_mat.new_stab_gen.values()
            == self.init_gen.stab_gen.values()
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

        return

    def build(self, get_rec: Callable) -> stim.Circuit:
        """Returns the stim circuit with the corresponding detectors.

        Parameters
        ----------
        get_rec
            Function that given ``(qubit_label, rel_meas_id)`` returns the
            ``target_rec`` integer. The intention is to give the
            ``Model.meas_target`` method.
        """
        if self.frame == "0":
            basis = self.init_gen
        elif self.frame == "r":
            basis = self.curr_gen
        else:
            raise ValueError(f"'frame' must be '0' or '1', but {self.frame} was given.")

        # convert self.prev_gen and self.curr_gen to the frame basis

        # get all outcomes that need to be XORed

        # build the stim circuit
        detectors_stim = stim.Circuit()

        # update generators
        self.previous_gen = deepcopy(self.curr_gen)

        return detectors_stim
