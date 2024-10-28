from collections.abc import Callable, Iterable, Collection
from copy import deepcopy

import numpy as np
import xarray as xr
import galois
import stim


GF2 = galois.GF(2)


class Detectors:
    def __init__(
        self,
        anc_qubits: Collection[str],
        frame: str,
        anc_coords: dict[str, Collection[float | int]] | None = None,
    ) -> None:
        """Initalises the ``Detectors`` class.

        Parameters
        ----------
        anc_qubits
            List of ancilla qubits.
        frame
            Detector frame to use when building the detectors.
            The options for the detector frames are described in the Notes section.
        anc_coords
            Ancilla qubit coordinates that are added to the detectors if specified.
            The coordinates of the detectors will be ``(*ancilla_coords[i], r)``,
            with ``r`` the number of rounds (starting at 0).

        Notes
        -----
        Detector frame ``'1'`` builds the detectors in the basis given by the
        stabilizer generators of the first QEC round.

        Detector frame ``'r'`` builds the detectors in the basis given by the
        stabilizer generators of the last-measured QEC round.

        Detector frame ``'r-1'`` builds the detectors in the basis given by the
        stabilizer generators of the previous-last-measured QEC round.

        Detector frame ``'t'`` builds the detectors as ``m_{a,r} ^ m_{a,r-1}``
        independently of how the stabilizer generators have been transformed.
        """
        if not isinstance(anc_qubits, Collection):
            raise TypeError(
                f"'anc_qubits' must be iterable, but {type(anc_qubits)} was given."
            )
        if not isinstance(frame, str):
            raise TypeError(f"'frame' must be a str, but {type(frame)} was given.")
        if anc_coords is None:
            anc_coords = {a: [] for a in anc_qubits}
        if not isinstance(anc_coords, dict):
            raise TypeError(
                f"'anc_coords' must be a dict, but {type(anc_coords)} was given."
            )
        if not (set(anc_coords) == set(anc_qubits)):
            raise ValueError("'anc_coords' must have 'anc_qubits' as its keys.")
        if any(not isinstance(c, Collection) for c in anc_coords.values()):
            raise TypeError("Values in 'anc_coords' must be a collection.")
        if len(set(len(c) for c in anc_coords.values())) != 1:
            raise ValueError("Values in 'anc_coords' must have the same lenght.")

        self.anc_qubits = anc_qubits
        self.frame = frame
        self.anc_coords = anc_coords

        self.new_circuit()

        return

    def new_circuit(self):
        """Resets all the current generators and number of rounds in order
        to create a different circuit.
        """
        generators = xr.DataArray(
            data=np.identity(len(self.anc_qubits), dtype=np.int64),
            coords=dict(
                stab_gen=self.anc_qubits,
                basis=range(len(self.anc_qubits)),
            ),
        )

        self.prev_gen = deepcopy(generators)
        self.curr_gen = deepcopy(generators)
        self.init_gen = deepcopy(generators)
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
        anc_reset: bool,
        anc_qubits: Iterable[str] | None = None,
    ) -> stim.Circuit:
        """Returns the stim circuit with the corresponding detectors
        given that the ancilla qubits have been measured.

        Parameters
        ----------
        get_rec
            Function that given ``qubit_label, rel_meas_id`` returns the
            corresponding ``stim.target_rec``. The intention is to give the
            ``Model.meas_target`` method.
        anc_reset
            Flag for if the ancillas are being reset in every QEC cycle.
        anc_qubits
            List of the ancilla qubits for which to build the detectors.
            By default, builds all the detectors.

        Returns
        -------
        detectors_stim
            Detectors defined in a ``stim`` circuit.
        """
        if not (isinstance(anc_qubits, Iterable) or (anc_qubits is None)):
            raise TypeError(
                f"'anc_qubits' must be iterable or None, but {type(anc_qubits)} was given."
            )
        if not isinstance(get_rec, Callable):
            raise TypeError(
                f"'get_rec' must be callable, but {type(get_rec)} was given."
            )

        if self.frame == "1":
            basis = self.init_gen
        elif self.frame == "r":
            basis = self.curr_gen
        elif self.frame == "r-1":
            basis = self.prev_gen
        elif self.frame == "t":
            anc_detector_labels = self.init_gen.stab_gen.values.tolist()
        else:
            raise ValueError(
                f"'frame' must be '1', 'r-1', 'r' or 't', but {self.frame} was given."
            )

        if anc_qubits is None:
            anc_qubits = self.curr_gen.stab_gen.values.tolist()

        self.num_rounds += 1

        if self.frame != "t":
            detectors = _get_ancilla_meas_for_detectors(
                self.curr_gen,
                self.prev_gen,
                basis=basis,
                num_rounds=self.num_rounds,
                anc_reset_curr=anc_reset,
                anc_reset_prev=anc_reset,
            )
        else:
            meas_comp = -2 if anc_reset else -3
            detectors = {}
            for anc in anc_detector_labels:
                dets = [(anc, -1)]
                if meas_comp + self.num_rounds >= 0:
                    dets.append((anc, meas_comp))
                detectors[anc] = dets

        # build the stim circuit
        detectors_stim = stim.Circuit()
        for anc, targets in detectors.items():
            if anc in anc_qubits:
                detectors_rec = [get_rec(*t) for t in targets]
            else:
                # create the detector but make it be always 0
                detectors_rec = []
            coords = [*self.anc_coords[anc], self.num_rounds - 1]
            instr = stim.CircuitInstruction(
                "DETECTOR", gate_args=coords, targets=detectors_rec
            )
            detectors_stim.append(instr)

        # update generators
        self.prev_gen = deepcopy(self.curr_gen)

        return detectors_stim

    def build_from_data(
        self,
        get_rec: Callable,
        adjacency_matrix: xr.DataArray,
        anc_reset: bool,
        reconstructable_stabs: Iterable[str],
        anc_qubits: Iterable[str] | None = None,
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
        anc_reset
            Flag for if the ancillas are being reset in every QEC cycle.
        reconstructable_stabs
            Stabilizers that can be reconstructed from the data qubit outcomes.
        anc_qubits
            List of the ancilla qubits for which to build the detectors.
            By default, builds all the detectors.

        Returns
        -------
        detectors_stim
            Detectors defined in a ``stim`` circuit.
        """
        if not isinstance(reconstructable_stabs, Iterable):
            raise TypeError(
                "'reconstructable_stabs' must be iterable, "
                f"but {type(reconstructable_stabs)} was given."
            )
        if not (isinstance(anc_qubits, Iterable) or (anc_qubits is None)):
            raise TypeError(
                f"'anc_qubits' must be iterable or None, but {type(anc_qubits)} was given."
            )
        if not isinstance(get_rec, Callable):
            raise TypeError(
                f"'get_rec' must be callable, but {type(get_rec)} was given."
            )

        if self.frame == "1":
            basis = self.init_gen
        elif self.frame == "r":
            basis = self.curr_gen
        elif self.frame == "r-1":
            basis = self.prev_gen
        elif self.frame == "t":
            anc_detector_labels = self.init_gen.stab_gen.values.tolist()
        else:
            raise ValueError(
                f"'frame' must be '1', 'r-1', 'r' or 't', but {self.frame} was given."
            )

        if anc_qubits is None:
            anc_qubits = self.curr_gen.stab_gen.values.tolist()

        self.num_rounds += 1

        if self.frame != "t":
            anc_detectors = _get_ancilla_meas_for_detectors(
                self.curr_gen,
                self.prev_gen,
                basis=basis,
                num_rounds=self.num_rounds,
                anc_reset_curr=True,
                anc_reset_prev=anc_reset,
            )
        else:
            anc_detectors = {}
            for anc in anc_detector_labels:
                dets = [(anc, -1)]
                if self.num_rounds > 1:
                    dets.append((anc, -2))
                if (not anc_reset) and (self.num_rounds > 2):
                    dets.append((anc, -3))
                anc_detectors[anc] = dets

        anc_detectors = {
            anc: d for anc, d in anc_detectors.items() if anc in reconstructable_stabs
        }

        # udpate the (anc, -1) to a the corresponding set of (data, -1)
        detectors = {}
        for anc_qubit, dets in anc_detectors.items():
            new_dets = []
            for det in dets:
                if det[1] != -1:
                    # rel_meas need to be updated because the ancillas have not
                    # been measured in the last round, only the data qubits
                    # e.g. ("X1", -2) should be ("X1", -1)
                    det = (det[0], det[1] + 1)
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
        for anc, targets in detectors.items():
            if anc in anc_qubits:
                detectors_rec = [get_rec(*t) for t in targets]
            else:
                # create the detector but make it be always 0
                detectors_rec = []
            coords = [*self.anc_coords[anc], self.num_rounds - 1]
            instr = stim.CircuitInstruction(
                "DETECTOR", gate_args=coords, targets=detectors_rec
            )
            detectors_stim.append(instr)

        # update generators
        self.prev_gen = deepcopy(self.curr_gen)

        return detectors_stim


def _get_ancilla_meas_for_detectors(
    curr_gen: xr.DataArray,
    prev_gen: xr.DataArray,
    basis: xr.DataArray,
    num_rounds: int,
    anc_reset_curr: bool,
    anc_reset_prev: bool,
) -> dict[str, list[tuple[str, int]]]:
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
    anc_reset_curr
        Flag for if the ancillas are being reset in the currently
        measured QEC cycle.
    anc_reset_prev
        Flag for if the ancillas are being reset in the second-last QEC cycle,
        corresponding to the previus cycle to the currently measured one.

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

        if not anc_reset_curr and num_rounds >= 2:
            targets += [(anc_qubits[ind], -2) for ind in c_gen_inds]
        if not anc_reset_prev and num_rounds >= 3:
            targets += [(anc_qubits[ind], -3) for ind in p_gen_inds]

        detectors[anc_qubit] = targets

    return detectors
