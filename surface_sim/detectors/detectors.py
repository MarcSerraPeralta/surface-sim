from collections.abc import Callable, Iterable, Sequence
from copy import deepcopy

import numpy as np
import xarray as xr
import galois
import stim

from ..layouts import Layout


GF2 = galois.GF(2)


class Detectors:
    def __init__(
        self,
        anc_qubits: Sequence[str],
        frame: str,
        anc_coords: dict[str, Sequence[float | int]] | None = None,
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
        if not isinstance(anc_qubits, Sequence):
            raise TypeError(
                f"'anc_qubits' must be a Sequence, but {type(anc_qubits)} was given."
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
        if any(not isinstance(c, Sequence) for c in anc_coords.values()):
            raise TypeError("Values in 'anc_coords' must be a collection.")
        if len(set(len(c) for c in anc_coords.values())) != 1:
            raise ValueError("Values in 'anc_coords' must have the same lenght.")

        self.anc_qubit_labels = anc_qubits
        self.frame = frame
        self.anc_coords = anc_coords

        self.new_circuit()

        return

    def new_circuit(self):
        """Resets all the current generators and number of rounds in order
        to create a different circuit.
        """
        generators = xr.DataArray(
            data=np.identity(len(self.anc_qubit_labels), dtype=np.int64),
            coords=dict(
                stab_gen=self.anc_qubit_labels,
                basis=range(len(self.anc_qubit_labels)),
            ),
        )

        self.prev_gen = deepcopy(generators)
        self.curr_gen = deepcopy(generators)
        self.init_gen = deepcopy(generators)
        self.anc_qubits = {a: False for a in self.anc_qubit_labels}
        self.num_rounds = {a: 0 for a in self.anc_qubit_labels}
        self.total_num_rounds = 0

        return

    def activate_detectors(self, anc_qubits: Iterable[str]):
        """Activates the given ancilla detectors."""
        if not isinstance(anc_qubits, Iterable):
            raise TypeError(
                f"'anc_qubits' must be an Iterable, but {type(anc_qubits)} was given."
            )
        if set(anc_qubits) > set(self.anc_qubits):
            raise ValueError(
                "Elements in 'anc_qubits' are not ancilla qubits in this object."
            )

        for anc in anc_qubits:
            if self.anc_qubits[anc]:
                raise ValueError(f"Ancilla {anc} was already active.")

            self.anc_qubits[anc] = True

        return

    def deactivate_detectors(self, anc_qubits: Iterable[str]):
        """Deactivates the given ancilla detectors."""
        if not isinstance(anc_qubits, Iterable):
            raise TypeError(
                f"'anc_qubits' must be an Iterable, but {type(anc_qubits)} was given."
            )
        if set(anc_qubits) > set(self.anc_qubits):
            raise ValueError(
                "Elements in 'anc_qubits' are not ancilla qubits in this object."
            )

        for anc in anc_qubits:
            if not self.anc_qubits[anc]:
                raise ValueError(f"Ancilla {anc} was already active.")

            self.anc_qubits[anc] = False

        return

    def update_from_dict(self, new_stab_gens: dict[str, list[str]]) -> None:
        """Update the current stabilizer generators with the dictionary
        descriving the effect of the logical gate.

        See module ``surface_sim.log_gates`` to see how to prepare
        the layout to run logical gates.

        Parameters
        ---------
        new_stab_gens
            Dictionary that maps ancilla qubits (representing the new stabilizer
            generators) to a list of ancilla qubits (representing the old
            stabilizer generators).
            If the dictionary is missing ancillas, their stabilizer generators
            are assumed to not be transformed by the logical gate.
            See ``get_new_stab_dict_from_layout`` for more information.
            For example, ``{"X1": ["X1", "Z1"]}`` is interpreted as that the
            logical gate has transformed X1 to X1*Z1.
        """
        if not isinstance(new_stab_gens, dict):
            raise TypeError(
                "'new_stab_gens' must be a dict, "
                f"but {type(new_stab_gens)} was given."
            )
        if any(not isinstance(s, Iterable) for s in new_stab_gens.values()):
            raise TypeError("Elements in 'new_stab_gens' must be lists.")
        if set(new_stab_gens) > set(self.anc_qubit_labels):
            raise ValueError(
                "Elements in 'new_stab_gens' are not ancilla qubits in this Detectors class."
            )

        unitary_mat = xr.DataArray(
            data=np.identity(len(self.anc_qubit_labels)),
            coords=dict(
                new_stab_gen=self.anc_qubit_labels, stab_gen=self.anc_qubit_labels
            ),
        )

        for new_stab, support_old_stabs in new_stab_gens.items():
            # remove '1' entry due to np.identity
            unitary_mat.loc[dict(stab_gen=new_stab, new_stab_gen=new_stab)] = 0
            for old_stab in support_old_stabs:
                unitary_mat.loc[dict(new_stab_gen=new_stab, stab_gen=old_stab)] = 1

        # galois requires that the arrays are integers, not floats.
        unitary_mat = unitary_mat.astype(int)

        self.update(unitary_mat)

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
                f"'anc_qubits' must be an Iterable or None, but {type(anc_qubits)} was given."
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
        anc_qubits = [a for a in anc_qubits if self.anc_qubits[a]]

        self.total_num_rounds += 1
        self.num_rounds = {
            anc: self.num_rounds[anc] + 1 if active else self.num_rounds[anc]
            for anc, active in self.anc_qubits.items()
        }

        # generate detectors for all ancillas, then remove the ones
        # that are not needed
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
                if meas_comp + self.num_rounds[anc] >= 0:
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
            coords = [*self.anc_coords[anc], self.total_num_rounds - 1]
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
        anc_support: dict[str, Sequence[str]],
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
        anc_support
            Dictionary descriving the data qubit support on the stabilizers.
            The keys are the ancilla qubits and the values are the collection
            of data qubits.
            See ``surface_sim.Layout.adjacency_matrix`` and
            ``surface_sim.detectors.get_support_from_adj_matrix`` for more information.
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
        if not isinstance(anc_support, dict):
            raise TypeError(
                f"'anc_support' must be a dict, but {type(anc_support)} was given."
            )
        if set(anc_support) < set(reconstructable_stabs):
            raise ValueError(
                "Elements in 'reconstructable_stabs' must be present in 'anc_support'."
            )

        # for a logical measurement, one always needs to build the Z- or X-type
        # detectors (depending on the logical measurement basis). One should not
        # try to build any other type of detectors as it is not possible (because
        # we have only measured the data qubits in an specific basis), so we
        # do not have access to all stabilizers.
        if self.frame == "1":
            basis = self.curr_gen
        elif self.frame == "r":
            basis = self.curr_gen
        elif self.frame == "r-1":
            basis = self.curr_gen
        elif self.frame == "t":
            anc_detector_labels = self.init_gen.stab_gen.values.tolist()
        else:
            raise ValueError(
                f"'frame' must be '1', 'r-1', 'r' or 't', but {self.frame} was given."
            )

        if anc_qubits is None:
            anc_qubits = self.curr_gen.stab_gen.values.tolist()
        anc_qubits = [a for a in anc_qubits if self.anc_qubits[a]]

        # Logical measurement is not considered a QEC cycle but a logical operation.
        # therefore, it does not increase the number of rounds.
        # However, the way building the detectors is implemented relies on
        # faking that ancilla qubits have been measured instead of the data qubits.
        fake_num_rounds = {
            anc: self.num_rounds[anc] + 1 if active else self.num_rounds[anc]
            for anc, active in self.anc_qubits.items()
        }

        # generate detectors for all ancillas, then remove the ones
        # that are not needed.
        if self.frame != "t":
            anc_detectors = _get_ancilla_meas_for_detectors(
                self.curr_gen,
                self.prev_gen,
                basis=basis,
                num_rounds=fake_num_rounds,
                anc_reset_curr=True,
                anc_reset_prev=anc_reset,
            )
        else:
            anc_detectors = {}
            for anc in anc_detector_labels:
                dets = [(anc, -1)]
                if fake_num_rounds[anc] > 1:
                    dets.append((anc, -2))
                if (not anc_reset) and (fake_num_rounds[anc] > 2):
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

                new_dets += [(q, -1) for q in anc_support[det[0]]]

            detectors[anc_qubit] = new_dets

        # Build the stim circuit.
        # the coordinates of the detectors from the logical measurement are
        # set to half a timestep unit (instead of a full unit as in the QEC cycle),
        # because they are considered logical operations, not QEC cycles.
        detectors_stim = stim.Circuit()
        for anc, targets in detectors.items():
            if anc in anc_qubits:
                detectors_rec = [get_rec(*t) for t in targets]
            else:
                # create the detector but make it be always 0
                detectors_rec = []
            coords = [*self.anc_coords[anc], fake_num_rounds[anc] - 1.5]
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
    num_rounds: dict[str, int],
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
    anc_qubits = curr_gen.stab_gen.values.tolist()
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
        if num_rounds[anc_qubit] >= 2:
            targets += [(anc_qubits[ind], -2) for ind in p_gen_inds]

        if not anc_reset_curr and num_rounds[anc_qubit] >= 2:
            targets += [(anc_qubits[ind], -2) for ind in c_gen_inds]
        if not anc_reset_prev and num_rounds[anc_qubit] >= 3:
            targets += [(anc_qubits[ind], -3) for ind in p_gen_inds]

        detectors[anc_qubit] = targets

    return detectors


def get_new_stab_dict_from_layout(
    layout: Layout, log_gate: str
) -> dict[str, list[str]]:
    """Returns a dictionary that describes the stabilizer generator transformation
    due to the given logical gate.

    For example, the output ``{"X1": ["X1", "Z1"]}`` is interpreted as that the
    logical gate has transformed X1 to X1*Z1.

    Parameters
    ----------
    layout
        Layout that has information about the ``log_gate``.
    log_gate
        Name of the logical gate.

    Returns
    -------
    new_stab_gens
        Dictionary that maps ancilla qubits (representing the new stabilizer
        generators) to a list of ancilla qubits (representing the old
        stabilizer generators).
        If the dictionary is missing ancillas, their stabilizer generators
        are assumed to not be transformed by the logical gate.
    """
    if not isinstance(layout, Layout):
        raise TypeError(f"'layout' must be a Layout, but {type(layout)} was given.")
    if not isinstance(log_gate, str):
        raise TypeError(f"'log_gate' must be a str, but {type(log_gate)} was given.")

    anc_qubits = layout.get_qubits(role="anc")
    new_stab_gens = {}
    for anc_qubit in anc_qubits:
        log_gate_attrs = layout.param(log_gate, anc_qubit)
        if log_gate_attrs is None:
            raise ValueError(
                f"New stabilizer generators for {log_gate} "
                f"are not specified for qubit {anc_qubit}."
                "They should be setted with 'surface_sim.log_gates'."
            )
        new_stab_gens[anc_qubit] = log_gate_attrs["new_stab_gen"]

    return new_stab_gens


def get_support_from_adj_matrix(
    adjacency_matrix: xr.DataArray, anc_qubits: Sequence[str]
) -> dict[str, list[str]]:
    """Returns a dictionary that maps ancilla qubits to the data qubits
    they have support on.

    Parameters
    ----------
    adjacency_matrix
        Adjacency matrix of the qubits in the layout.
        It must have ``to_qubit`` and ``from_qubit`` coordinates.
        It can be built using ``surface_sim.Layout.adjacency_matrix``.
    anc_qubits
        Sequence of ancilla qubits for which to compute they data-qubit
        support.

    Returns
    -------
    support
        Dictionary with ancilla qubits as keys, whose values are a list
        of the data qubits they have support on.
    """
    if not isinstance(adjacency_matrix, xr.DataArray):
        raise TypeError(
            f"'adjacency_matrix' must be a xr.DataArray, but {type(adjacency_matrix)} was given."
        )
    if not isinstance(anc_qubits, Sequence):
        raise TypeError(
            f"'anc_qubits' must be a collection, but {type(anc_qubits)} was given."
        )

    support = {}
    for anc_qubit in anc_qubits:
        support_vec = adjacency_matrix.sel(from_qubit=anc_qubit)
        data_qubits = [
            q
            for q, sup in zip(support_vec.to_qubit.values.tolist(), support_vec)
            if sup
        ]
        support[anc_qubit] = data_qubits

    return support
