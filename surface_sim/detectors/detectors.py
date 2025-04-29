from __future__ import annotations
from collections.abc import Callable, Iterable, Sequence
from copy import deepcopy
import stim

from ..layouts.layout import Layout


class Detectors:
    def __init__(
        self,
        anc_qubits: Sequence[str],
        frame: str,
        anc_coords: dict[str, Sequence[float | int]] | None = None,
        include_gauge_dets: bool = False,
    ) -> None:
        """Initalises the ``Detectors`` class.

        Parameters
        ----------
        anc_qubits
            List of ancilla qubits.
            The detector ordering will follow this list.
        frame
            Detector frame to use when building the detectors.
            The options for the detector frames are described in the Notes section.
        anc_coords
            Ancilla qubit coordinates that are added to the detectors if specified.
            The coordinates of the detectors will be ``(*ancilla_coords[i], r)``,
            with ``r`` the number of rounds (starting at 0).
        include_gauge_dets
            Flag to include or not the definition of gauge detectors.
            By default, ``False``.

        Notes
        -----
        Detector frame ``'post-gate'`` builds the detectors in the basis given by the
        stabilizer generators of the last-measured QEC round.

        Detector frame ``'pre-gate'`` builds the detectors in the basis given by the
        stabilizer generators of the previous-last-measured QEC round.

        Detector frame ``'gate-independent'`` builds the detectors as ``m_{a,r} ^ m_{a,r-1}``
        independently of how the stabilizer generators have been transformed.
        """
        if not isinstance(anc_qubits, Sequence):
            raise TypeError(
                f"'anc_qubits' must be a Sequence, but {type(anc_qubits)} was given."
            )
        if not isinstance(frame, str):
            raise TypeError(f"'frame' must be a str, but {type(frame)} was given.")
        if frame not in ["pre-gate", "post-gate", "gate-independent"]:
            raise ValueError(
                "'frame' must be 'pre-gate', 'post-gate', or 'gate-independent',"
                f" but {frame} was given."
            )
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
        self.include_gauge_dets = include_gauge_dets

        self.new_circuit()

        return

    @classmethod
    def from_layouts(
        cls: type[Detectors],
        frame: str,
        *layouts: Layout,
        include_gauge_dets: bool = False,
    ) -> "Detectors":
        """Creates a ``Detectors`` object using the information from the layouts.
        It loads all the ancilla qubits and their coordinates.
        """
        anc_coords, anc_qubits = {}, []
        for layout in layouts:
            anc_coords |= layout.anc_coords  # updates dict
            anc_qubits += layout.anc_qubits
        return cls(
            anc_qubits=anc_qubits,
            frame=frame,
            anc_coords=anc_coords,
            include_gauge_dets=include_gauge_dets,
        )

    def new_circuit(self):
        """Resets all the current generators and number of rounds in order
        to create a different circuit.
        """
        self.detectors = {}  # {anc_label: propagation = set of ancilla labels}
        self.num_rounds = {a: 0 for a in self.anc_qubit_labels}
        self.total_num_rounds = 0
        self.update_dict_list = []
        self.gauge_detectors = set()
        return

    def activate_detectors(
        self, anc_qubits: Iterable[str], gauge_dets: Iterable[str] | None = None
    ):
        """Activates the given ancilla detectors.

        Parameters
        ----------
        anc_qubits
            List of ancilla detectors to activate.
        gauge_dets
            List of ancilla detectors that do not have a deterministic
            outcome in their first QEC cycle. This is only important if
            ``include_gauge_dets = False`` was set when initializing this object.
        """
        if not isinstance(anc_qubits, Iterable):
            raise TypeError(
                f"'anc_qubits' must be an Iterable, but {type(anc_qubits)} was given."
            )
        if set(anc_qubits) > set(self.anc_qubit_labels):
            raise ValueError(
                "Elements in 'anc_qubits' are not ancilla qubits in this object."
            )
        if not set(anc_qubits).isdisjoint(self.detectors):
            raise ValueError("Ancilla(s) were already active.")
        if (gauge_dets is None) and (not self.include_gauge_dets):
            raise ValueError(
                "When not including gauge detectors, one must specify 'gauge_dets'."
            )
        if gauge_dets is None:
            gauge_dets = []
        if not isinstance(gauge_dets, Iterable):
            raise TypeError(
                f"'gauge_dets' must be an Iterable, but {type(gauge_dets)} was given."
            )
        if set(gauge_dets) > set(self.anc_qubit_labels):
            raise ValueError(
                "Elements in 'gauge_dets' are not ancilla qubits in this object."
            )
        if not set(gauge_dets).isdisjoint(self.gauge_detectors):
            raise ValueError("Ancilla(s) were already set as gauge detectors.")

        for anc in anc_qubits:
            self.detectors[anc] = set([anc])
            self.num_rounds[anc] = 0

        self.gauge_detectors.update(gauge_dets)

        return

    def deactivate_detectors(self, anc_qubits: Iterable[str]):
        """Deactivates the given ancilla detectors."""
        if not isinstance(anc_qubits, Iterable):
            raise TypeError(
                f"'anc_qubits' must be an Iterable, but {type(anc_qubits)} was given."
            )
        if set(anc_qubits) > set(self.anc_qubit_labels):
            raise ValueError(
                "Elements in 'anc_qubits' are not ancilla qubits in this object."
            )

        for anc in anc_qubits:
            exists = self.detectors.pop(anc, None)
            if exists is None:
                raise ValueError(f"Ancilla {anc} was already inactive.")

        self.gauge_detectors.difference_update(anc_qubits)

        return

    def update(
        self, new_stab_gens: dict[str, set[str]], new_stab_gens_inv: dict[str, set[str]]
    ) -> None:
        """Update the current stabilizer generators with the dictionary
        descriving the effect of the logical gate. It allows to perform
        more than one logical gate between QEC cycles.

        See module ``surface_sim.log_gates`` to see how to prepare
        the layout to run logical gates.

        Note that it does not really update the stabilizer generators but it
        stores the change. They are only updated when calling ``build_from_anc``
        and ``build_from_data`` functions due to the ``"post-gate"`` frame.
        This behavior is due to the ``"post-gate"`` frame.

        Parameters
        ---------
        new_stab_gens
            Dictionary that maps ancilla qubits (representing the stabilizer
            generators) to a list of ancilla qubits (representing the decomposition
            of propagated stabilizer generators through the logical gate
            in terms of the stabilizer generators).
            If the dictionary is missing ancillas, their stabilizer generators
            are assumed to not be transformed by the logical gate.
            See ``get_new_stab_dict_from_layout`` for more information.
            For example, ``{"X1": ["X1", "Z1"]}`` is interpreted as that the
            logical gate has transformed ``X1`` to ``X1*Z1``.
        new_stab_gens_inv:
            Same as ``new_stab_gens`` for the logical gate inverse.

        Notes
        -----
        The ``new_stab_gens`` dictionary (or equivalently matrix) can be calculated by

        .. math::

            S'_i = U_L^\\dagger S_i U_L

        with :math:`U_L` the logical gate and :math:`S_i` (:math:`S'_i`) the
        stabilizer generator :math:`i` before (after) the logical gate.
        From `this reference <https://arthurpesah.me/blog/2023-03-16-stabilizer-formalism-2/>`_.

        The ``new_stab_gens_inv`` dictionary can be calculated by

        .. math::

            S'_i = U_L S_i U_L^\\dagger

        """
        if self.frame == "gate-independent":
            return
        elif self.frame == "pre-gate":
            # make a copy because the dict is modified on place later on
            update_dict = deepcopy(new_stab_gens)
        elif self.frame == "post-gate":
            # make a copy because the dict is modified on place later on
            update_dict = deepcopy(new_stab_gens_inv)

        if not isinstance(update_dict, dict):
            raise TypeError(
                "'new_stab_gens' and 'new_stab_gens_inv' must be a dict, "
                f"but {type(update_dict)} was given."
            )
        if any(not isinstance(s, set) for s in update_dict.values()):
            raise TypeError(
                "Elements in 'new_stab_gens' and 'new_stab_gens_inv' must be sets."
            )
        if set(update_dict) > set(self.anc_qubit_labels):
            raise ValueError(
                "Elements in 'new_stab_gens' and 'new_stabs_gens_inv' are not "
                "ancilla qubits in this Detectors class."
            )

        for anc, propagation in update_dict.items():
            # this is useful for updating the self.detector progations
            propagation.symmetric_difference_update([anc])

        if self.frame == "pre-gate":
            self.update_dict_list.append(update_dict)  # insert at the end
        elif self.frame == "post-gate":
            self.update_dict_list.insert(0, update_dict)  # insert at beginning

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

        Notes
        -----
        This function assumes that all QEC cycles happen at the same time
        for all logical qubits. It is not possible to have some qubits
        performing some logical gates and some qubits performing QEC cycles.
        This is because the dicts for updating the qubits are stored globally,
        not per ancilla qubit.
        """
        if anc_qubits is None:
            # use only active detectors
            anc_qubits = list(self.detectors)
        if not isinstance(anc_qubits, Iterable):
            raise TypeError(
                f"'anc_qubits' must be an Iterable or None, but {type(anc_qubits)} was given."
            )
        if not isinstance(get_rec, Callable):
            raise TypeError(
                f"'get_rec' must be callable, but {type(get_rec)} was given."
            )

        # remove any inactive detector that was given, this is caused by the
        # arbitrary logical circuit generator becuase if we have M 0 I 1 TICK
        # the 'TICK'/QEC cycle will try to build the detectors for logical qubit 0,
        # which have been deactivated by the measurement.
        anc_qubits = [q for q in anc_qubits if q in self.detectors]
        if not self.include_gauge_dets:
            anc_qubits = [q for q in anc_qubits if q not in self.gauge_detectors]
        anc_qubits = set(anc_qubits)

        self.total_num_rounds += 1
        for anc in self.detectors:
            self.num_rounds[anc] += 1

        if self.frame == "gate-independent":
            # build the detectors
            meas_comp = -2 if anc_reset else -3
            detectors = {}
            for anc in anc_qubits:
                dets = [(anc, -1)]
                if meas_comp + self.num_rounds[anc] >= 0:
                    dets.append((anc, meas_comp))
                detectors[anc] = dets
        else:
            # update the detectors given the logical gates
            for update_dict in self.update_dict_list:
                for propagation in self.detectors.values():
                    for q in deepcopy(propagation):
                        # as the detectors are updated PER LOGICAL GATE,
                        # the stabilizers of the 2nd logical qubit are not
                        # in the update list of the 1st logical qubit if the
                        # gate is not a two-qubit gate
                        if q in update_dict:
                            propagation.symmetric_difference_update(update_dict[q])

            # build the detectors
            detectors = {}
            anc_reset_curr, anc_reset_prev = anc_reset, anc_reset
            for anc_qubit, (p_gen, c_gen) in zip(
                self.detectors, self.detectors.items()
            ):
                p_gen = set([p_gen])  # p_gen is just the label of the ancilla

                if self.frame == "post-gate":
                    c_gen, p_gen = p_gen, c_gen

                targets = [(q, -1) for q in c_gen]
                if self.num_rounds[anc_qubit] >= 2:
                    targets += [(q, -2) for q in p_gen]

                if not anc_reset_curr and self.num_rounds[anc_qubit] >= 2:
                    targets += [(q, -2) for q in c_gen]
                if not anc_reset_prev and self.num_rounds[anc_qubit] >= 3:
                    targets += [(q, -3) for q in p_gen]

                detectors[anc_qubit] = targets

        # build the stim circuit
        # the detectors are built in the same ordering as 'self.anc_qubit_labels' to
        # make it reproducible and so that the user can choose it.
        detectors_stim = stim.Circuit()
        for anc in self.anc_qubit_labels:
            if anc in anc_qubits:
                # simplify the expression of the detectors by removing the pairs
                targets = detectors[anc]
                targets = remove_pairs(targets)
                detectors_rec = [get_rec(*t) for t in targets]
            else:
                # create the detector but make it be always 0
                detectors_rec = []
            coords = [*self.anc_coords[anc], self.total_num_rounds - 1]
            instr = stim.CircuitInstruction(
                "DETECTOR", gate_args=coords, targets=detectors_rec
            )
            detectors_stim.append(instr)

        # reset detectors and update_dict list
        self.detectors = {q: set([q]) for q in self.detectors}
        self.update_dict_list = []
        self.gauge_detectors = set()

        return detectors_stim

    def build_from_data(
        self,
        get_rec: Callable,
        anc_support: dict[str, Iterable[str]],
        anc_reset: bool,
        reconstructable_stabs: Iterable[str],
        anc_qubits: Iterable[str] | None = None,
    ) -> stim.Circuit:
        """Returns the stim circuit with the corresponding detectors
        given that the data qubits have been measured.

        Note that the detectors for the ``"pre-gate"`` and ``"post-gate"``
        frames are both constructed in the ``"post-gate"`` frame! See section Notes
        for more explanation.

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
            See ``surface_sim.Layout.get_support`` for more information.
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

        Notes
        -----
        The reason that the detectors in the ``"pre-gate"`` frame are built in
        the ``"post-gate"`` frame is that there can be situations in which the
        detectors cannot be built in the ``"pre-gate"`` frame. For example,

        R 0 1
        TICK
        CX 0 1
        M 0

        As there is no QEC cycle performed in (logical) qubit 1 and the stabilizer
        generators of qubit 0 are propagated to qubit 1, we cannot build the
        detectors in the ``"pre-gate"`` frame.

        Note that if one always performs (at least) one QEC cycle after each logical
        gate, then there is no difference in building the detectors for the
        measurement in the ``"pre-gate"`` or in the ``"post-gate"`` frame
        as one will always have:

        TICK
        M 0
        """
        if not isinstance(reconstructable_stabs, Iterable):
            raise TypeError(
                "'reconstructable_stabs' must be iterable, "
                f"but {type(reconstructable_stabs)} was given."
            )
        if anc_qubits is None:
            # use only active detectors
            anc_qubits = list(self.detectors)
        if not isinstance(anc_qubits, Iterable):
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
        reconstructable_stabs = set(reconstructable_stabs)
        anc_qubits = [q for q in anc_qubits if q in reconstructable_stabs]
        if not self.include_gauge_dets:
            anc_qubits = [q for q in anc_qubits if q not in self.gauge_detectors]

        # Logical measurement is not considered a QEC cycle but a logical operation.
        # therefore, it does not increase the number of rounds.
        # However, the way building the detectors is implemented relies on
        # faking that ancilla qubits have been measured instead of the data qubits.
        fake_num_rounds = deepcopy(self.num_rounds)
        for anc in self.detectors:
            fake_num_rounds[anc] += 1

        if self.frame == "gate-independent":
            anc_detectors = {}
            for anc in anc_qubits:
                dets = [(anc, -1)]
                if fake_num_rounds[anc] > 1:
                    dets.append((anc, -2))
                if (not anc_reset) and (fake_num_rounds[anc] > 2):
                    dets.append((anc, -3))
                anc_detectors[anc] = dets
        else:
            # always use the "post-gate" frame
            if self.frame == "pre-gate":
                self.update_dict_list.reverse()

            # update the detectors given the logical gates
            for update_dict in self.update_dict_list:
                for propagation in self.detectors.values():
                    for q in deepcopy(propagation):
                        # as the detectors are updated PER LOGICAL GATE,
                        # the stabilizers of the 2nd logical qubit are not
                        # in the update list of the 1st logical qubit if the
                        # gate is not a two-qubit gate
                        if q in update_dict:
                            propagation.symmetric_difference_update(update_dict[q])

            # build the detectors
            anc_detectors = {}
            anc_reset_curr, anc_reset_prev = True, anc_reset
            for anc_qubit, (p_gen, c_gen) in zip(
                self.detectors, self.detectors.items()
            ):
                p_gen = set([p_gen])  # p_gen is just the label of the ancilla

                # always use the "post-gate" frame
                c_gen, p_gen = p_gen, c_gen

                targets = [(q, -1) for q in c_gen]
                if fake_num_rounds[anc_qubit] >= 2:
                    targets += [(q, -2) for q in p_gen]

                if not anc_reset_curr and fake_num_rounds[anc_qubit] >= 2:
                    targets += [(q, -2) for q in c_gen]
                if not anc_reset_prev and fake_num_rounds[anc_qubit] >= 3:
                    targets += [(q, -3) for q in p_gen]

                anc_detectors[anc_qubit] = targets

        # udpate the (anc, -1) to a the corresponding set of (data, -1)
        detectors = {}
        for anc_qubit in anc_qubits:
            dets = anc_detectors[anc_qubit]
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

        # build the stim circuit
        # the detectors are built in the same ordering as 'self.anc_qubit_labels' to
        # make it reproducible and so that the user can choose it.
        # Contrary to 'build_from_anc' we do not add empty detectors because
        # only one (logical) qubit is measured, so we don't need to report the stabilizers
        # of other (logical) qubits.
        detectors_stim = stim.Circuit()
        for anc in reconstructable_stabs:
            if anc in anc_qubits:
                # simplify the expression of the detectors by removing the pairs
                targets = detectors[anc]
                targets = remove_pairs(targets)
                detectors_rec = [get_rec(*t) for t in targets]
            else:
                # create the detector but make it be always 0
                detectors_rec = []
            coords = [*self.anc_coords[anc], self.total_num_rounds - 0.5]
            instr = stim.CircuitInstruction(
                "DETECTOR", gate_args=coords, targets=detectors_rec
            )
            detectors_stim.append(instr)

        # reset detectors, but the update_dict_list is not updated
        # as there could qubits performing the QEC cycle that have undergone
        # some logical gates. However, it needs to be reversed to counteract
        # the reverse suffered during this function (only for the pre-gate frame).
        self.detectors = {q: set([q]) for q in self.detectors}
        if self.frame == "pre-gate":
            self.update_dict_list.reverse()

        return detectors_stim


def get_new_stab_dict_from_layout(
    layout: Layout, log_gate: str
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
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
    new_stab_gens_inv
        Same as ``new_stab_gens`` but for the gate inverse.
    """
    if not isinstance(layout, Layout):
        raise TypeError(f"'layout' must be a Layout, but {type(layout)} was given.")
    if not isinstance(log_gate, str):
        raise TypeError(f"'log_gate' must be a str, but {type(log_gate)} was given.")

    anc_qubits = layout.anc_qubits
    new_stab_gens = {}
    new_stab_gens_inv = {}
    for anc_qubit in anc_qubits:
        log_gate_attrs = layout.param(log_gate, anc_qubit)
        if log_gate_attrs is None:
            raise ValueError(
                f"New stabilizer generators for {log_gate} "
                f"are not specified for qubit {anc_qubit}."
                "They should be setted with 'surface_sim.log_gates'."
            )
        new_stab_gens[anc_qubit] = set(log_gate_attrs["new_stab_gen"])
        new_stab_gens_inv[anc_qubit] = set(log_gate_attrs["new_stab_gen_inv"])

    return new_stab_gens, new_stab_gens_inv


def remove_pairs(elements: list) -> list:
    """Removes all possible pairs inside the given list."""
    output = []
    for element in elements:
        if elements.count(element) % 2 == 1:
            output.append(element)
    return output
