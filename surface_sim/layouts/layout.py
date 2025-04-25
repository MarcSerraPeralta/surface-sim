from __future__ import annotations
from collections.abc import Sequence
from typing import TypedDict, overload, Literal

from copy import deepcopy
from os import path
from pathlib import Path

import networkx as nx
import numpy as np
import yaml
from xarray import DataArray

IntDirections = list[str]
IntOrder = IntDirections | dict[str, IntDirections]


class LogQubitsDict(TypedDict):
    ind: int
    log_x: Sequence[str]
    log_z: Sequence[str]


class QubitDict(TypedDict):
    qubit: str
    ind: int
    neighbors: dict[str, str]


class LayoutDict(TypedDict):
    code: str
    logical_qubits: dict[str, LogQubitsDict]
    interaction_order: dict[str, list]
    layout: Sequence[QubitDict]


class Layout:
    """Layout class for a QEC code.

    Initialization and storage
    --------------------------
    - ``__init__``
    - ``__copy__``
    - ``from_dict``
    - ``to_dict``
    - ``from_yaml``
    - ``to_yaml``

    Get information from physical qubits
    ------------------------------------
    - ``param``
    - ``get_inds``
    - ``qubit_inds``
    - ``get_max_ind``
    - ``get_min_ind``
    - ``get_qubits``
    - ``get_neighbors``
    - ``get_coords``
    - ``get_support``
    - ``get_labels_from_inds``
    - ``qubits``
    - ``data_qubits``
    - ``anc_qubits``
    - ``num_qubits``
    - ``num_data_qubits``
    - ``num_anc_qubits``
    - ``qubit_coords``
    - ``anc_coords``
    - ``data_coords``

    Get information from logical qubits
    -----------------------------------
    - ``logical_param``
    - ``get_logical_inds``
    - ``logical_qubit_inds``
    - ``get_max_logical_ind``
    - ``get_min_logical_ind``
    - ``get_logical_qubits``
    - ``get_logical_labels_from_inds``

    Set information
    ---------------
    - ``set_param``

    Matrix generation
    -----------------
    - ``adjacency_matrix``
    - ``expansion_matrix``
    - ``projection_matrix``

    Notes
    -----
    The parameters of the layout must be updated or set by the appropiate
    methods of this class to guarantee a correct behavior.
    In this sense, the qubits and their connections
    cannot be updated once the layout object has been created.
    """

    ############################
    # initialization and storage

    def __init__(self, setup: LayoutDict) -> None:
        """Initiailizes the layout for a particular code.

        Parameters
        ----------
        setup
            The layout setup, provided as a dict.

            The setup dictionary is expected to have a 'layout' item, containing
            a list of dictionaries. Each such dictionary (``dict[str, object]``) must define the
            qubit label (``str``) corresponding the ``'qubit'`` item. In addition, each dictionary
            must also have a ``'neighbors'`` item that defines a dictonary (``dict[str, str]``)
            of ordinal directions and neighbouring qubit labels. Apart from these two items,
            each dictionary can hold any other metadata or parameter relevant to these qubits.

            In addition to the layout list, the setup dictionary can also optionally
            define the name of the layout (``str``), a description (``str``) of the layout as well
            as the interaction order of the different types of check, if the layout is used
            for a QEC code.

        Raises
        ------
        ValueError
            If the type of the setup provided is not a dictionary.
        """
        if not isinstance(setup, dict):
            raise ValueError(f"'setup' must be a dict, instead got {type(setup)}.")

        self.name = setup.get("name", "")
        self.code = setup.get("code", "")
        self._log_qubits = setup.get("logical_qubits", {})
        self.distance = setup.get("distance", -1)
        self.distance_z = setup.get("distance_z", -1)
        self.distance_x = setup.get("distance_x", -1)
        self.description = setup.get("description")
        self.interaction_order = setup.get("interaction_order", {})

        self._load_layout(setup)
        self._check_logical_qubits()

        # precompute specific attributes
        # make then tuples so that they areunmutable
        self.qubits = tuple(self.get_qubits())
        self.data_qubits = tuple(self.get_qubits(role="data"))
        self.anc_qubits = tuple(self.get_qubits(role="anc"))
        self.logical_qubits = tuple(self._log_qubits)

        self.num_qubits = len(self.qubits)
        self.num_data_qubits = len(self.data_qubits)
        self.num_anc_qubits = len(self.anc_qubits)
        self.num_logical_qubits = len(self.logical_qubits)

        self._qubit_coords = {
            q: c for q, c in zip(self.qubits, self.get_coords(self.qubits))
        }
        self._data_qubit_coords = {
            q: c for q, c in zip(self.data_qubits, self.get_coords(self.data_qubits))
        }
        self._anc_qubit_coords = {
            q: c for q, c in zip(self.anc_qubits, self.get_coords(self.anc_qubits))
        }

        self._qubit_label_to_ind = {v: k for k, v in self.qubit_inds.items()}
        self._logical_qubit_label_to_ind = {
            v: k for k, v in self.logical_qubit_inds.items()
        }

        return

    def _check_logical_qubits(self) -> None:
        """Checks that the ``Layout._log_qubits`` has the correct structure."""
        if not isinstance(self._log_qubits, dict):
            raise TypeError(
                f"'logical_qubits' must be a dict, but {type(self._log_qubits)} was given."
            )
        if any(not isinstance(l, str) for l in self._log_qubits):
            raise TypeError(
                "The keys in 'logical_qubits' must be str, "
                f"but {list(self._log_qubits)} was given."
            )
        if any(not isinstance(l, dict) for l in self._log_qubits.values()):
            raise TypeError(
                "The values in 'logical_qubits' must be dict, "
                f"but {list(self._log_qubits.values())} was given."
            )

        for key in ["ind", "log_x", "log_z"]:
            if any(key not in p for p in self._log_qubits.values()):
                raise ValueError(f"Each logical qubit must have '{key}' specified.")

        for log_p in ["log_x", "log_z"]:
            for l, params in self._log_qubits.items():
                if not isinstance(params[log_p], Sequence):
                    raise TypeError(
                        f"'{log_p}' in logical {l} must be an Sequence, "
                        f"but {type(params[log_p])} was given."
                    )
                if set(params[log_p]) > set(self._qubit_inds):
                    raise ValueError(
                        f"'{log_p}' in logical {l} has support on qubits not present in this layout."
                    )

        return

    def _load_layout(self, setup: LayoutDict) -> None:
        """Internal function that loads the directed graph from the
        setup dictionary that is provided during initialization.

        Parameters
        ----------
        setup
            The setup dictionary that must specify the 'layout' list
            of dictionaries, containing the qubit informaiton.

        Raises
        ------
        ValueError
            If there are unlabeled qubits in the any of the layout dictionaries.
        ValueError
            If any qubit label is repeated in the layout list.
        """
        layout = deepcopy(setup.get("layout"))
        if layout is None:
            raise ValueError("'setup' does not contain a 'layout' key.")

        self._qubit_inds = {}
        self.graph = nx.DiGraph()

        for qubit_info in layout:
            qubit = qubit_info.pop("qubit", None)
            if qubit is None:
                raise ValueError("Each qubit in the layout must be labeled.")

            if qubit in self.graph:
                raise ValueError("Qubit label repeated, ensure labels are unique.")

            ind = qubit_info.get("ind", None)
            if ind is None:
                raise ValueError(
                    "Each qubit in the layout must be indexed (have 'ind' param)."
                )
            self._qubit_inds[qubit] = ind

            self.graph.add_node(qubit, **qubit_info)

        for node, attrs in self.graph.nodes(data=True):
            nbr_dict = attrs.get("neighbors", None)
            if nbr_dict is None:
                raise ValueError(
                    "All elements in 'setup' must have the 'neighbors' attribute."
                )

            for edge_dir, nbr_qubit in nbr_dict.items():
                if nbr_qubit is not None:
                    self.graph.add_edge(node, nbr_qubit, direction=edge_dir)

        if all((i is None) for i in self._qubit_inds.values()):
            qubits = list(self.graph.nodes)
            self._qubit_inds = dict(zip(qubits, range(len(qubits))))

        if any((i is None) for i in self._qubit_inds.values()):
            raise ValueError("Either all qubits have indicies or none of them.")

        if len(self._qubit_inds) != len(set(self._qubit_inds.values())):
            raise ValueError("Qubit index repeated, ensure indices are unique.")

        return

    def __copy__(self) -> Layout:
        """Copies the Layout."""
        return Layout(self.to_dict())

    @classmethod
    def from_dict(cls, setup: LayoutDict) -> "Layout":
        """Loads the layout class from a dictionary.

        Parameters
        ----------
        setup
            The layout setup, see ``Layout.__init__``.

        Returns
        -------
        Layout
            The initialized layout object.
        """
        return cls(setup)

    def to_dict(self) -> LayoutDict:
        """Return a setup dictonary for the layout.

        Returns
        -------
        setup
            The dictionary of the setup.
            A copyt of this ``Layout`` can be initalized using ``Layout(setup)``.
        """
        setup = dict()

        setup["name"] = self.name
        setup["code"] = self.code
        setup["distance"] = self.distance
        setup["distance_z"] = self.distance_z
        setup["distance_x"] = self.distance_x
        setup["logical_qubits"] = self._log_qubits
        setup["description"] = self.description
        setup["interaction_order"] = self.interaction_order

        layout = []
        for node, attrs in self.graph.nodes(data=True):
            node_dict = deepcopy(attrs)
            node_dict["qubit"] = node

            nbr_dict = dict()
            adj_view = self.graph.adj[node]

            for nbr_node, edge_attrs in adj_view.items():
                edge_dir = edge_attrs["direction"]
                nbr_dict[edge_dir] = nbr_node

            node_dict["neighbors"] = nbr_dict

            layout.append(node_dict)
        setup["layout"] = layout

        return setup

    @classmethod
    def from_yaml(cls, filename: str | Path) -> "Layout":
        """Loads the layout class from a YAML file.

        The file must define the setup dictionary that initializes
        the layout.

        Parameters
        ----------
        filename
            The pathfile name of the YAML setup file.

        Returns
        -------
        Layout
            The initialized layout object.
        """
        if not path.exists(filename):
            raise ValueError("Given path doesn't exist")

        with open(filename, "r") as file:
            layout_setup = yaml.safe_load(file)
            return cls(layout_setup)

    def to_yaml(self, filename: str | Path) -> None:
        """Saves the layout as a YAML file.

        Parameters
        ----------
        filename
            The pathfile name of the YAML setup file.

        """
        setup = self.to_dict()
        with open(filename, "w") as file:
            yaml.dump(setup, file, default_flow_style=False)

    ######################################
    # get information from physical qubits

    def param(self, param: str, qubit: str) -> object:
        """Returns the parameter value of a given qubit

        Parameters
        ----------
        param
            The label of the qubit parameter.
        qubit
            The label of the qubit that is being queried.

        Returns
        -------
        object
            The value of the parameter if specified for the given qubit,
            else ``None``.
        """
        if param not in self.graph.nodes[qubit]:
            return None
        else:
            return self.graph.nodes[qubit][param]

    def get_inds(self, qubits: Sequence[str]) -> tuple[int, ...]:
        """Returns the indices of the qubits.

        Parameters
        ----------
        qubits
            List of qubits.

        Returns
        -------
        The list of qubit indices.
        """
        return tuple(self._qubit_inds[qubit] for qubit in qubits)

    @property
    def qubit_inds(self) -> dict[str, int]:
        """Returns a dictionary mapping all the qubits to their indices."""
        return {k: v for k, v in self._qubit_inds.items()}

    def get_max_ind(self) -> int:
        """Returns the largest qubit index in the layout."""
        return max(self._qubit_inds.values())

    def get_min_ind(self) -> int:
        """Returns the smallest qubit index in the layout."""
        return min(self._qubit_inds.values())

    def get_qubits(self, **conds: object) -> tuple[str]:
        """Return the qubit labels that meet a set of conditions.

        Parameters
        ----------
        **conds
            Dictionary of the conditions.

        Returns
        -------
        nodes
            The list of qubit labels that meet all conditions.

        Notes
        -----
        The order that the qubits appear in is defined during the initialization
        of the layout and remains fixed.

        The conditions conds are the keyward arguments that specify the value (``object``)
        that each parameter label (``str``) needs to take.
        """
        if conds:
            node_view = self.graph.nodes(data=True)
            nodes = tuple(
                node for node, attrs in node_view if valid_attrs(attrs, **conds)
            )
            return nodes

        return tuple(self.graph.nodes)

    @overload
    def get_neighbors(
        self,
        qubits: Sequence[str],
        direction: str | None = None,
        as_pairs: Literal[False] = False,
    ) -> tuple[str, ...]: ...

    @overload
    def get_neighbors(
        self,
        qubits: Sequence[str],
        direction: str | None = None,
        as_pairs: Literal[True] = True,
    ) -> tuple[str, ...]: ...

    def get_neighbors(
        self,
        qubits: Sequence[str],
        direction: str | None = None,
        as_pairs: bool = False,
    ) -> tuple[str, ...] | tuple[tuple[str, str], ...]:
        """Returns the list of qubit labels, neighboring specific qubits
        that meet a set of conditions.

        Parameters
        ----------
        qubits
            The qubit labels, whose neighbors are being considered.

        direction
            The direction along which to consider the neigbors along.

        Returns
        -------
        end_notes
            The list of qubit label, neighboring qubit, that meet the conditions.

        Notes
        -----
        The order that the qubits appear in is defined during the initialization
        of the layout and remains fixed.

        The conditions conds are the keyward arguments that specify the value (``object``)
        that each parameter label (``str``) needs to take.
        """
        edge_view = self.graph.out_edges(qubits, data=True)

        start_nodes = []
        end_nodes = []
        for start_node, end_node, attrs in edge_view:
            if direction is None or attrs["direction"] == direction:
                start_nodes.append(start_node)
                end_nodes.append(end_node)

        if as_pairs:
            return tuple(zip(start_nodes, end_nodes))
        return tuple(end_nodes)

    def get_coords(self, qubits: Sequence[str]) -> tuple[tuple[float | int], ...]:
        """Returns the coordinates of the given qubits.

        Parameters
        ----------
        qubits
            List of qubits.

        Returns
        -------
        Coordinates of the given qubits.
        """
        all_coords = nx.get_node_attributes(self.graph, "coords")

        if set(qubits) > set(all_coords):
            raise ValueError("Some of the given qubits do not have coordinates.")

        return tuple(tuple(all_coords[q]) for q in qubits)

    def get_support(self, qubits: Sequence[str]) -> dict[str, tuple[str, ...]]:
        """Returns a dictionary mapping the qubits to their support."""
        return {q: self.get_neighbors([q]) for q in qubits}

    def get_labels_from_inds(self, inds: Sequence[int]) -> tuple[str, ...]:
        """Returns list of qubit labels for the given qubit indicies."""
        return tuple(self._qubit_label_to_ind[ind] for ind in inds)

    @property
    def qubit_coords(self) -> dict[str, tuple[float | int, ...]]:
        """Returns a dictionary mapping all the qubits to their coordinates."""
        return {k: v for k, v in self._qubit_coords.items()}

    @property
    def anc_coords(self) -> dict[str, tuple[float | int, ...]]:
        """Returns a dictionary mapping all ancilla qubits to their coordinates."""
        return {k: v for k, v in self._anc_qubit_coords.items()}

    @property
    def data_coords(self) -> dict[str, tuple[float | int, ...]]:
        """Returns a dictionary mapping all data qubits to their coordinates."""
        return {k: v for k, v in self._data_qubit_coords.items()}

    #####################################
    # get information from logical qubits

    def logical_param(self, param: str, logical_qubit: str) -> object:
        """Returns the parameter value of a given logical qubit.

        Parameters
        ----------
        param
            The label of the logical qubit parameter.
        logical_qubit
            The label of the logical qubit that is being queried.

        Returns
        -------
        object
            The value of the parameter if specified for the given logical qubit,
            else ``None``.
        """
        params = self._log_qubits.get(logical_qubit)
        if params is None:
            return None
        return params.get(param)

    def get_logical_inds(self, logical_qubits: Sequence[str]) -> tuple[int, ...]:
        """Returns the indices of the specified logical qubits."""
        if set(logical_qubits) > set(self._log_qubits):
            raise ValueError(
                f"At least one of the given logical qubits ({logical_qubits}) are not present in this layout."
            )
        return tuple(self._log_qubits[l]["ind"] for l in logical_qubits)

    @property
    def logical_qubit_inds(self) -> dict[str, int]:
        """Returns a dictionary mapping all the logical qubits to their indices."""
        return {k: self._log_qubits[k]["ind"] for k in self._log_qubits}

    def get_max_logical_ind(self) -> int:
        """Returns the largest logical qubit index in the layout."""
        return max(self.logical_qubit_inds.values())

    def get_min_logical_ind(self) -> int:
        """Returns the largest logical qubit index in the layout."""
        return min(self.logical_qubit_inds.values())

    def get_logical_labels_from_inds(self, inds: Sequence[int]) -> tuple[str, ...]:
        """Returns list of logical qubit labels for the given logical qubit indicies."""
        label_to_ind = {v: k for k, v in self.logical_qubit_inds.items()}
        return tuple(label_to_ind[ind] for ind in inds)

    #################
    # set information

    def set_param(self, param: str, qubit: str, value: object) -> None:
        """Sets the value of a given qubit parameter

        Parameters
        ----------
        param
            The label of the qubit parameter.
        qubit
            The label of the qubit that is being queried.
        value
            The new value of the qubit parameter.
        """
        self.graph.nodes[qubit][param] = value

    ###################
    # matrix generation

    def adjacency_matrix(self) -> DataArray:
        """Returns the adjaceny matrix corresponding to the layout.

        The layout is encoded as a directed graph, such that there are two edges
        in opposite directions between each pair of neighboring qubits.

        Returns
        -------
        ajd_matrix
            The adjacency matrix.
        """
        qubits = list(self.qubits)
        adj_matrix = nx.adjacency_matrix(self.graph)

        data_arr = DataArray(
            data=adj_matrix.toarray(),
            dims=["from_qubit", "to_qubit"],
            coords=dict(
                from_qubit=qubits,
                to_qubit=qubits,
            ),
        )
        return data_arr

    def expansion_matrix(self) -> DataArray:
        """Returns the expansion matrix corresponding to the layout.
        The matrix can expand a vector of measurements/defects to a 2D array
        corresponding to layout of the ancilla qubits.
        Used for convolutional neural networks.

        Returns
        -------
        DataArray
            The expansion matrix.
        """
        node_view = self.graph.nodes(data=True)

        coords = [node_view[anc]["coords"] for anc in self.anc_qubits]

        rows, cols = zip(*coords)

        row_inds, num_rows = index_coords(rows, reverse=True)
        col_inds, num_cols = index_coords(cols)

        anc_inds = range(self.num_anc_qubits)

        tensor = np.zeros((self.num_anc_qubits, num_rows, num_cols), dtype=bool)
        tensor[anc_inds, row_inds, col_inds] = True
        expanded_tensor = np.expand_dims(tensor, axis=1)

        expansion_tensor = DataArray(
            expanded_tensor,
            dims=["anc_qubit", "channel", "row", "col"],
            coords=dict(
                anc_qubit=list(self.anc_qubits),
            ),
        )
        return expansion_tensor

    def projection_matrix(self, stab_type: str) -> DataArray:
        """Returns the projection matrix, mapping
        data qubits (defined by a parameter ``'role'`` equal to ``'data'``)
        to ancilla qubits (defined by a parameter ``'role'`` equal to ``'anc'``)
        measuing a given stabilizerr type (defined by a parameter
        ``'stab_type'`` equal to stab_type).

        This matrix can be used to project a final set of data-qubit
        measurements to a set of syndromes.

        Parameters
        ----------
        stab_type
            The type of the stabilizers that the data qubit measurement
            is being projected to.

        Returns
        -------
        DataArray
            The projection matrix.
        """
        adj_mat = self.adjacency_matrix()

        anc_qubits = list(self.get_qubits(role="anc", stab_type=stab_type))

        proj_mat = adj_mat.sel(from_qubit=list(self.data_qubits), to_qubit=anc_qubits)
        return proj_mat.rename(from_qubit="data_qubit", to_qubit="anc_qubit")


def valid_attrs(attrs: dict[str, object], **conditions: object) -> bool:
    """Checks if the items in attrs match each condition in conditions.
    Both attrs and conditions are dictionaries mapping parameter labels (str)
    to values (object).

    Parameters
    ----------
    attrs
        The attribute dictionary.

    Returns
    -------
    bool
        Whether the attributes meet a set of conditions.
    """
    for key, val in conditions.items():
        attr_val = attrs[key]
        if attr_val is None or attr_val != val:
            return False
    return True


def index_coords(
    coords: tuple[int, ...], reverse: bool = False
) -> tuple[tuple[int, ...], int]:
    """Indexes a list of coordinates.

    Parameters
    ----------
    coords
        The list of coordinates.
    reverse
        Whether to return the values in reverse, by default False

    Returns
    -------
    indices
        The list of indexed coordinates.
    num_unique_vals
        The number of unique coordinates.
    """
    unique_vals = set(coords)
    num_unique_vals = len(unique_vals)

    if reverse:
        unique_inds = reversed(range(num_unique_vals))
    else:
        unique_inds = range(num_unique_vals)

    mapping = dict(zip(unique_vals, unique_inds))

    indicies = tuple(mapping[coord] for coord in coords)
    return indicies, num_unique_vals
