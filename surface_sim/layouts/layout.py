from __future__ import annotations
from collections.abc import Iterable

from copy import deepcopy
from os import path
from pathlib import Path

import networkx as nx
import numpy as np
import yaml
from xarray import DataArray

IntDirections = list[str]
IntOrder = IntDirections | dict[str, IntDirections]


class Layout:
    """Layout class for a QEC code.

    Initialization and storage
    --------------------------
    - ``__init__``
    - ``__copy__``
    - ``to_dict``
    - ``from_yaml``
    - ``to_yaml``

    Get information
    ---------------
    - ``param``
    - ``get_inds``
    - ``qubit_inds``
    - ``get_max_ind``
    - ``get_min_ind``
    - ``get_qubits``
    - ``get_logical_qubits``
    - ``get_neighbors``
    - ``get_coords``
    - ``qubit_coords``
    - ``anc_coords``

    Set information
    ---------------
    - ``set_param``

    Matrix generation
    -----------------
    - ``adjacency_matrix``
    - ``expansion_matrix``
    - ``projection_matrix``

    """

    def __init__(self, setup: dict[str, object]) -> None:
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
        self._log_qubits = setup.get("logical_qubit_labels", [])
        self.distance = setup.get("distance", -1)
        self.distance_z = setup.get("distance_z", -1)
        self.distance_x = setup.get("distance_x", -1)
        self.log_z = setup.get("log_z", {})
        self.log_x = setup.get("log_x", {})
        self.description = setup.get("description")
        self.interaction_order = setup.get("interaction_order", {})
        self._qubit_inds = {}

        if (set(self._log_qubits) != set(self.log_z)) or (
            set(self._log_qubits) != set(self.log_x)
        ):
            raise ValueError(
                "'logical_qubit_labels' does not match 'log_x' and/or 'log_z'."
            )

        self.graph = nx.DiGraph()
        self._load_layout(setup)

        return

    def __copy__(self) -> Layout:
        """Copies the Layout."""
        return Layout(self.to_dict())

    def to_dict(self) -> dict[str, object]:
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
        setup["logical_qubit_labels"] = self._log_qubits
        setup["log_z"] = self.log_z
        setup["log_x"] = self.log_x
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

            for ver_dir in ("north", "south"):
                for hor_dir in ("east", "west"):
                    edge_dir = f"{ver_dir}_{hor_dir}"
                    if edge_dir not in nbr_dict:
                        nbr_dict[edge_dir] = None

            node_dict["neighbors"] = nbr_dict

            layout.append(node_dict)
        setup["layout"] = layout
        return setup

    def get_inds(self, qubits: Iterable[str]) -> list[int]:
        """Returns the indices of the qubits.

        Parameters
        ----------
        qubits
            List of qubits.

        Returns
        -------
        The list of qubit indices.
        """
        return [self._qubit_inds[qubit] for qubit in qubits]

    def qubit_inds(self) -> dict[str, int]:
        """Returns a dictionary mapping all the qubits to their indices."""
        return {k: v for k, v in self._qubit_inds.items()}

    def get_max_ind(self) -> int:
        """Returns the largest qubit index in the layout."""
        return max(self._qubit_inds.values())

    def get_min_ind(self) -> int:
        """Returns the smallest qubit index in the layout."""
        return min(self._qubit_inds.values())

    def get_qubits(self, **conds: object) -> list[str]:
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
            nodes = [node for node, attrs in node_view if valid_attrs(attrs, **conds)]
            return nodes

        nodes = list(self.graph.nodes)
        return nodes

    def get_logical_qubits(self) -> list[str]:
        """Returns the logical qubit labels."""
        return deepcopy(self._log_qubits)

    def get_neighbors(
        self,
        qubits: Iterable[str],
        direction: str | None = None,
        as_pairs: bool = False,
    ) -> list[str] | list[tuple[str, str]]:
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
            return list(zip(start_nodes, end_nodes))
        return end_nodes

    def get_coords(self, qubits: Iterable[str]) -> list[list[float | int]]:
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

        return [all_coords[q] for q in qubits]

    def qubit_coords(self) -> dict[str, list[float | int]]:
        """Returns a dictionary mapping all the qubits to their coordinates."""
        return nx.get_node_attributes(self.graph, "coords")

    def anc_coords(self) -> dict[str, list[float | int]]:
        """Returns a dictionary mapping all ancilla qubits to their coordinates."""
        anc_qubits = self.get_qubits(role="anc")
        anc_coords = self.get_coords(anc_qubits)
        return {q: c for q, c in zip(anc_qubits, anc_coords)}

    def adjacency_matrix(self) -> DataArray:
        """Returns the adjaceny matrix corresponding to the layout.

        The layout is encoded as a directed graph, such that there are two edges
        in opposite directions between each pair of neighboring qubits.

        Returns
        -------
        ajd_matrix
            The adjacency matrix.
        """
        qubits = self.get_qubits()
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

        anc_qubits = [node for node, data in node_view if data["role"] == "anc"]
        coords = [node_view[anc]["coords"] for anc in anc_qubits]

        rows, cols = zip(*coords)

        row_inds, num_rows = index_coords(rows, reverse=True)
        col_inds, num_cols = index_coords(cols)

        num_anc = len(anc_qubits)
        anc_inds = range(num_anc)

        tensor = np.zeros((num_anc, num_rows, num_cols), dtype=bool)
        tensor[anc_inds, row_inds, col_inds] = True
        expanded_tensor = np.expand_dims(tensor, axis=1)

        expansion_tensor = DataArray(
            expanded_tensor,
            dims=["anc_qubit", "channel", "row", "col"],
            coords=dict(
                anc_qubit=anc_qubits,
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

        anc_qubits = self.get_qubits(role="anc", stab_type=stab_type)
        data_qubits = self.get_qubits(role="data")

        proj_mat = adj_mat.sel(from_qubit=data_qubits, to_qubit=anc_qubits)
        return proj_mat.rename(from_qubit="data_qubit", to_qubit="anc_qubit")

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

        Raises
        ------
        ValueError
            If the specified file does not exist.
        ValueError
            If the specified file is not a string.
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

    def _load_layout(self, setup: dict[str, object]) -> None:
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

        for qubit_info in layout:
            qubit = qubit_info.pop("qubit", None)
            if qubit is None:
                raise ValueError("Each qubit in the layout must be labeled.")

            if qubit in self.graph:
                raise ValueError("Qubit label repeated, ensure labels are unique.")

            self._qubit_inds[qubit] = qubit_info.get("ind", None)

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


def index_coords(coords: list[int], reverse: bool = False) -> tuple[list[int], int]:
    """Indexes a list of coordinates.

    Parameters
    ----------
    coords
        The list of coordinates.
    reverse
        Whether to return the values in reverse, by default False

    Returns
    -------
    tuple[list[int], int]
        The list of indexed coordinates and the number of unique coordinates.
    """
    unique_vals = set(coords)
    num_unique_vals = len(unique_vals)

    if reverse:
        unique_inds = reversed(range(num_unique_vals))
    else:
        unique_inds = range(num_unique_vals)

    mapping = dict(zip(unique_vals, unique_inds))

    indicies = [mapping[coord] for coord in coords]
    return indicies, num_unique_vals
