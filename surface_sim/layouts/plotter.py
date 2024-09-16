"""Layout plotting module."""

import re
from typing import Sequence, Tuple, Reversible, Iterable
from copy import deepcopy

import numpy as np

from matplotlib.axes import Axes
from matplotlib.patches import Circle, Polygon
from matplotlib.lines import Line2D
from matplotlib.text import Text

from .layout import Layout

Coordinates = Tuple[float, float]
CoordRange = Tuple[float, float]

RE_FILTER = re.compile("([a-zA-Z]+)([0-9]+)")  # Regex to filter qubit labels

ZORDERS = dict(circle=3, patch=1, line=2, text=4)


def clockwise_sort(coordinates: Sequence[Coordinates]) -> Sequence[Coordinates]:
    """
    clockwise_sort Sorts a sequence of coordinates in clockwise order.

    Parameters
    ----------
    coordinates : Sequence[Coordinates]
        The coordinates to sort.

    Returns
    -------
    Sequence[Coordinates]
        The sorted coordinates.
    """
    coords = list(coordinates)
    x_coords, y_coords = zip(*coords)

    x_center = np.mean(x_coords)
    y_center = np.mean(y_coords)

    x_vectors = [x - x_center for x in x_coords]
    y_vectors = [y - y_center for y in y_coords]

    angles = np.arctan2(x_vectors, y_vectors)
    inds = np.argsort(angles)
    sorted_coords = [coords[ind] for ind in inds]
    return sorted_coords


def invert(sequence: Reversible) -> Tuple:
    """
    invert Inverts a sequence.

    Parameters
    ----------
    sequence : Sequence
        The sequence to invert.

    Returns
    -------
    Tuple
        The inverted sequence.
    """
    return tuple(reversed(sequence))


def get_label(qubit: str, coords: Coordinates, **kwargs) -> Text:
    """
    label_qubit Labels a qubit.

    Parameters
    ----------
    axis : Axes
        The axis to label the qubit on.
    qubit : str
        The qubit label.
    coords : Tuple[float, float]
        The coordinates of the qubit.

    Raises
    ------
    ValueError
        If the qubit label is not in the expected format.
    """
    match = RE_FILTER.match(str(qubit))
    if match is None:
        raise ValueError(f"Unexpected qubit label {qubit}")
    name, ind = match.groups()
    text = f"${name}_\\mathrm{{{ind}}}$"

    x, y = coords
    zorder = ZORDERS["text"]
    label = Text(x, y, text, zorder=zorder, **kwargs)
    return label


def get_circle(center: Coordinates, radius: float, **kwargs) -> Circle:
    """
    get_circle Returns a circle.

    Parameters
    ----------
    coords : Tuple[float, float]
        The coordinates of the centre of the circle.
    radius : float
        The radius of the circle.

    Returns
    -------
    Circle
        The circle.
    """
    zorder = ZORDERS["circle"]
    circle = Circle(center, radius=radius, zorder=zorder, **kwargs)
    return circle


def get_patch(patch_coords: Sequence[Coordinates], **kwargs) -> Polygon:
    """
    draw_patch Draws a patch. Matplotlib parameters can be passed as kwargs.

    Parameters
    ----------
    axis : Axes
        The axis to draw the patch on.
    patch_coords : Sequence[Tuple[float, float]]
        The coordinates of the patch.
    """
    zorder = ZORDERS["patch"]
    patch = Polygon(patch_coords, closed=True, zorder=zorder, **kwargs)
    return patch


def get_line(coordinates: Sequence[Coordinates], **kwargs) -> Line2D:
    """
    draw_connection Draws a connection between two qubits. Matplotlib parameters can be passed as kwargs.

    Parameters
    ----------
    axis : Axes
        The axis to draw the connection on.
    qubit_coords : Iterable[Tuple[float, float]]
        The coordinates of the qubits.
    """
    x_coords, y_coords = zip(*coordinates)
    zorder = ZORDERS["line"]
    line = Line2D(x_coords, y_coords, zorder=zorder, **kwargs)
    return line


def qubit_labels(layout: Layout) -> Iterable[Text]:
    qubits = layout.get_qubits()

    for qubit in qubits:
        coords = invert(layout.param("coords", qubit))

        metaparams = layout.param("metaparams", qubit)
        text_params = metaparams.get("text")

        if text_params:
            yield get_label(qubit, coords, **text_params)
        else:
            yield get_label(qubit, coords)


def qubit_connections(layout: Layout) -> Iterable[Line2D]:
    anc_qubits = layout.get_qubits(role="anc")

    for anc_qubit in anc_qubits:
        anc_coords = invert(layout.param("coords", anc_qubit))

        metaparams = layout.param("metaparams", anc_qubit)
        line_params = None
        if metaparams is not None:
            line_params = metaparams.get("line")

        neighbors = layout.get_neighbors(anc_qubit)
        for nbr in neighbors:
            nbr_coords = invert(layout.param("coords", nbr))
            line_coords = (anc_coords, nbr_coords)

            if line_params:
                yield get_line(line_coords, **line_params)
            else:
                yield get_line(line_coords)


def qubit_artists(layout: Layout) -> Iterable[Circle]:
    """
    draw_qubits Draws the qubits of a layout.

    Parameters
    ----------
    axis : Axes
        The axis to draw the qubits on.
    layout : Layout
        The layout to draw the qubits of.
    """
    qubits = layout.get_qubits()
    default_radius = 0.3

    for qubit in qubits:
        coords = invert(layout.param("coords", qubit))

        metaparams = layout.param("metaparams", qubit)
        circle_params = None
        if metaparams is not None:
            circle_params = metaparams.get("circle")

        if circle_params is not None:
            params_copy = deepcopy(circle_params)
            radius = params_copy.pop("radius", default_radious)
            yield get_circle(coords, radius, **params_copy)
        else:
            yield get_circle(coords, default_radius)


def patch_artists(layout: Layout) -> Iterable[Polygon]:
    """
    draw_patches Draws the stabilizer patches of a layout.

    Parameters
    ----------
    axis : Axes
        The axis to draw the patches on.
    layout : Layout
        The layout to draw the patches of.
    """
    anc_qubits = layout.get_qubits(role="anc")
    for anc_qubit in anc_qubits:
        anc_coords = invert(layout.param("coords", anc_qubit))

        metaparams = layout.param("metaparams", anc_qubit)
        patch_params = metaparams.get("patch")

        neigbors = layout.get_neighbors(anc_qubit)

        coords = [invert(layout.param("coords", nbr)) for nbr in neigbors]

        num_neigbors = len(neigbors)
        if num_neigbors == 2:
            coords.append(anc_coords)

        patch_coords = clockwise_sort(coords)
        if patch_params:
            yield get_patch(patch_coords, **patch_params)
        else:
            yield get_patch(patch_coords)


def get_coord_range(layout: Layout) -> Tuple[CoordRange, CoordRange]:
    qubits = layout.get_qubits()

    inv_coords = (layout.param("coords", qubit) for qubit in qubits)
    coords = map(invert, inv_coords)
    x_coords, y_coords = zip(*coords)

    x_range: CoordRange = (min(x_coords), max(x_coords))
    y_range: CoordRange = (min(y_coords), max(y_coords))
    return x_range, y_range


def plot(
    ax: Axes,
    layout: Layout,
    add_labels: bool = True,
    add_patches: bool = True,
    add_connections: bool = True,
    pad: float = 1,
) -> Axes:
    """
    Plots a layout.

    Parameters
    ----------
    ax
        The axis to plot the layout on.
    layout
        The layout to plot.
    add_labels
        Flag to add qubit labels, by default True.
    add_patches
        Flag to plot stabilizer patches, by default True.
    add_connections
        Flag to plot lines indicating the connectivity, by default True.
    pad
        The padding to the bottom axis, by default 1.

    Returns
    -------
    ax
        The figure the layout was plotted on.
    """
    for artist in qubit_artists(layout):
        ax.add_artist(artist)

    if add_patches:
        for artist in patch_artists(layout):
            ax.add_artist(artist)

    if add_connections:
        for artist in qubit_connections(layout):
            ax.add_artist(artist)

    if add_labels:
        for artist in qubit_labels(layout):
            ax.add_artist(artist)

    x_range, y_range = get_coord_range(layout)

    x_min, x_max = x_range
    ax.set_xlim(x_min - pad, x_max + pad)

    y_min, y_max = y_range
    ax.set_ylim(y_min - pad, y_max + pad)

    return ax
