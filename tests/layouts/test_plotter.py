import matplotlib.pyplot as plt

from surface_sim.layouts import rot_surf_code
from surface_sim.layouts import plot


def test_plot(show_figures):
    _, ax = plt.subplots()
    layout = rot_surf_code(3)

    plot(ax, layout)

    if show_figures:
        plt.show()
    plt.close()

    return


def test_plot_large_code(show_figures):
    _, ax = plt.subplots()
    layout = rot_surf_code(7)

    plot(ax, layout, label_fontsize=5)

    if show_figures:
        plt.show()
    plt.close()

    return
