import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from ECSbook_simcode import plotting_convention
from matplotlib.ticker import FormatStrFormatter


def make_figure():
    fig = plt.figure(figsize=[8, 5])
    ax = fig.add_subplot(111, xlim=[0.7e-5, 704800], xlabel="time scale",
                         ylim=[0.7e-4, 1.5e3], ylabel="spatial scale",
                         )
    fig.subplots_adjust(left=0.25, bottom=0.25, right=0.97)


    plt.xscale("log")
    plt.yscale("log")
    x_minor = matplotlib.ticker.LogLocator(base = 10.0,
                                           subs = np.arange(1.0, 10.0) * 0.1,
                                           numticks = 12)
    ax.xaxis.set_minor_locator(x_minor)

    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax.set_xticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 60, 3600, 86400, 604800])
    ax.set_xticklabels(["10 µs", "0.1 ms", "1 ms", "10 ms", "0.1 s", "1 s", "10 s", "1 m", "1 h", "1 d", "1 w"])

    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, ])
    ax.set_yticklabels(["0.1 µm", "1 µm", "10 µm", "0.1 mm", "1 mm", "1 cm", "10 cm", "1 m"])

    plotting_convention.simplify_axes(ax)
    ax.grid(True)

    plt.savefig("measurement_scales_in_neuroscience.pdf")


if __name__ == '__main__':

    make_figure()