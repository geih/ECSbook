
import numpy as np
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Ellipse
import LFPy
from ECSbook_simcode.plotting_convention import mark_subplots, simplify_axes
from ECSbook_simcode.neural_simulations import return_hay_cell

np.random.seed(12345)

def insert_synaptic_input(cell):

    synapse_parameters = dict(
                          idx = 0,
                          e = 0., # reversal potential
                          weight = 0.1, # 0.001, # synapse weight
                          record_current = True, # record synapse current
                          syntype = 'Exp2Syn',
                          tau1 = 0.1, #Time constant, rise
                          tau2 = 1, #Time constant, decay
                          )
    synapse = LFPy.Synapse(cell, **synapse_parameters)
    synapse.set_spike_times(np.array([1.]))
    return synapse, cell

def make_figure():

    tstop = 10
    dt = 2**-5
    cell = return_hay_cell(tstop=tstop, dt=dt, make_passive=False)
    syn, cell = insert_synaptic_input(cell)
    cell.simulate(rec_imem=True, rec_vmem=True)

    elec_radii = np.array([1, 10, 25, 50])
    elec_clrs = {r: plt.cm.rainbow(i / (len(elec_radii) - 1))
                 for i, r in enumerate(elec_radii)}
    eaps = []

    elec_params = dict(
                sigma = 0.3,      # extracellular conductivity
                x = np.array([20]),
                y = np.array([0]),
                z = np.array([0]),
                method = 'soma_as_point',
            )

    for elec_radius in elec_radii:
        if elec_radius > 1e-9:
            elec_params.update(
                N = np.array([[1, 0, 0]]), # surface normals
                r = elec_radius,           # contact site radius
                n = elec_radius * 10,      # datapoints for averaging
            )

        elec = LFPy.RecExtElectrode(cell, **elec_params)
        elec.calc_lfp()
        eaps.append(elec.LFP[0] * 1000)

    plt.close("all")
    fig = plt.figure(figsize=[5, 6])
    fig.subplots_adjust(left=0.14, bottom=0.18, top=0.85, right=0.98,
                        hspace=0.5)

    ax_morph = fig.add_axes([0.0, 0.47, 1.0, 0.5], frameon=False, aspect=1,
                            xticks=[], yticks=[], xlim=[-150, 150],
                            ylim=[-100, 100])

    ax_eap = fig.add_axes([0.17, 0.22, 0.32, 0.2],
                          xlabel="time (ms)", ylabel="µV")
    ax_eap_norm = fig.add_axes([0.67, 0.22, 0.32, 0.2],
                               xlabel="time (ms)", ylabel="normalized")

    zips = []
    for x, z in cell.get_pt3d_polygons():
        zips.append(list(zip(x, z)))
    polycol = PolyCollection(zips, edgecolors='none',
                             facecolors='gray', zorder=100, rasterized=False)
    ax_morph.add_collection(polycol)

    lines = []
    line_names = []
    for i, elec_radius in enumerate(elec_radii):
        c = elec_clrs[elec_radius]

        if elec_radius > 1e-9:
            el = Ellipse((elec.x[0], elec.z[0]), elec_radius / 2, 2*elec_radius,
                     facecolor=c, clip_on=False, zorder=-i)
            ax_morph.add_artist(el)
        else:
            ax_morph.plot(elec.x[0], elec.z[0], '.', c=c, ms=3)

        l, = ax_eap.plot(cell.tvec, eaps[i], c=c)
        ax_eap_norm.plot(cell.tvec, eaps[i] / np.max(np.abs(eaps[i])), c=c)
        lines.append(l)
        line_names.append("r=%d µm" % elec_radius)

    fig.legend(lines, line_names, loc="lower center", frameon=False, ncol=2)
    mark_subplots(ax_morph, ypos=0.95, xpos=0.1)
    mark_subplots([ax_eap, ax_eap_norm], "BC")
    simplify_axes(fig.axes)
    fig.savefig("fig_elec_size_effect.png", dpi=300)
    fig.savefig("fig_elec_size_effect.pdf", dpi=300)


if __name__ == '__main__':
    make_figure()

