
import numpy as np
import matplotlib
# matplotlib.use("AGG")
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
                          weight = 0.07, # synapse weight
                          record_current = True, # record synapse current
                          syntype = 'Exp2Syn',
                          tau1 = 0.1, #Time constant, rise
                          tau2 = 1.0, #Time constant, decay
                          )
    synapse = LFPy.Synapse(cell, **synapse_parameters)
    synapse.set_spike_times(np.array([1.]))
    return synapse, cell


def make_figure():

    tstop = 12
    dt = 2**-5
    cell = return_hay_cell(tstop=tstop, dt=dt, make_passive=False)
    syn, cell = insert_synaptic_input(cell)
    cell.simulate(rec_imem=True, rec_vmem=True)

    xmin, xmax = [-50, 70]
    zmin, zmax = [-50, 130]

    dx = 20
    dz = 20
    x_grid, z_grid = np.mgrid[xmin:xmax+dx:dx, zmin:zmax+dz:dz]
    num_elecs = len(x_grid.flatten())
    elec_grid_params = dict(
                sigma = 0.3,      # extracellular conductivity
                x = x_grid.flatten(),
                y = np.zeros(num_elecs),
                z = z_grid.flatten(),
                method = 'root_as_point',
            )

    eap_idxs = np.where((np.abs(elec_grid_params["z"] - 10) < 1e-9) & (elec_grid_params["x"] > 0))[0]

    eap_clrs = {idx: plt.cm.Reds_r(num / (len(eap_idxs))) for num, idx in enumerate(eap_idxs)}

    print(elec_grid_params["x"][eap_idxs])
    print(elec_grid_params["z"][eap_idxs])

    elec = LFPy.RecExtElectrode(cell, **elec_grid_params)
    M_elec = elec.get_transformation_matrix()
    eaps = M_elec @ cell.imem * 1000

    plt.close("all")
    fig = plt.figure(figsize=[7, 6])
    fig.subplots_adjust(left=0.14, bottom=0.18, top=0.85, right=0.98,
                        hspace=0.5)

    ax_morph = fig.add_axes([0.01, 0.01, 0.65, 0.98], frameon=False, aspect=1,
                            xticks=[], yticks=[], xlim=[xmin - 5, xmax + 10],
                            ylim=[zmin - 10, zmax + 5])

    ax_vm = fig.add_axes([0.72, 0.60, 0.27, 0.3], title="membrane\npotential",
                          frameon=False, xticks=[])

    ax_eap = fig.add_axes([0.72, 0.1, 0.27, 0.3], title="normalized spikes", frameon=False, xticks=[], yticks=[])

    for n, elec_idx in enumerate(eap_idxs[::-1]):
        c = eap_clrs[elec_idx]
        eap_norm = eaps[elec_idx] / np.max(np.abs(eaps[elec_idx]))
        ax_eap.plot(cell.tvec, eap_norm, c=c, lw=2)
        x = int(elec_grid_params["x"][elec_idx])
        ax_eap.text(6.5, -0.7 + n * 0.15, "x={:d} µm".format(x), c=c)
    zips = []
    for x, z in cell.get_pt3d_polygons():
        zips.append(list(zip(x, z)))
    polycol = PolyCollection(zips, edgecolors='none',
                             facecolors='0.8', zorder=-1, rasterized=False)
    ax_morph.add_collection(polycol)

    lines = []
    line_names = []

    print(np.max(np.abs(eaps)))
    eap_norm = dz * 0.9 / np.max(np.abs(eaps))
    t_norm = cell.tvec / cell.tvec[-1] * dz * 0.7
    for elec_idx in range(num_elecs):
        c = eap_clrs[elec_idx] if elec_idx in eap_idxs else 'k'
        x, z = elec.x[elec_idx], elec.z[elec_idx]
        ax_morph.plot(x, z, '.', c='k', ms=3)
        eap = eaps[elec_idx] * eap_norm
        ax_morph.plot(x + t_norm, z + eap, c=c, lw=2)

    l, = ax_vm.plot(cell.tvec, cell.vmem[0, :], c='k', lw=2)
    ax_vm.plot([6, 6], [-50, 0], c='k', lw=2)
    ax_vm.text(6.5, -25, "50 mV")

    v0 = int(cell.vmem[0, 0])

    ax_morph.plot([20, 40], [15, 15], c='gray', lw=2)
    ax_morph.text(30, 17, "20 µm", ha="center", c='gray')

    ax_morph.plot([82, 82], [-10 - 500 * eap_norm, -10], c='k', lw=2,
                  clip_on=False)
    ax_morph.text(79, -10 - 500 * eap_norm / 2, "500 µV", ha="right",
                  c='k', va="center")

    ax_vm.plot([6, 7], [v0, v0], c='k', lw=2)
    ax_vm.text(6.5, v0 - 1, "1 ms", va="top", ha='center')

    ax_eap.plot([6, 7], [-1, -1], c='k', lw=2)
    ax_eap.text(6.5, - 1, "1 ms", va="top", ha='center')

    ax_vm.set_yticks([v0])
    ax_vm.set_yticklabels(["{:d} mV".format(v0)])

    fig.legend(lines, line_names, loc="lower center", frameon=False, ncol=2)
    mark_subplots(fig.axes[0], 'A', ypos=0.98, xpos=0.0)
    mark_subplots(fig.axes[1:], 'BCDE')
    simplify_axes(fig.axes)

    fig.savefig("fig_hay_eap_simpler.png", dpi=300)
    fig.savefig("fig_hay_eap_simpler.pdf", dpi=300)


if __name__ == '__main__':
    make_figure()

