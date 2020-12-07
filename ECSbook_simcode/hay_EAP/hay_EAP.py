
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
                          weight = 0.1, # 0.001, # synapse weight
                          record_current = True, # record synapse current
                          syntype = 'Exp2Syn',
                          tau1 = 0.1, #Time constant, rise
                          tau2 = 1, #Time constant, decay
                          )
    synapse = LFPy.Synapse(cell, **synapse_parameters)
    synapse.set_spike_times(np.array([1.]))
    return synapse, cell


def return_width_at_half_max(eaps_lat, dt):
    widths = np.zeros(len(eaps_lat))
    for idx, eap in enumerate(eaps_lat):
        half_max = np.min(eap) / 2
        idxs = np.where(eap < half_max)[0]
        # plt.plot(eap)
        # print(idxs)
        # plt.plot(idxs, np.ones(len(idxs)) * half_max)
        # plt.show()
        widths[idx] = dt * len(idxs)
    return widths


def make_figure():

    tstop = 10
    dt = 2**-5
    cell = return_hay_cell(tstop=tstop, dt=dt, make_passive=False)
    syn, cell = insert_synaptic_input(cell)
    cell.simulate(rec_imem=True, rec_vmem=True)

    xmin, xmax = [-70, 70]
    zmin, zmax = [-50, 200]

    dx = 20
    dz = 20
    x_grid, z_grid = np.mgrid[xmin:xmax+dx:dx, zmin:zmax+dz:dz]
    num_elecs = len(x_grid.flatten())
    elec_grid_params = dict(
                sigma = 0.3,      # extracellular conductivity
                x = x_grid.flatten(),
                y = np.zeros(num_elecs),
                z = z_grid.flatten(),
                method = 'soma_as_point',
            )

    elec_lateral_params = dict(
                sigma = 0.3,      # extracellular conductivity
                x = np.linspace(10, 200, 15),
                y = np.zeros(15),
                z = np.zeros(15),
                method = 'soma_as_point',
            )


    elec = LFPy.RecExtElectrode(cell, **elec_grid_params)
    elec_lat = LFPy.RecExtElectrode(cell, **elec_lateral_params)
    elec.calc_lfp()
    elec_lat.calc_lfp()
    eaps = elec.LFP * 1000
    eaps_lat = elec_lat.LFP * 1000
    eap_amps = np.max(eaps_lat, axis=1) - np.min(eaps_lat, axis=1)

    widths = return_width_at_half_max(eaps_lat, dt)

    plt.close("all")
    fig = plt.figure(figsize=[5, 6])
    fig.subplots_adjust(left=0.14, bottom=0.18, top=0.85, right=0.98,
                        hspace=0.5)

    ax_morph = fig.add_axes([0.0, 0.05, 0.6, 0.95], frameon=False, aspect=1,
                            xticks=[], yticks=[], xlim=[xmin - 20, xmax + 20],
                            ylim=[zmin - 20, zmax + 50])

    ax_vm = fig.add_axes([0.8, 0.83, 0.15, 0.12],
                          xlabel="time (ms)", ylabel="membrane\npotential (mV)")

    ax_eap = fig.add_axes([0.8, 0.6, 0.15, 0.12],
                          xlabel="time (ms)", ylabel="EAP (µV)")

    ax_decay = fig.add_axes([0.8, 0.35, 0.15, 0.12], xscale="log", yscale="log",
                          xlabel="distance (µm)", ylabel="amplitude (µV)", xlim=[5, 200],
                            )

    ax_width = fig.add_axes([0.8, 0.1, 0.15, 0.12], xlim=[0, 200], ylim=[0, 1],
                          xlabel="distance (µm)", ylabel="spike\nwidth (ms)")

    ax_eap.plot(cell.tvec, eaps_lat[0], 'k')
    ax_decay.loglog(elec_lat.x, eap_amps, 'k')

    ax_width.plot(elec_lat.x, widths, 'k')
    zips = []
    for x, z in cell.get_pt3d_polygons():
        zips.append(list(zip(x, z)))
    polycol = PolyCollection(zips, edgecolors='none',
                             facecolors='0.8', zorder=-1, rasterized=False)
    ax_morph.add_collection(polycol)

    lines = []
    line_names = []

    eap_norm = np.max(np.abs(eaps))
    t_norm = cell.tvec / cell.tvec[-1] * dz * 0.7
    for elec_idx in range(num_elecs):
        x, z = elec.x[elec_idx], elec.z[elec_idx]
        ax_morph.plot(x, z, '.', c='k', ms=3)
        eap = eaps[elec_idx] / eap_norm * dz * 0.9
        ax_morph.plot(x + t_norm, z + eap, 'k')

    l, = ax_vm.plot(cell.tvec, cell.vmem[0, :], c='k')

    fig.legend(lines, line_names, loc="lower center", frameon=False, ncol=2)
    mark_subplots(fig.axes[0], 'A', ypos=0.95, xpos=0.1)
    mark_subplots(fig.axes[1:], 'BCDE')
    simplify_axes(fig.axes)

    ax_decay.set_xticks([10, 100])
    ax_decay.set_yticks([1, 10, 100, 1000])
    ax_decay.grid(True)
    fig.savefig("fig_hay_eap.png", dpi=300)
    # fig.savefig("fig_elec_size_effect.pdf", dpi=300)


if __name__ == '__main__':
    make_figure()

