import os
from os.path import join
import numpy as np
import matplotlib
# matplotlib.use("AGG")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Ellipse
import LFPy
import neuron
from neuron import h
import ECSbook_simcode.neural_simulations as ns
from ECSbook_simcode.plotting_convention import mark_subplots, simplify_axes
from ECSbook_simcode.neural_simulations import return_hay_cell

neuron.load_mechanisms(ns.cell_models_folder)

np.random.seed(12345)

tstop = 15
dt = 2**-4

def insert_synaptic_input(cell, synidx):

    synapse_parameters = dict(
                          idx = synidx,
                          e = 0., # reversal potential
                          weight = 0.01, # synapse weight
                          record_current = True, # record synapse current
                          syntype = 'Exp2Syn',
                          tau1 = 0.1, #Time constant, rise
                          tau2 = 1.0, #Time constant, decay
                          )
    synapse = LFPy.Synapse(cell, **synapse_parameters)
    synapse.set_spike_times(np.array([3.]))
    return synapse, cell


def return_electrode_grid():
    xmin, xmax = [-250, 250]
    zmin, zmax = [-150, 1100]

    dx = 100
    dz = 100
    x_grid, z_grid = np.mgrid[xmin:xmax+dx:dx, zmin:zmax+dz:dz]
    num_elecs = len(x_grid.flatten())
    elec_grid_params = dict(
                sigma = 0.3,      # extracellular conductivity
                x = x_grid.flatten(),
                y = np.zeros(num_elecs),
                z = z_grid.flatten(),
                method = 'pointsource',
            )
    return elec_grid_params


def hay_LFP():

    cell = return_hay_cell(tstop=tstop, dt=dt, make_passive=False)

    synidx = cell.get_closest_idx(x=0, y=0, z=500)

    syn, cell = insert_synaptic_input(cell, synidx)
    remove_list = ["Nap_Et2", "NaTa_t", "NaTs2_t", "SKv3_1",
        "SK_E2", "K_Tst", "K_Pst",
        "Im", "Ih", "CaDynamics_E2", "Ca_LVAst", "Ca", "Ca_HVA"]
    cell = ns.remove_active_mechanisms(remove_list, cell)
    h.dt = dt
    for sec in neuron.h.allsec():
        if "soma" in sec.name():
            print("g_pas: {}, e_pas: {}, cm: {}, "
                  "Ra: {}, soma_diam: {}, soma_L: {}".format(sec.g_pas,
                                                             sec.e_pas, sec.cm,
                                                             sec.Ra, sec.diam,
                                                             sec.L))

    cell.simulate(rec_imem=True, rec_vmem=True)

    plot_results(cell, "fig_hay_LFP", "reconstructed neuron", "B")
    cell.__del__()


def ball_and_stick_LFP(somatic_vmem):

    cell = ns.return_ball_and_stick_cell(tstop, dt)
    for sec in neuron.h.allsec():
        # Insert same passive params as Hay model
        sec.g_pas = 3.38e-05
        sec.e_pas = -90
        sec.cm = 1.0
        sec.Ra = 100
    h.dt = dt

    for sec in neuron.h.allsec():
        if "soma" in sec.name():
            print("Inserting vclamp")
            vclamp = h.SEClamp_i(sec(0.5))
            vclamp.dur1 = 1e9
            vclamp.rs = 1e-9
            vmem_to_insert = h.Vector(somatic_vmem)
            vmem_to_insert.play(vclamp._ref_amp1, h.dt)

    cell.simulate(rec_imem=True, rec_vmem=True)
    plot_results(cell, "fig_ball_and_stick_LFP", "ball and stick", "C")
    cell.__del__()


def two_comp_LFP():

    cell = ns.return_two_comp_cell(tstop, dt)

    synidx = cell.get_closest_idx(x=0, y=0, z=800)
    syn, cell = insert_synaptic_input(cell, synidx)

    h.dt = dt
    for sec in neuron.h.allsec():
        # Insert same passive params as Hay model
        sec.g_pas = 3.38e-05
        sec.e_pas = -90
        sec.cm = 1.0
        sec.Ra = 100

    cell.simulate(rec_imem=True, rec_vmem=True)
    cell.z = np.array([[-10., 10.],
                      [790., 810.]])

    plot_results(cell, "fig_two_comp_LFP", "two-compartment", "D")
    cell.__del__()


def plot_results(cell, figname, figtitle, subplot_marker):

    elec_grid_params = return_electrode_grid()

    elec = LFPy.RecExtElectrode(cell, **elec_grid_params)
    M_elec = elec.get_transformation_matrix()
    eaps = M_elec @ cell.imem * 1000

    xmin = np.min(elec_grid_params["x"])
    xmax = np.max(elec_grid_params["x"])
    zmin = np.min(elec_grid_params["z"])
    zmax = np.max(elec_grid_params["z"])

    eap_idxs = np.where((np.abs(elec_grid_params["z"] - 850) < 1e-9) &
                        (elec_grid_params["x"] > 0))[0]


    eap_clrs = {idx: plt.cm.Reds_r(num / (len(eap_idxs)))
                for num, idx in enumerate(eap_idxs)}

    fig = plt.figure(figsize=[5, 6.5])

    # fig.suptitle(figtitle)
    ax_morph = fig.add_axes([0.01, 0.01, 0.65, 0.93], frameon=False, aspect=1,
                            xticks=[], yticks=[], xlim=[xmin - 25, xmax + 100],
                            ylim=[zmin-10, zmax + 5])

    ax_imem = fig.add_axes([0.67, 0.55, 0.3, 0.25], title="normalized\ntransmembrane\ncurrents",
                          xticks=[], yticks=[], frameon=False,)

    ax_eap = fig.add_axes([0.67, 0.15, 0.3, 0.25], title="normalized\nLFP traces",
                          xticks=[], yticks=[], frameon=False,
                          ylim=[-1.05, 0.1])

    ax_imem.plot(cell.tvec, cell.imem[0, :] /
                 np.max(np.abs(cell.imem[0, :])), 'blue', lw=2)
    ax_imem.plot(cell.tvec, cell.imem[cell.synidx[0], :] /
                 np.max(np.abs(cell.imem[cell.synidx[0], :])), 'green', lw=2)


    ax_imem.text(5, 0.5, "bottom\ncomp.", ha="left", c="b")
    ax_imem.text(5, -0.7, "top\ncomp.", ha="left", c="g")


    for n, elec_idx in enumerate(eap_idxs[::-1]):
        c = eap_clrs[elec_idx]
        eap_norm = eaps[elec_idx] / np.max(np.abs(eaps[elec_idx]))
        ls = '-'# if n == (len(eap_idxs) - 1) else '-'
        ax_eap.plot(cell.tvec, eap_norm, c=c, lw=2, ls=ls)
        x = int(elec_grid_params["x"][elec_idx])
        fig.text(0.80, 0.2 + n * 0.04, "x={:d} µm".format(x), c=c)

    if "two_comp" in figname:
        ax_morph.plot(cell.x[0].mean(), cell.z[0].mean(), 'bo', ms=12)
        ax_morph.plot(cell.x[0].mean(), cell.z[0].mean(), 'w+', ms=12)
        ax_morph.plot(cell.x[1].mean(), cell.z[1].mean(), 'go', ms=12)
        ax_morph.plot(cell.x[1].mean(), cell.z[1].mean(), 'w_', ms=12)


    else:
        zips = []
        for x, z in cell.get_pt3d_polygons():
            zips.append(list(zip(x, z)))
        polycol = PolyCollection(zips, edgecolors='none',
                                 facecolors='0.4', zorder=-1, rasterized=False)
        ax_morph.add_collection(polycol)

        l_syn, = ax_morph.plot(cell.x[cell.synidx].mean(axis=1),
                      cell.z[cell.synidx].mean(axis=1), c='y', marker='*', ls='none')
        fig.legend([l_syn], ["synapse"], loc=(0.63, 0.92),
               frameon=False, handlelength=0.5)
    t1 = 10
    t1_idx = np.argmin(np.abs(cell.tvec - t1))
    dz = np.abs(np.diff(elec.z))[0]
    num_elecs = len(elec.x)
    eap_norm = dz * 0.7 / np.max(np.abs(eaps))
    t_norm = cell.tvec[:t1_idx] / t1 * dz * 0.7
    for elec_idx in range(num_elecs):
        c = eap_clrs[elec_idx] if elec_idx in eap_idxs else 'k'
        x, z = elec.x[elec_idx], elec.z[elec_idx]
        ax_morph.plot(x, z, '.', c='k', ms=3)
        eap = eaps[elec_idx, :t1_idx] * eap_norm
        ax_morph.plot(x + t_norm, z + eap, c=c, lw=2)

    ax_morph.plot([150, 250], [-15, -15], c='gray', lw=2)
    ax_morph.text(200, 0, "100 µm", ha="center", c='gray')

    ax_morph.plot([150, 150 + t_norm[-1]], [775, 775], c='k', lw=2)
    ax_morph.text(157, 777, "{:d} ms".format(int(t1)), ha="center",
                  va="bottom", c='k')

    ax_morph.plot([130, 130], [720 - 1 * eap_norm, 720], c='k', lw=2,
                  clip_on=False)
    ax_morph.text(135, 720 - 0.5 * eap_norm, "1 µV", ha="left",
                  c='k', va="center")


    ax_imem.plot([0, 15], [-1.1, -1.1], c='k', lw=2, clip_on=False)
    ax_imem.text(7, -1.15, "15 ms", va='top', ha='center')

    ax_eap.plot([0, 15], [-1.1, -1.1], c='k', lw=2, clip_on=False)
    ax_eap.text(7, -1.15, "15 ms", va='top', ha='center')

    mark_subplots(ax_morph, "A", xpos=0.05, ypos=1.03)
    mark_subplots(ax_imem, "B", xpos=0.05, ypos=1.38)
    mark_subplots(ax_eap, "C", xpos=0.05, ypos=1.2)



    fig.savefig(join(os.path.dirname(__file__), "{}.png".format(figname)),
                dpi=300)


if __name__ == '__main__':
    # save_somatic_spike_vmem()
    # soma_t, soma_vmem = np.load("somatic_vmem.npy")
    # hay_LFP()
    # ball_and_stick_spike_replay(soma_vmem)
    two_comp_LFP()


