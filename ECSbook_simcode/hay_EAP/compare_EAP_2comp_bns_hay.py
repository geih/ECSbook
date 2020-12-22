import os
import numpy as np
import matplotlib
# matplotlib.use("AGG")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import LFPy
import neuron
from neuron import h
import ECSbook_simcode.neural_simulations as ns
from ECSbook_simcode.plotting_convention import mark_subplots, simplify_axes
from ECSbook_simcode.neural_simulations import return_hay_cell

neuron.load_mechanisms(ns.cell_models_folder)

np.random.seed(12345)

tstop = 8
dt = 2**-6

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


def return_electrode_grid():
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
                method = 'linesource',
            )
    return elec_grid_params


def plot_eap_grid(cell, eaps, elec, ax, eap_clrs, eap_idxs):

    zips = []
    for x, z in cell.get_pt3d_polygons():
        zips.append(list(zip(x, z)))
    polycol = PolyCollection(zips, edgecolors='none',
                             facecolors='0.8', zorder=-1, rasterized=False)
    ax.add_collection(polycol)

    dz = np.abs(np.diff(elec.z))[0]
    num_elecs = len(elec.x)
    eap_norm = dz * 0.9 / np.max(np.abs(eaps))
    t_norm = cell.tvec / cell.tvec[-1] * dz * 0.7
    for elec_idx in range(num_elecs):
        c = eap_clrs[elec_idx] if elec_idx in eap_idxs else 'k'
        x, z = elec.x[elec_idx], elec.z[elec_idx]
        ax.plot(x, z, '.', c='k', ms=3)
        eap = eaps[elec_idx] * eap_norm
        ax.plot(x + t_norm, z + eap, c=c, lw=2)

    ax.plot([20, 40], [15, 15], c='gray', lw=2)
    ax.text(30, 17, "20 µm", ha="center", c='gray')

    ax.plot([82, 82], [-10 - 500 * eap_norm, -10], c='k', lw=2,
                  clip_on=False)
    ax.text(79, -10 - 500 * eap_norm / 2, "500 µV", ha="right",
                  c='k', va="center")


def save_somatic_spike_vmem():

    cell = return_hay_cell(tstop=tstop, dt=dt, make_passive=False)
    syn, cell = insert_synaptic_input(cell)
    cell.simulate(rec_imem=True, rec_vmem=True)
    np.save("somatic_vmem.npy", [cell.tvec, cell.somav])
    np.save("imem_orig.npy", cell.imem)
    np.save("vmem_orig.npy", cell.vmem)
    fig = plt.figure(figsize=[2., 2.])
    fig.subplots_adjust(left=0.35, top=0.75, right=0.85)

    ax_vm = fig.add_subplot(111, title="membrane\npotential",
                           frameon=False, xticks=[])
    l, = ax_vm.plot(cell.tvec, cell.vmem[0, :], c='k', lw=2)
    ax_vm.plot([5, 5], [-50, 0], c='k', lw=2)
    ax_vm.text(5.5, -25, "50 mV")
    mark_subplots(ax_vm, "A")

    v0 = int(cell.vmem[0, 0])

    ax_vm.plot([6, 7], [v0, v0], c='k', lw=2)
    ax_vm.text(6.5, v0 - 1, "1 ms", va="top", ha='center')
    ax_vm.set_yticks([v0])
    ax_vm.set_yticklabels(["{:d} mV".format(v0)])
    plt.savefig("somatic_vmem.png", dpi=300)
    plt.close("all")

    plot_results(cell, "fig_hay_orig_spike", "original active", 'A')
    cell.__del__()


def hay_spike_replay(somatic_vmem):

    cell = return_hay_cell(tstop=tstop, dt=dt, make_passive=False)
    remove_list = ["Nap_Et2", "NaTa_t", "NaTs2_t", "SKv3_1",
        "SK_E2", "K_Tst", "K_Pst",
        "Im", "Ih", "CaDynamics_E2", "Ca_LVAst", "Ca", "Ca_HVA"]
    #cell = ns.remove_active_mechanisms(remove_list, cell)
    # syn, cell = insert_synaptic_input(cell)

    h.dt = dt

    for sec in neuron.h.allsec():
        if "soma" in sec.name():
            print("g_pas: {}, e_pas: {}, cm: {}, "
                  "Ra: {}, soma_diam: {}, soma_L: {}".format(sec.g_pas,
                                                             sec.e_pas, sec.cm,
                                                             sec.Ra, sec.diam,
                                                             sec.L))
            print("Inserting vclamp")
            vclamp = h.SEClamp_i(sec(0.5))
            vclamp.dur1 = 1e9
            vclamp.rs = 1e-9
            vmem_to_insert = h.Vector(somatic_vmem[1:])
            vmem_to_insert.play(vclamp._ref_amp1, h.dt)

    cell.simulate(rec_imem=True, rec_vmem=True)

    np.save("imem_replay_active.npy", cell.imem)
    np.save("vmem_replay_active.npy", cell.vmem)
    plot_results(cell, "fig_hay_replay_spike_active", "reconstructed neuron", "B")
    cell.__del__()


def ball_and_stick_spike_replay(somatic_vmem):

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
    plot_results(cell, "fig_ball_and_stick_replay_spike", "ball and stick", "C")
    cell.__del__()


def two_comp_spike_replay(somatic_vmem):

    cell = ns.return_two_comp_cell(tstop, dt)
    h.dt = dt
    for sec in neuron.h.allsec():
        # Insert same passive params as Hay model
        sec.g_pas = 3.38e-05
        sec.e_pas = -90
        sec.cm = 1.0
        sec.Ra = 100
    for sec in neuron.h.allsec():
        if "soma" in sec.name():
            print("Inserting vclamp")
            vclamp = h.SEClamp_i(sec(0.5))
            vclamp.dur1 = 1e9
            vclamp.rs = 1e-9
            vmem_to_insert = h.Vector(somatic_vmem)
            vmem_to_insert.play(vclamp._ref_amp1, h.dt)

    cell.simulate(rec_imem=True, rec_vmem=True)

    plot_results(cell, "fig_two_comp_replay_spike", "two-compartment", "D")
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

    eap_idxs = np.where((np.abs(elec_grid_params["z"] + 10) < 1e-9) &
                        (elec_grid_params["x"] > 0))[0]

    eap_clrs = {idx: plt.cm.Reds_r(num / (len(eap_idxs)))
                for num, idx in enumerate(eap_idxs)}

    fig = plt.figure(figsize=[3, 5.])

    fig.suptitle(figtitle)
    ax_morph = fig.add_axes([0.05, 0.22, 0.9, 0.68], frameon=False, aspect=1,
                            xticks=[], yticks=[], xlim=[xmin - 5, xmax + 10],
                            ylim=[zmin - 10, zmax + 5])

    # ax_vm = fig.add_axes([0.72, 0.56, 0.24, 0.3], title="membrane\npotential",
    #                       frameon=False, xticks=[])

    ax_eap = fig.add_axes([0.1, 0.01, 0.85, 0.17], #title="normalized\nspikes",
                          frameon=False, xticks=[], yticks=[],
                          ylim=[-1.05, 0.5])

    for n, elec_idx in enumerate(eap_idxs[::-1]):
        c = eap_clrs[elec_idx]
        eap_norm = eaps[elec_idx] / np.max(np.abs(eaps[elec_idx]))
        ls = '-'# if n == (len(eap_idxs) - 1) else '-'
        ax_eap.plot(cell.tvec, eap_norm, c=c, lw=2, ls=ls)
        x = int(elec_grid_params["x"][elec_idx])
        ax_eap.text(5., -1.05 + n * 0.25, "x={:d} µm".format(x), c=c)

    zips = []
    for x, z in cell.get_pt3d_polygons():
        zips.append(list(zip(x, z)))
    polycol = PolyCollection(zips, edgecolors='none',
                             facecolors='0.8', zorder=-1, rasterized=False)
    ax_morph.add_collection(polycol)

    dz = np.abs(np.diff(elec.z))[0]
    num_elecs = len(elec.x)
    eap_norm = dz * 0.9 / np.max(np.abs(eaps))
    t_norm = cell.tvec / cell.tvec[-1] * dz * 0.7
    for elec_idx in range(num_elecs):
        c = eap_clrs[elec_idx] if elec_idx in eap_idxs else 'k'
        x, z = elec.x[elec_idx], elec.z[elec_idx]
        ax_morph.plot(x, z, '.', c='k', ms=3)
        eap = eaps[elec_idx] * eap_norm
        ax_morph.plot(x + t_norm, z + eap, c=c, lw=2)

    ax_morph.plot([20, 40], [15, 15], c='gray', lw=2)
    ax_morph.text(30, 17, "20 µm", ha="center", c='gray')

    ax_morph.plot([82, 82], [-10 - 500 * eap_norm, -10], c='k', lw=2,
                  clip_on=False)
    ax_morph.text(79, -20, "500 µV", ha="right",
                  c='k', va="center")

    mark_subplots(ax_morph, subplot_marker, xpos=-0.07, ypos=1.1)

    # l, = ax_vm.plot(cell.tvec, cell.vmem[0, :], c='k', lw=2)
    # ax_vm.plot([4, 4], [-50, 0], c='k', lw=2)
    # ax_vm.text(4.5, -25, "50 mV")
    #
    # v0 = int(cell.vmem[0, 0])
    #
    # ax_vm.plot([6, 7], [v0, v0], c='k', lw=2)
    # ax_vm.text(6.5, v0 - 1, "1 ms", va="top", ha='center')
    #
    ax_eap.text(-1, 0.07, "normalized\nspikes", fontsize=10)
    ax_eap.plot([1, 2], [-0.55, -0.55], c='k', lw=2)
    ax_eap.text(1.5, - 0.6, "1 ms", va="top", ha='center')
    #
    # ax_vm.set_yticks([v0])
    # ax_vm.set_yticklabels(["{:d} mV".format(v0)])

    fig.savefig("{}.png".format(figname), dpi=300)


if __name__ == '__main__':
    # save_somatic_spike_vmem()
    soma_t, soma_vmem = np.load("somatic_vmem.npy")

    hay_spike_replay(soma_vmem)
    # ball_and_stick_spike_replay(soma_vmem)
    # two_comp_spike_replay(soma_vmem)


