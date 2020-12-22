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
sigma = 0.3

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
                sigma = sigma,      # extracellular conductivity
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

    # np.save("two_comp_imem.npy", cell.imem)
    # np.save("two_comp_xyz.npy", [cell.x, cell.y, cell.z])
    plot_results(cell, "fig_two_comp_LFP", "two-compartment", "D")
    # plot_dipole_decay(cell)
    cell.__del__()

def two_comp_dipole_decay():

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

    # plot_two_monopole_versus_dipole(cell)
    plot_two_monopole_decay_directions(cell)
    cell.__del__()




def plot_grid_LFP(cell, grid_elec_params, grid_x, grid_z,
                  ax, synapses, scale_max=None):

    # Create a grid of measurement locations, in (um)
    grid_x, grid_z = np.mgrid[-450:451:20, -370:1200:20]
    grid_y = np.ones(grid_x.shape) * 0

    # Define electrode parameters
    grid_elec_params = {
        'sigma': sigma,      # extracellular conductivity
        'x': grid_x.flatten(),  # electrode positions
        'y': grid_y.flatten(),
        'z': grid_z.flatten(),
        'method': 'linesource'
    }


    ax_lfp_dict = dict(aspect=1, frameon=False, xticks=[], yticks=[],
                            ylim=[np.min(grid_z), np.max(grid_z)],
                            xlim=[np.min(grid_x), np.max(grid_x)])

    grid_electrode = LFPy.RecExtElectrode(cell, **grid_elec_params)
    M_elec_ps = grid_electrode.get_transformation_matrix()
    lfp_ = M_elec_ps @ cell.imem * 1000


    max_amp_elec_idx = np.argmax(np.max(np.abs(lfp_), axis=1))
    max_amp_t_idx = np.argmax(np.abs(lfp_[max_amp_elec_idx, :]))

    max_amp_LFP = np.max(np.abs(lfp_))
    if not max_amp_LFP == np.abs(lfp_[max_amp_elec_idx, max_amp_t_idx]):
        raise RuntimeError("Wrong with chosen max value")

    LFP = lfp_[:, max_amp_t_idx].reshape(grid_x.shape)

    num = 15
    levels = np.linspace(0.01, 1, num=num)

    print(np.max(np.abs(LFP)))
    scale_max = np.max(np.abs(LFP)) if scale_max is None else scale_max
    print(scale_max)

    levels_norm = scale_max * np.concatenate((-levels[::-1], levels))
    rainbow_cmap = plt.cm.get_cmap('PRGn')  # rainbow, spectral, RdYlBu

    colors_from_map = [rainbow_cmap(i*np.int(255/(len(levels_norm) - 2)))
                       for i in range(len(levels_norm) -1)]
    colors_from_map[num - 1] = (1.0, 1.0, 1.0, 1.0)

    [ax.plot([cell.x[idx, 0], cell.x[idx, 1]],
                 [cell.z[idx, 0], cell.z[idx, 1]], lw=1, c='gray')
     for idx in range(cell.totnsegs)]

    [ax.plot(cell.x[syn.idx].mean(), cell.z[syn.idx].mean(),
             marker='o', c='cyan', ms=5, mec='k')
     for syn in synapses]

    ep_intervals = ax.contourf(grid_x, grid_z, LFP,
                               zorder=2, colors=colors_from_map,
                               levels=levels_norm, extend='both')

    ax.contour(grid_x, grid_z, LFP, colors='k', linewidths=(1), zorder=2,
               levels=levels_norm)
    return ep_intervals


def plot_two_monopole_versus_dipole(cell):

    max_t_idx = np.argmax(np.abs(cell.imem[0, :]))

    i = cell.imem[:, max_t_idx]
    ia = np.abs(i[0])

    l_d = cell.z.mean(axis=1)[1] - cell.z.mean(axis=1)[0]
    cell.z -= l_d / 2

    x, y, z = cell.x.mean(axis=1), cell.y.mean(axis=1), cell.z.mean(axis=1)

    # Unit vector pointing from negative to positive current
    e_p_vec = np.array([0, 0, -1])
    p = ia * l_d * e_p_vec

    error_radius = 800

    num_elecs = 100

    elec_params_0deg = dict(
            sigma = sigma,      # extracellular conductivity
            x = np.zeros(num_elecs),
            y = np.zeros(num_elecs),
            z = np.max(z) + 100 + np.linspace(0, 3000, num_elecs),
            method = 'pointsource',
        )

    elec_params_60deg = dict(
            sigma = sigma,      # extracellular conductivity
            x = np.sin(np.deg2rad(60)) * np.linspace(0, 3000, num_elecs),
            y = np.zeros(num_elecs),
            z = np.cos(np.deg2rad(60)) * np.linspace(0, 3000, num_elecs),
            method = 'pointsource',
        )

    dist_0deg = np.sqrt(elec_params_0deg['x'] ** 2 + elec_params_0deg['z'] ** 2)
    dist_60deg = np.sqrt(elec_params_60deg['x'] ** 2 + elec_params_60deg['z'] ** 2)

    idxs_0deg = np.where(dist_0deg > error_radius)
    idxs_60deg = np.where(dist_60deg > error_radius)

    elec_0deg = LFPy.RecExtElectrode(cell, **elec_params_0deg)
    M_elec_0deg = elec_0deg.get_transformation_matrix()
    lfp_0deg_2m = M_elec_0deg @ i * 1000

    elec_60deg = LFPy.RecExtElectrode(cell, **elec_params_60deg)
    M_elec_60deg = elec_60deg.get_transformation_matrix()
    lfp_60deg_2m = M_elec_60deg @ i * 1000

    electrode_locs_0deg = np.array([elec_params_0deg["x"],
                               elec_params_0deg["y"],
                               elec_params_0deg["z"]]).T
    electrode_locs_60deg = np.array([elec_params_60deg["x"],
                               elec_params_60deg["y"],
                               elec_params_60deg["z"]]).T

    r_mean_0deg = electrode_locs_0deg - np.array([x.mean(),
                                                  y.mean(),
                                                  z.mean()])
    r_mean_60deg = electrode_locs_60deg - np.array([x.mean(),
                                                    y.mean(),
                                                    z.mean()])

    lfp_0deg_dp = 1000 * 1. / (4 * np.pi * sigma) * (np.dot(r_mean_0deg, p.T)
                / np.linalg.norm(r_mean_0deg, axis=1) ** 3)

    lfp_60deg_dp = 1000 * 1. / (4 * np.pi * sigma) * (np.dot(r_mean_60deg, p.T)
                / np.linalg.norm(r_mean_60deg, axis=1) ** 3)


    grid_x, grid_z = np.mgrid[-2000:2001:27, -3000:3002:27]
    grid_y = np.zeros(grid_x.shape)

    # Define electrode parameters
    grid_electrode_parameters = {
        'sigma' : sigma,      # extracellular conductivity
        'x' : grid_x.flatten(),  # electrode requires 1d vector of positions
        'y' : grid_y.flatten(),
        'z' : grid_z.flatten(),
        'method': 'pointsource'
    }
    elec_grid = LFPy.RecExtElectrode(cell, **grid_electrode_parameters)
    M_elec_grid = elec_grid.get_transformation_matrix()
    lfp_2m = M_elec_grid @ i * 1000
    lfp_2m = lfp_2m.reshape(grid_x.shape)

    rvec_0deg = np.dot(np.linspace(0, 3000, 100)[:, None],
                       np.array([np.sin(0), 0, np.cos(0)])[:, None].T)

    rvec_60deg = np.dot(np.linspace(0, 3000, 100)[:, None],
                        np.array([np.sin(np.deg2rad(60)),
                                  0,
                                  np.cos(np.deg2rad(60))])[:, None].T)

    rvec_0deg += np.array([0, 0, np.max(z) + 100])
    rvec_60deg += np.array([0, 0, l_d/2])

    # dipole grid
    electrode_locs = np.array([grid_x.flatten(),
                               grid_y.flatten(),
                               grid_z.flatten()]).T
    r_mean = electrode_locs - np.array([x.mean(), y.mean(), z.mean()])

    lfp_dp_grid = 1000 * 1. / (4 * np.pi * sigma) * (np.dot(r_mean, p.T)
                / np.linalg.norm(r_mean, axis=1) ** 3).reshape(grid_x.shape)


    plt.close("all")
    fig = plt.figure(figsize=[10, 4])
    fig.subplots_adjust(left=0.03, wspace=0.5, right=0.98, bottom=0.17)

    ax_2m = fig.add_subplot(141, aspect=1, title="two-monopole",
                            frameon=False, xticks=[], yticks=[],
                            xlim=[-2000, 2000], ylim=[-3100, 3100])
    ax_dp = fig.add_subplot(142, aspect=1, title="dipole", frameon=False,
                            xticks=[], yticks=[],
                            xlim=[-2000, 2000], ylim=[-3100, 3100])
    ax_diff = fig.add_subplot(143, aspect=1, title="difference", frameon=False,
                              xticks=[], yticks=[],
                            xlim=[-800, 800], ylim=[-1200, 1200])
    ax2 = fig.add_subplot(144, xlabel="distance (µm)", ylabel="|$\phi$| (µV)")

    num = 15
    levels = np.logspace(-3, 0, num=num)

    print(np.max(np.abs(lfp_2m)))
    scale_max = 10 #np.max(np.abs(lfp_2m))
    print(scale_max)

    levels_norm = scale_max * np.concatenate((-levels[::-1], levels))
    rainbow_cmap = plt.cm.get_cmap('PRGn')  # rainbow, spectral, RdYlBu

    colors_from_map = [rainbow_cmap(i*np.int(255/(len(levels_norm) - 2)))
                       for i in range(len(levels_norm) -1)]
    colors_from_map[num - 1] = (1.0, 1.0, 1.0, 1.0)


    ep_2m = ax_2m.contourf(grid_x, grid_z, lfp_2m,
                               zorder=2, colors=colors_from_map,
                               levels=levels_norm, extend='both')

    ax_2m.contour(grid_x, grid_z, lfp_2m, colors='k',
                  linewidths=(1), zorder=2, levels=levels_norm)

    ep_dp = ax_dp.contourf(grid_x, grid_z, lfp_dp_grid,
                               zorder=2, colors=colors_from_map,
                               levels=levels_norm, extend='both')

    ax_dp.contour(grid_x, grid_z, lfp_dp_grid, colors='k',
                  linewidths=(1), zorder=2, levels=levels_norm)

    ep_diff = ax_diff.contourf(grid_x, grid_z, lfp_dp_grid - lfp_2m,
                               zorder=2, colors=colors_from_map,
                               levels=levels_norm, extend='both')

    ax_diff.contour(grid_x, grid_z, lfp_dp_grid - lfp_2m, colors='k',
                    linewidths=(1), zorder=2,
               levels=levels_norm)

    imgs = [ep_2m, ep_dp, ep_diff]

    for i, ax in enumerate([ax_2m, ax_dp, ax_diff]):
        ax.plot(x, z, 'o', c='k')
        [ax.plot(x[i], z[i], '+_'[i], c='w') for i in range(2)]

        if i < 2:
            ax.plot(elec_params_0deg['x'][idxs_0deg],
                    elec_params_0deg['z'][idxs_0deg], ['-', '--'][i], c='b')
            ax.plot(elec_params_60deg['x'][idxs_60deg],
                    elec_params_60deg['z'][idxs_60deg], ['-', '--'][i], c='r')

        ax.add_patch(plt.Circle((0, 0), radius=error_radius,
                                   color='none', zorder=50, ls='--',
                                   fill=True, ec='cyan', lw=3))
        ax_x1, ax_y1, ax_w, ax_h = ax.get_position().bounds

        cax = fig.add_axes([ax_x1, 0.19, ax_w, 0.01], frameon=False)
        cbar = fig.colorbar(imgs[i], cax=cax, orientation="horizontal")
        cbar.set_label('$\phi$ (µV)', labelpad=0)
        cbar.set_ticks(scale_max * np.array([-1, -0.1, -0.01, 0, 0.01, 0.1, 1]))

        cax.set_xticklabels(cax.get_xticklabels(), rotation=40)

    ax2.axvline(error_radius, lw=2, c='cyan', ls='--')
    l1, = ax2.plot(dist_0deg[idxs_0deg], np.abs(lfp_0deg_2m[idxs_0deg]), 'b')
    l2, = ax2.plot(dist_0deg[idxs_0deg], np.abs(lfp_0deg_dp[idxs_0deg]), 'b--')
    l3, = ax2.plot(dist_60deg[idxs_60deg], np.abs(lfp_60deg_2m[idxs_60deg]), 'r')
    l4, = ax2.plot(dist_60deg[idxs_60deg], np.abs(lfp_60deg_dp[idxs_60deg]), 'r--')

    ax2.legend([l1, l2, l3, l4], [r"2-monopole $\theta=0^{\circ}$",
                                  r"dipole $\theta=0^{\circ}$",
                                  r"2-monopole $\theta=60^{\circ}$",
                                  r"dipole $\theta=60^{\circ}$",
                                  ], frameon=False, fontsize=9.5, loc=(0.1, 0.7))

    ax_2m.plot([1800, 1800], [-2000, -1000], lw=2, c='k', clip_on=False)
    ax_2m.text(1900, -1500, "1000 µm", va='center')

    ax_dp.plot([1800, 1800], [-2000, -1000], lw=2, c='k', clip_on=False)
    ax_dp.text(1900, -1500, "1000 µm", va='center')

    ax_diff.plot([430, 430], [-750, -1150], lw=2, c='k', clip_on=False)
    ax_diff.text(460, -950, "400 µm", va='center')

    simplify_axes(ax2)
    mark_subplots([ax_2m, ax_dp, ax_diff, ax2], ypos=1.05, xpos=0.)
    plt.savefig(join(os.path.dirname(__file__), "dipole_decay.png"))
    plt.close("all")

    # Also make other simpler figure
    fig = plt.figure(figsize=[4, 4])
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.17, top=0.98)

    ax2 = fig.add_subplot(111, xlabel="distance (µm)", ylabel="|$\phi$| (µV)")

    # ax2.axvline(error_radius, lw=2, c='cyan', ls='--')
    l1, = ax2.plot(dist_0deg[idxs_0deg], np.abs(lfp_0deg_2m[idxs_0deg]), 'b')
    l2, = ax2.plot(dist_0deg[idxs_0deg], np.abs(lfp_0deg_dp[idxs_0deg]), 'b--')
    l3, = ax2.plot(dist_60deg[idxs_60deg], np.abs(lfp_60deg_2m[idxs_60deg]), 'r')
    l4, = ax2.plot(dist_60deg[idxs_60deg], np.abs(lfp_60deg_dp[idxs_60deg]), 'r--')

    ax2.legend([l1, l2, l3, l4], [r"2-monopole $\theta=0^{\circ}$",
                                  r"dipole $\theta=0^{\circ}$",
                                  r"2-monopole $\theta=60^{\circ}$",
                                  r"dipole $\theta=60^{\circ}$",
                                  ], frameon=False, loc="upper right")

    simplify_axes(ax2)
    plt.savefig(join(os.path.dirname(__file__), "dipole_decay_simpler.pdf"), dpi=300)
    plt.close("all")


def plot_two_monopole_decay_directions(cell):

    max_t_idx = np.argmax(np.abs(cell.imem[0, :]))

    i = cell.imem[:, max_t_idx]

    l_d = cell.z.mean(axis=1)[1] - cell.z.mean(axis=1)[0]
    cell.z -= l_d / 2

    x, y, z = cell.x.mean(axis=1), cell.y.mean(axis=1), cell.z.mean(axis=1)

    print(x, y, z)

    error_radius = 600

    num_elecs = 200

    elec_params_0deg = dict(
            sigma = sigma,      # extracellular conductivity
            x = np.zeros(num_elecs),
            y = np.zeros(num_elecs),
            z = np.linspace(0, 10000, num_elecs),
            method = 'pointsource',
        )

    elec_params_60deg = dict(
            sigma = sigma,      # extracellular conductivity
            x = np.sin(np.deg2rad(60)) * np.linspace(0, 10000, num_elecs),
            y = np.zeros(num_elecs),
            z = np.cos(np.deg2rad(60)) * np.linspace(0, 10000, num_elecs),
            method = 'pointsource',
        )

    elec_params_perp = dict(
            sigma = sigma,      # extracellular conductivity
            x = np.linspace(0, 10000, num_elecs),
            y = np.zeros(num_elecs),
            z = np.ones(num_elecs) * l_d / 2,
            method = 'pointsource',
        )

    dist_0deg = np.sqrt(elec_params_0deg['x'] ** 2 + elec_params_0deg['z'] ** 2)
    dist_60deg = np.sqrt(elec_params_60deg['x'] ** 2 + elec_params_60deg['z'] ** 2)
    dist_perp = np.sqrt(elec_params_perp['x'] ** 2 + elec_params_perp['z'] ** 2)

    idxs_0deg = np.where(dist_0deg > error_radius)
    idxs_60deg = np.where(dist_60deg > error_radius)
    idxs_perp = np.where(dist_perp > error_radius)

    elec_0deg = LFPy.RecExtElectrode(cell, **elec_params_0deg)
    M_elec_0deg = elec_0deg.get_transformation_matrix()
    lfp_0deg_2m = M_elec_0deg @ i * 1000

    elec_60deg = LFPy.RecExtElectrode(cell, **elec_params_60deg)
    M_elec_60deg = elec_60deg.get_transformation_matrix()
    lfp_60deg_2m = M_elec_60deg @ i * 1000

    elec_perp = LFPy.RecExtElectrode(cell, **elec_params_perp)
    M_elec_perp = elec_perp.get_transformation_matrix()
    lfp_perp_2m = M_elec_perp @ i * 1000

    grid_x, grid_z = np.mgrid[-2000:2001:27, -3000:3002:27]
    grid_y = np.zeros(grid_x.shape)

    # Define electrode parameters
    grid_electrode_parameters = {
        'sigma' : sigma,      # extracellular conductivity
        'x' : grid_x.flatten(),  # electrode requires 1d vector of positions
        'y' : grid_y.flatten(),
        'z' : grid_z.flatten(),
        'method': 'pointsource'
    }
    elec_grid = LFPy.RecExtElectrode(cell, **grid_electrode_parameters)
    M_elec_grid = elec_grid.get_transformation_matrix()
    lfp_2m = M_elec_grid @ i * 1000
    lfp_2m = lfp_2m.reshape(grid_x.shape)

    plt.close("all")
    fig = plt.figure(figsize=[5, 4])
    fig.subplots_adjust(left=0.03, wspace=0.6, right=0.98, bottom=0.17)

    ax_2m = fig.add_subplot(121, aspect=1,
                            frameon=False, xticks=[], yticks=[],
                            xlim=[-2000, 2000], ylim=[-3100, 3100])

    ax2 = fig.add_subplot(122, xlabel="distance (mm)",
                          xscale="log", yscale="log",
                          ylabel="|$\phi$| (µV)")

    num = 15
    levels = np.logspace(-3, 0, num=num)

    print(np.max(np.abs(lfp_2m)))
    scale_max = 10 #np.max(np.abs(lfp_2m))
    print(scale_max)

    levels_norm = scale_max * np.concatenate((-levels[::-1], levels))
    rainbow_cmap = plt.cm.get_cmap('PRGn')  # rainbow, spectral, RdYlBu

    colors_from_map = [rainbow_cmap(i*np.int(255/(len(levels_norm) - 2)))
                       for i in range(len(levels_norm) -1)]
    colors_from_map[num - 1] = (1.0, 1.0, 1.0, 1.0)


    ep_2m = ax_2m.contourf(grid_x, grid_z, lfp_2m,
                               zorder=2, colors=colors_from_map,
                               levels=levels_norm, extend='both')

    ax_2m.contour(grid_x, grid_z, lfp_2m, colors='k',
                  linewidths=(1), zorder=2, levels=levels_norm)


    ax_2m.plot(x, z, 'o', c='k')
    [ax_2m.plot(x[i], z[i], '+_'[i], c='w') for i in range(2)]

    ax_2m.plot(elec_params_0deg['x'][idxs_0deg],
            elec_params_0deg['z'][idxs_0deg], '--', c='b')
    ax_2m.plot(elec_params_60deg['x'][idxs_60deg],
            elec_params_60deg['z'][idxs_60deg], '--', c='r')

    ax_2m.plot(elec_params_perp['x'][idxs_perp],
            elec_params_perp['z'][idxs_perp], '--', c='orange')


    ax_2m.add_patch(plt.Circle((0, 0), radius=error_radius,
                               color='none', zorder=50, ls='--',
                               fill=True, ec='cyan', lw=3))
    ax_2m.plot(0, 0, 'o', c='cyan')
    ax_x1, ax_y1, ax_w, ax_h = ax_2m.get_position().bounds

    cax = fig.add_axes([ax_x1, 0.19, ax_w, 0.01], frameon=False)
    cbar = fig.colorbar(ep_2m, cax=cax, orientation="horizontal")
    cbar.set_label('$\phi$ (µV)', labelpad=0)
    cbar.set_ticks(scale_max * np.array([-1, -0.1, -0.01, 0, 0.01, 0.1, 1]))

    cax.set_xticklabels(cax.get_xticklabels(), rotation=40)

    ax2.axvline(error_radius / 1000, lw=2, c='cyan', ls='--')
    l1, = ax2.loglog(dist_0deg[idxs_0deg] / 1000, np.abs(lfp_0deg_2m[idxs_0deg]), 'b')
    # l2, = ax2.plot(dist_0deg[idxs_0deg], np.abs(lfp_0deg_dp[idxs_0deg]), 'b--')
    l3, = ax2.loglog(dist_60deg[idxs_60deg] / 1000, np.abs(lfp_60deg_2m[idxs_60deg]), 'r')
    l4, = ax2.loglog(dist_perp[idxs_perp] / 1000, np.abs(lfp_perp_2m[idxs_perp]), 'orange')
    # l4, = ax2.plot(dist_60deg[idxs_60deg], np.abs(lfp_60deg_dp[idxs_60deg]), 'r--')

    ax2.legend([l1, l3, l4], [r"$\theta=0^{\circ}$",
                              r"$\theta=60^{\circ}$",
                              r"perpendicular",
                              ],
               frameon=False, fontsize=9.5, loc=(0.25, 0.75))

    # Making 1/r**3 markers
    r1 = dist_perp[idxs_perp][-1] / 1000
    r0 = r1 * 0.5
    r = np.linspace(r0, r1, 10)
    slope_factor = np.abs(lfp_perp_2m[idxs_perp])[-1] * r1**3
    y = slope_factor / r ** 3
    ax2.plot(r, y, lw=3, c='k')
    ax2.text(r0*0.85, y[-1], "1/r$^3$", ha="left")

    # Making 1/r**2 markers
    r1 = dist_0deg[idxs_0deg][-1] / 1000
    r0 = r1 * 0.5
    r = np.linspace(r0, r1, 10)
    slope_factor = np.abs(lfp_0deg_2m[idxs_0deg])[-1] * r1**2
    y = slope_factor / r ** 2
    ax2.plot(r, y, lw=3, c='k')
    ax2.text(r0*1.1, y[0], "1/r$^2$", ha="left")

    ax_2m.plot([1800, 1800], [-2000, -1000], lw=2, c='k', clip_on=False)
    ax_2m.text(1900, -1500, "1 mm", va='center')

    ax2.grid(True)

    simplify_axes(ax2)
    mark_subplots([ax_2m, ax2], ypos=1.05, xpos=0.)
    plt.savefig(join(os.path.dirname(__file__), "two_monopole_decay_direction.png"))
    plt.close("all")





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

    ax_imem = fig.add_axes([0.67, 0.55, 0.3, 0.25],
                           title="normalized\ntransmembrane\ncurrents",
                          xticks=[], yticks=[], frameon=False,)

    ax_eap = fig.add_axes([0.67, 0.15, 0.3, 0.25],
                          title="normalized\nLFP traces",
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
                      cell.z[cell.synidx].mean(axis=1), c='y',
                               marker='*', ls='none')
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
    # two_comp_LFP()
    two_comp_dipole_decay()


