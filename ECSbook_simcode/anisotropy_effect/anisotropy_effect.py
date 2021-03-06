import os
import sys
from os.path import join
import numpy as np
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import LFPy
from ECSbook_simcode.plotting_convention import mark_subplots, simplify_axes
from ECSbook_simcode.neural_simulations import return_hay_cell

np.random.seed(12345)

# Create a grid of measurement locations, in (um)
grid_x_zoom, grid_z_zoom = np.mgrid[-75:76:20, 100:250:20]
grid_y_zoom = np.ones(grid_x_zoom.shape) * 0

# Define electrode parameters
grid_elec_params_zoom = {
    'x': grid_x_zoom.flatten(),  # electrode positions
    'y': grid_y_zoom.flatten(),
    'z': grid_z_zoom.flatten(),
    'method': 'linesource'
}

ax_lfp_dict_zoom = dict(aspect=1, frameon=False, xticks=[], yticks=[],
                        ylim=[np.min(grid_z_zoom), np.max(grid_z_zoom)],
                        xlim=[np.min(grid_x_zoom), np.max(grid_x_zoom)])

# Create a grid of measurement locations, in (um)
grid_x, grid_z = np.mgrid[-450:451:20, -370:1200:20]
grid_y = np.ones(grid_x.shape) * 0

# Define electrode parameters
grid_elec_params = {
    'x': grid_x.flatten(),  # electrode positions
    'y': grid_y.flatten(),
    'z': grid_z.flatten(),
    'method': 'linesource'
}


ax_lfp_dict = dict(aspect=1, frameon=False, xticks=[], yticks=[],
                        ylim=[np.min(grid_z), np.max(grid_z)],
                        xlim=[np.min(grid_x), np.max(grid_x)])


def insert_synaptic_input(idx, cell):

    synapse_parameters = {'e': 0., # reversal potential
                          'weight': 0.002, # 0.001, # synapse weight
                          'record_current': True, # record synapse current
                          'syntype': 'Exp2Syn',
                          'tau1': 1, #Time constant, rise
                          'tau2': 3, #Time constant, decay
                          }
    synapse_parameters['idx'] = idx
    synapse = LFPy.Synapse(cell, **synapse_parameters)
    synapse.set_spike_times(np.array([1.]))
    return synapse, cell


def plot_grid_LFP(cell, grid_elec_params, grid_x, grid_z,
                  ax, synapses, scale_max=None):

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

def make_figure():

    plt.close("all")
    fig = plt.figure(figsize=[4.5, 4.])
    fig.subplots_adjust(top=.85, bottom=0., wspace=0.1, left=0.02, right=0.98)
    tstop = 10
    dt = 2**-4

    scalemax_zoom = 3
    scalemax = 1

    syn_height = 150

    # Plot single-synapse dipoles
    cell = return_hay_cell(tstop=tstop, dt=dt, make_passive=True)
    idx = cell.get_closest_idx(x=0, y=0, z=syn_height)
    syn_pos = cell.x[idx].mean(), cell.y[idx].mean(), cell.z[idx].mean()
    syn, cell = insert_synaptic_input(idx, cell)
    cell.simulate(rec_imem=True)

    num_lateral_elecs = 20
    sigma_anis = [0.3, 0.3, 0.45]
    sigma_iso = [0.3, 0.3, 0.3]
    # Define electrode parameters
    # lateral_elec_params = dict(
    #     x = np.linspace(syn_pos[0] + 2, 70, num_lateral_elecs),
    #     y = np.ones(num_lateral_elecs) * syn_pos[1],
    #     z = np.ones(num_lateral_elecs) * syn_pos[2],
    #     method = 'root_as_point'
    # )

    # lateral_elec_params["sigma"] = sigma_iso
    # lat_elec_ps = LFPy.RecExtElectrode(cell, **lateral_elec_params)
    # M_elec_ps = lat_elec_ps.get_transformation_matrix()
    # lfp_ps = M_elec_ps @ cell.imem * 1000

    # max_amp_elec_idx = np.argmax(np.max(np.abs(lfp_ps), axis=1))
    # max_amp_t_idx = np.argmax(np.abs(lfp_ps[max_amp_elec_idx, :]))

    # lateral_elec_params["sigma"] = sigma_anis
    # lat_elec_ls = LFPy.RecExtElectrode(cell, **lateral_elec_params)
    # M_elec_ls = lat_elec_ls.get_transformation_matrix()
    # lfp_ls = M_elec_ls @ cell.imem * 1000

    # max_amp_LFP = np.max(np.abs(lfp_ps))
    # if not max_amp_LFP == np.abs(lfp_ps[max_amp_elec_idx, max_amp_t_idx]):
    #     raise RuntimeError("Wrong with chosen max value")

    # ax_ps_1 = fig.add_axes([0.03, 0.55, 0.35, 0.4],
    #                        title="point source\nzoom-in", **ax_lfp_dict_zoom)
    ax_iso = fig.add_subplot(121,
                           title="isotropic\n$\sigma_T$=[0.3, 0.3, 0.3]", **ax_lfp_dict)
    # ax_ls_1 = fig.add_axes([0.45, 0.55, 0.35, 0.4],
    #                        title="line source\nzoom-in", **ax_lfp_dict_zoom)
    ax_anis = fig.add_subplot(122,
                           title="aniostropic\n$\sigma_T$=[0.3, 0.3, 0.45]", **ax_lfp_dict)

    # ax_diff = fig.add_subplot(133,
                           #xticks=[0, 25, 50], ylim=[0.2e-2, 5e0],
                           # ylabel="difference")#, xlabel="µm")

    # rel_diff = (np.abs(lfp_ps[:, max_amp_t_idx] -
    #                   lfp_ls[:, max_amp_t_idx]) /
    #             np.max(np.abs(lfp_ls[:, max_amp_t_idx])))

    # dist = lat_elec_ps.x - syn_pos[0]

    # ax_diff.semilogy(dist, rel_diff, c='r', lw=2, ls=":")
    # ax_diff.set_yticks([1e-2, 1e-1, 1e0])

    # rel_err_10p = np.argmin(np.abs(rel_diff - 0.1))
    # rel_err_1p = np.argmin(np.abs(rel_diff - 0.01))

    # print(rel_diff[rel_err_10p], dist[rel_err_10p])
    # print(rel_diff[rel_err_1p], dist[rel_err_1p])

    # ax_ps_1.plot(lat_elec_ps.x, lat_elec_ps.z, c='r', lw=2, ls=":")

    grid_elec_params_zoom["sigma"] = sigma_iso
    grid_elec_params["sigma"] = sigma_iso
    # ep_intervals_ps = plot_grid_LFP(cell, grid_elec_params_zoom,
    #                                 grid_x_zoom, grid_z_zoom, ax_ps_1,
    #                                 [syn], scale_max=scalemax_zoom)
    ep_intervals_iso = plot_grid_LFP(cell, grid_elec_params,
                                     grid_x, grid_z, ax_iso,
                                    [syn], scale_max=scalemax)

    grid_elec_params_zoom["sigma"] = sigma_anis
    grid_elec_params["sigma"] = sigma_anis
    # ep_intervals_ls = plot_grid_LFP(cell, grid_elec_params_zoom,
    #                                 grid_x_zoom, grid_z_zoom, ax_ls_1,
    #                                 [syn], scale_max=scalemax_zoom)
    ep_intervals_anis = plot_grid_LFP(cell, grid_elec_params, grid_x,
                                     grid_z, ax_anis,
                                    [syn], scale_max=scalemax)
    cell.__del__()

    # for ax in [ax_ps_1, ax_ls_1]:
    #     ax.plot([-70, -20], [105, 105], lw=2, c='k')
    #     ax.text(-45, 107, "50 µm", ha='center')

    for ax in [ax_iso]:
        ax.plot([200, 200], [1000, 500], lw=2, c='k', clip_on=False)
        ax.text(220, 750, "500 µm", ha='left')

    # simplify_axes(ax_diff)

    # cax_zoom = fig.add_axes([0.81, 0.72, 0.01, 0.2], frameon=False)
    # cbar_zoom = fig.colorbar(ep_intervals_ps, cax=cax_zoom)
    # cbar_zoom.set_label('$\phi$ (µV)', labelpad=0)
    # cbar_zoom.set_ticks(np.array([np.arange(-scalemax_zoom, scalemax_zoom + 1)]))

    cax = fig.add_axes([0.47, 0.1, 0.01, 0.2], frameon=False)
    cbar = fig.colorbar(ep_intervals_iso, cax=cax)
    cbar.set_label('$\phi$ (µV)', labelpad=0)
    cbar.set_ticks(np.array([np.arange(-scalemax, scalemax + 1)]))

    # axes_to_mark = [ax_ps_1, ax_ls_1, ax_iso, ax_anis, ax_diff]
    axes_to_mark = [ax_iso, ax_anis]#, ax_diff]
    mark_subplots(axes_to_mark, xpos=-0.0, ypos=1.12)

    fig.savefig("fig_anisotropy_effect.png", dpi=150)
    fig.savefig("fig_anisotropy_effect.pdf", dpi=150)


if __name__ == '__main__':
    make_figure()

