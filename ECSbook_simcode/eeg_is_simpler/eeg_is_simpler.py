import os
import sys
from os.path import join
import numpy as np
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Ellipse
import neuron
import LFPy
from ECSbook_simcode.plotting_convention import mark_subplots
from ECSbook_simcode.neural_simulations import return_hay_cell
import ECSbook_simcode.neural_simulations as ns
import seaborn as sns
# sns.set_theme()
cmap = plt.cm.get_cmap('icefire')

neuron.load_mechanisms(ns.cell_models_folder)


# Create a grid of measurement locations, in (um)

sigma = 0.3

num_elecs = 12
# Define electrode parameters
elec_params = {
    'sigma': sigma,      # extracellular conductivity
    'x': np.zeros(num_elecs),  # electrode positions
    'y': np.zeros(num_elecs),
    'z': np.linspace(-1200, -10, num_elecs),
    'method': 'soma_as_point'
}
dz = np.abs(elec_params["z"][1] - elec_params["z"][0])

synapse_params = {
    'syntype' : 'ExpSynI',      #conductance based exponential synapse
    'tau' : 1.,                #Time constant, rise
    'weight' : 0.005,           #Synaptic weight
    'record_current' : False,    #record synaptic currents
}

# four_sphere properties
radii = [79000., 80000., 85000., 90000.]
sigmas = [0.3, 1.5, 0.015, 0.3]

rad_tol = 1e-2


def insert_synapses(cell, synapse_params, z_min,  spiketimes):
    ''' Find n compartments to insert synapses onto '''
    n = len(spiketimes)
    idx = cell.get_rand_idx_area_norm(section="allsec", nidx=n, z_min=z_min)
    for i in idx:
        synapse_params.update({'idx' : int(i)})
        s = LFPy.Synapse(cell, **synapse_params)
        s.set_spike_times(np.array([spiketimes[i]]))


def plot_four_sphere_model(ax):

    # show 4S-illustration
    head_colors = plt.cm.Pastel1([0, 1, 2, 3])
    radii_tweaked = [radii[0]] + [r + 500 for r in radii[1:]]
    for i in range(4):
        ax.add_patch(plt.Circle((0, 0), radius=radii_tweaked[-1 - i],
                                   color=head_colors[-1-i],
                                   fill=True, ec='k', lw=.1))
    ax.add_patch(Ellipse((0, radii_tweaked[-1]), 2000, 500))

def plot_laminar_lfp(lfp, ax, tvec, normalize):

    z = elec_params["z"]
    dz = np.abs(z[1] - z[0])
    lfp_ = lfp / normalize
    for elec in range(lfp.shape[0]):
        ax.plot(tvec, lfp_[elec] * dz / 1.5 + z[elec], c='w')
    img = ax.imshow(lfp, cmap=cmap, origin="lower",
                    vmax=normalize, vmin=-normalize,
                    extent=[0, tvec[-1], np.min(z) - dz/2, np.max(z) + dz/2])
    ax.axis("auto")
    return img

def make_figure():

    num_syns = 1000
    input_t_center = 15
    input_t_std = 2
    tstop = 30
    dt = 2**-4
    tvec = np.arange(tstop / dt + 1) * dt

    z_mins = np.array([-600, -400, -200])
    depth_clrs = {z_min: plt.cm.rainbow(i / (len(z_mins) - 1))
                  for i, z_min in enumerate(z_mins)}

    ax_h_eeg = 0.1
    ax_h_lfp = 0.6
    ax_lfp_hstart = 0.05
    ax_eeg_hstart = 0.72
    ax_w = 0.22
    ax_wspace = 0.01
    ax_left = 0.0

    plt.close("all")
    fig = plt.figure(figsize=[10, 5])

    ax_grid_lfp = [ax_left, 0, ax_w, 0.66]
    ax_morph = fig.add_axes(ax_grid_lfp, frameon=False, aspect=1,
                            xticks=[], yticks=[], xlim=[-350, 350],
                            ylim=[-1400, 100])

    ax_grid_4s = [0.01, ax_eeg_hstart, 0.2, 0.2]
    ax_4s = fig.add_axes(ax_grid_4s, frameon=False, aspect=1,
                         xticks=[], yticks=[],
                         xlim=[-17000, 17000],
                         ylim=[77000, 92000])
    ax_4s.set_title("head model")
    plot_four_sphere_model(ax_4s)

    ax_morph.plot(elec_params["x"], elec_params["z"], 'o',
                  c='gray', ms=5, zorder=1)
    ax_morph.axhline(0, c='gray', ls="--")
    ax_morph.text(-400, 20, "z=0 µm", va="bottom")

    num_trials = 10
    seeds = np.random.randint(1, 124121, num_trials)
    lfp_dict = {}
    eeg_dict = {}

    for i, z_min in enumerate(z_mins):
        eeg_trial = []
        lfp_trial = []

        for trial, seed in enumerate(seeds):
            np.random.seed(seed)
            spiketimes = np.random.normal(input_t_center, input_t_std, size=num_syns)
            cell = return_hay_cell(tstop=tstop, dt=dt, make_passive=True)
            cell.set_pos(z=-np.max(cell.zend) - 10)
            insert_synapses(cell, synapse_params, z_min, spiketimes)
            cell.simulate(rec_imem=True, rec_vmem=True,
                      rec_current_dipole_moment=True)
            print("Max vmem: ", np.max(cell.vmem))

            electrode = LFPy.RecExtElectrode(cell, **elec_params)
            electrode.calc_lfp()
            print("Max LFP: ", np.max(np.abs(electrode.LFP * 1000)))

            somapos = np.array([0., 0., radii[0] + cell.zmid[0]])
            r_soma_syns = [cell.get_intersegment_vector(idx0=0, idx1=i)
                           for i in cell.synidx]
            r_mid = np.average(r_soma_syns, axis=0)
            r_mid = somapos + r_mid/2.

            eeg_coords_top = np.array([[0., 0., radii[-1] - rad_tol]])

            four_sphere_top = LFPy.FourSphereVolumeConductor(eeg_coords_top,
                                                             radii, sigmas)
            pot_db_4s_top = four_sphere_top.calc_potential(
                cell.current_dipole_moment, r_mid)

            eeg_trial.append(np.array(pot_db_4s_top)[0] * 1e6)
            print("Max EEG: ", np.max(np.abs(np.array(pot_db_4s_top)[0] * 1e6)))
            lfp_trial.append(electrode.LFP * 1000)

            cell.__del__()
        eeg_dict[z_min] = np.average(eeg_trial, axis=0)
        lfp_dict[z_min] = np.average(lfp_trial, axis=0)

    cell = return_hay_cell(tstop=tstop, dt=dt, make_passive=True)
    cell.set_pos(z=-np.max(cell.zend) - 10)
    for idx in range(cell.totnsegs):
        ax_morph.plot([cell.xstart[idx], cell.xend[idx]],
                   [cell.zstart[idx], cell.zend[idx]],
                      c='k', lw=1, zorder=-1)
        ax_4s.plot([cell.xstart[idx], cell.xend[idx]],
                   [cell.zstart[idx] + radii[0],
                    cell.zend[idx] + radii[0]], c='k', lw=0.5)
    cell.__del__()

    lfp_normalize = np.max([np.max(np.abs(lfp)) for lfp in lfp_dict.values()])
    eeg_normalize = np.max([np.max(np.abs(eeg)) for eeg in eeg_dict.values()])

    ax_lfp_dict = dict(ylim=[-1350, 50],
                       frameon=False, xticks=[], yticks=[])
    ax_eeg_dict = dict(frameon=False, xticks=[], yticks=[],
                       ylim=[-eeg_normalize * 1.05, eeg_normalize/5])

    img = None
    for i, z_min in enumerate(z_mins):

        ax_grid_lfp = [ax_left + (i + 1) * (ax_w + ax_wspace),
                   ax_lfp_hstart, ax_w, ax_h_lfp]
        ax_grid_eeg = [ax_left + (i + 1) * (ax_w + ax_wspace),
                   ax_eeg_hstart, ax_w, ax_h_eeg]

        ax_lfp = fig.add_axes(ax_grid_lfp, **ax_lfp_dict)
        ax_eeg = fig.add_axes(ax_grid_eeg, **ax_eeg_dict)
        ax_eeg.text(0.05, 0.01, "EEG", va="bottom")
        ax_lfp.text(0.05, dz / 2, "LFP", va="bottom")
        ax_eeg.text(tstop/2, 0.15, "input above z=%d µm" % z_min, ha='center')
        img = plot_laminar_lfp(lfp_dict[z_min], ax_lfp, tvec, lfp_normalize)
        ax_eeg.plot(tvec, eeg_dict[z_min])

        ax_morph.plot([200 + i * 17, 200 + i * 17], [0, z_min],
                      lw=2, c=depth_clrs[z_min])
        ax_lfp.plot([-1, -1], [dz/2, z_min + dz/2], c=depth_clrs[z_min])
        ax_eeg.plot([1, 1], [-0.01, -0.11], c='k', lw=2)
        ax_eeg.text(2, -0.06, "0.1 nV", va="center")

        ax_eeg.plot([1, 6], [-0.11, -0.11], c='k', lw=2)
        ax_eeg.text(3.5, -0.12, "5 ms", va="top", ha="center")
        mark_subplots(ax_eeg, "BCD"[i], ypos=2.5, xpos=0.)

    mark_subplots(ax_4s, "A", ypos=1.35, xpos=0.1)
    # axes_to_mark.append(ax_lfp)

    #
    # mark_subplots(axes_to_mark, xpos=0.15, ypos=0.95)
    #
    cax = fig.add_axes([0.915, 0.05, 0.01, 0.57], frameon=False)
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label('$\phi$ (µV)', labelpad=0)
    # cbar.set_ticks([-50, -5, -0.5, 0.5, 5, 50, 500])

    fig.savefig("fig_eeg_is_simpler.png", dpi=300)
    fig.savefig("fig_eeg_is_simpler.pdf", dpi=300)


if __name__ == '__main__':
    # make_all_basal_dipoles()
    make_figure()

