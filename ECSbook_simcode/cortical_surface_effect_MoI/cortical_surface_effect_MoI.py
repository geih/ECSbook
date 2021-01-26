import numpy as np
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import neuron
import LFPy
from ECSbook_simcode.plotting_convention import mark_subplots
from ECSbook_simcode.neural_simulations import return_hay_cell
import ECSbook_simcode.neural_simulations as ns
import seaborn as sns
# sns.set_theme()
cmap = plt.cm.get_cmap('icefire')

neuron.load_mechanisms(ns.cell_models_folder)

sigma = 0.3
num_elecs = 12
# Define electrode parameters
elec_params = {
    'sigma_G': sigma,      # extracellular conductivity
    'sigma_T': sigma,      # extracellular conductivity
    # 'sigma_S': sigma,      # extracellular conductivity
    'x': np.zeros(num_elecs),  # electrode positions
    'y': np.zeros(num_elecs),
    'z': np.linspace(-1200, -10, num_elecs),
    'method': 'pointsource',
    'h': 2000,
    'z_shift': -2000,
    }
dz = np.abs(elec_params["z"][1] - elec_params["z"][0])

synapse_params = {
    'syntype' : 'ExpSynI',      #conductance based exponential synapse
    'tau' : 1.,                #Time constant, rise
    'weight' : 0.005,           #Synaptic weight
    'record_current' : False,    #record synaptic currents
}


cover_sigmas = [0.0, 0.3, 1e9]
cover_sigma_names = ["0 S/m", "0.3 S/m", "$\infty$ S/m"]
def insert_synapses(cell, synapse_params, z_min,  spiketimes):
    ''' Find n compartments to insert synapses onto '''
    n = len(spiketimes)
    idx = cell.get_rand_idx_area_norm(section="allsec", nidx=n, z_min=z_min)
    for i in idx:
        synapse_params.update({'idx' : int(i)})
        s = LFPy.Synapse(cell, **synapse_params)
        s.set_spike_times(np.array([spiketimes[i]]))


def plot_laminar_lfp(lfp, ax, tvec, normalize):

    z = elec_params["z"]
    dz = np.abs(z[1] - z[0])
    lfp_ = lfp / normalize
    for elec in range(lfp.shape[0]):
        ax.plot(tvec, lfp_[elec] * dz / 1.5 + z[elec], c='gray')
    img = ax.imshow(lfp, cmap=cmap, origin="lower",
                    vmax=normalize, vmin=-normalize, rasterized=True,
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

    z_min = np.array([-200])

    ax_h_lfp = 0.78
    ax_lfp_hstart = 0.07
    ax_w = 0.215
    ax_wspace = 0.01
    ax_left = 0.0

    plt.close("all")
    fig = plt.figure(figsize=[10, 3.5])

    ax_grid_lfp = [ax_left, 0.01, ax_w, 0.95]
    ax_morph = fig.add_axes(ax_grid_lfp, frameon=False, aspect=1,
                            xticks=[], yticks=[], xlim=[-350, 350],
                            ylim=[-1400, 250])

    ax_morph.plot(elec_params["x"], elec_params["z"], 'o',
                  c='gray', ms=5, zorder=1)
    ax_morph.axhline(0, c='gray', ls="--")
    # ax_morph.text(-400, 20, "z=0 µm", va="bottom")
    ax_morph.axhspan(-2000, 0, fc='0.8', zorder=-50)
    ax_morph.axhspan(0, 2000, fc='lightblue', zorder=-50)
    ax_morph.text(-320, -25, "$\sigma_t$", va='top')
    ax_morph.text(-320, 25, "$\sigma_{cover}$", va='bottom')

    num_trials = 10
    seeds = np.random.randint(1, 124121, num_trials)
    lfp_dict = {}

    for i, cover_sigma in enumerate(cover_sigmas):
        lfp_trial = []
        elec_params["sigma_S"] = cover_sigma
        for seed in seeds:
            np.random.seed(seed)
            spiketimes = np.random.normal(input_t_center, input_t_std, size=num_syns)
            cell = return_hay_cell(tstop=tstop, dt=dt, make_passive=True)
            cell.set_pos(z=-np.max(cell.z) - 10)
            insert_synapses(cell, synapse_params, z_min, spiketimes)
            cell.simulate(rec_imem=True, rec_vmem=True)
            print("Max vmem: ", np.max(cell.vmem))

            electrode = LFPy.RecMEAElectrode(cell, **elec_params)
            M = electrode.get_transformation_matrix()
            LFP = M @ cell.imem * 1000
            print("Max LFP: ", np.max(np.abs(LFP)))
            lfp_trial.append(LFP)
            cell.__del__()
        lfp_dict[cover_sigma] = np.average(lfp_trial, axis=0)

    cell = return_hay_cell(tstop=tstop, dt=dt, make_passive=True)
    cell.set_pos(z=-np.max(cell.z) - 10)
    for idx in range(cell.totnsegs):
        ax_morph.plot([cell.x[idx, 0], cell.x[idx, 1]],
                   [cell.z[idx, 0], cell.z[idx, 1]],
                      c='k', lw=1, zorder=-1)

    cell.__del__()

    lfp_normalize = np.max([np.max(np.abs(lfp)) for lfp in lfp_dict.values()])

    ax_lfp_dict = dict(ylim=[-1350, 50],
                       frameon=False, xticks=[], yticks=[])

    img = None
    for i, cover_sigma in enumerate(cover_sigmas):

        ax_grid_lfp = [ax_left + (i + 1) * (ax_w + ax_wspace),
                   ax_lfp_hstart, ax_w, ax_h_lfp]

        ax_lfp = fig.add_axes(ax_grid_lfp, **ax_lfp_dict)
        ax_lfp.text(0.05, dz / 2, "$\sigma_{cover}=$%s" % cover_sigma_names[i],
                    va="bottom")
        img = plot_laminar_lfp(lfp_dict[cover_sigma], ax_lfp,
                               tvec, lfp_normalize)
        ax_lfp.plot([1, 6],
                    [elec_params["z"][0] + 20, elec_params["z"][0] + 20],
                    lw=2, c='w', clip_on=False)
        ax_lfp.text(3.5, elec_params["z"][0] + 40,
                    "5 ms", color="w", ha='center')

    mark_subplots(fig.axes[:1], xpos=-0.1, ypos=0.95)
    mark_subplots(fig.axes[1:], "BCDE", xpos=-0.05, ypos=1.1)

    cax = fig.add_axes([0.91, 0.07, 0.01, 0.57], frameon=False)
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label('$\phi$ (µV)', labelpad=0)

    fig.savefig("fig_cortical_surface_effect_MoI.png", dpi=300)
    fig.savefig("fig_cortical_surface_effect_MoI.pdf", dpi=300)


if __name__ == '__main__':
    make_figure()

