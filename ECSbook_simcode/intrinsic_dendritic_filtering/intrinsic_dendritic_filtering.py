
import os
from os.path import join
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import neuron
from neuron import h
import LFPy
import plotting_convention

def return_cell(tstop):

    h("forall delete_section()")
    h("""
    proc celldef() {
      topol()
      subsets()
      geom()
      biophys()
      geom_nseg()
    }

    create axon[1]

    proc topol() { local i
      basic_shape()
    }
    proc basic_shape() {
      axon[0] {pt3dclear()
      pt3dadd(0, 0, 0, 1)
      pt3dadd(0, 0, 1000, 1)}
    }

    objref all
    proc subsets() { local i
      objref all
      all = new SectionList()
        axon[0] all.append()

    }
    proc geom() {
    }
    proc geom_nseg() {
    forall {nseg = 200}
    }
    proc biophys() {
    }
    celldef()

    Ra = 150.
    cm = 1.
    Rm = 30000.

    forall {
        insert pas // 'pas' for passive, 'hh' for Hodgkin-Huxley
        g_pas = 1 / Rm
        }
    """)
    cell_params = {
                'morphology': h.all,
                'delete_sections': False,
                'v_init': -70.,
                'passive': False,
                'nsegs_method': None,
                'dt': tstop / 500,
                'tstart': 0.,
                'tstop': tstop,
            }
    cell = LFPy.Cell(**cell_params)
    cell.set_pos(x=-cell.xstart[0])
    return cell

frequencies = np.array([1, 100, 1000])
fig_folder = os.path.abspath('.')

LFP_dict = {}
imem_dict = {}
somav_dict = {}
syni_dict = {}
max_idx_dict = {}
tvec_dict = {}
color_dict = {frequencies[0]: 'gray',
              frequencies[1]: 'k',
              }

# Create a grid of measurement locations, in (um)
grid_x, grid_z = np.mgrid[-350:351:20, -400:1150:20]
grid_y = np.ones(grid_x.shape) * 0

sigma = 0.3

# Define electrode parameters
grid_electrode_parameters = {
    'sigma' : sigma,      # extracellular conductivity
    'x' : grid_x.flatten(),  # electrode requires 1d vector of positions
    'y' : grid_y.flatten(),
    'z' : grid_z.flatten(),
    'method': 'linesource'
}


for freq in frequencies:

    tstop = 1000. / freq
    cell = return_cell(tstop)

    stim_idx = 0
    stim_params = {
                 'idx': stim_idx,
                 'record_current': True,
                 'syntype': 'SinSyn',
                 'del': 0.,
                 'dur': 1e9,
                 'pkamp': 0.1,
                 'freq': freq,
                 'weight': 1.0,  # not in use, but required for Synapse
                }

    syn = LFPy.Synapse(cell, **stim_params)
    cell.simulate(rec_vmem=True, rec_imem=True)

    max_idx = np.argmin(cell.imem[0, :])
    # print(cell.tvec[max_idx])
    grid_electrode = LFPy.RecExtElectrode(cell, **grid_electrode_parameters)
    grid_electrode.calc_lfp(t_indices=max_idx)

    imem_dict[freq] = cell.imem[:, max_idx]
    somav_dict[freq] = [cell.tvec.copy(), cell.vmem[0].copy()]
    LFP_dict[freq] = 1000 * grid_electrode.LFP[:].reshape(grid_x.shape)
    syni_dict[freq] = syn.i
    max_idx_dict[freq] = max_idx
    tvec_dict[freq] = cell.tvec
    cell.__del__()


electrode_locs = np.array([grid_x.flatten(),
                           grid_y.flatten(),
                           grid_z.flatten()]).T

num = 11
levels = np.logspace(-2.5, 0, num=num)

scale_max = 10

levels_norm = scale_max * np.concatenate((-levels[::-1], levels))
rainbow_cmap = plt.cm.get_cmap('PRGn') # rainbow, spectral, RdYlBu

colors_from_map = [rainbow_cmap(i*np.int(255/(len(levels_norm) - 2)))
                   for i in range(len(levels_norm) -1)]
colors_from_map[num - 1] = (1.0, 1.0, 1.0, 1.0)

fig = plt.figure(figsize=[6, 4])
fig.subplots_adjust(bottom=0.15, top=0.93, right=0.9, left=0.01, wspace=0.4,
                    hspace=0.05)

cax = fig.add_axes([0.1, 0.125, 0.8, 0.02], frameon=False)

ax_dict = {}
gs = GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[0.2, 1])
for i, freq in enumerate(frequencies):

    imem = imem_dict[freq]
    LFP = LFP_dict[freq]

    ax_stim = fig.add_subplot(gs[i], frameon=False, xticks=[], yticks=[])
    ax_stim.set_title("{:d} Hz".format(freq), fontsize=16)

    ax_LFP = fig.add_subplot(gs[i + 3], aspect=1,
                          frameon=False,  xticks=[], yticks=[],
                         ylim=[np.min(grid_electrode.z), np.max(grid_electrode.z)],
                          xlim=[np.min(grid_electrode.x), np.max(grid_electrode.x)])

    [ax_LFP.plot([cell.xstart[idx], cell.xend[idx]],
                 [cell.zstart[idx], cell.zend[idx]], lw=2, c='b')
     for idx in range(cell.totnsegs)]

    ax_LFP.plot(cell.xmid[stim_idx], cell.zmid[stim_idx], '*', c='y', ms=14, mec='k')

    ep_intervals = ax_LFP.contourf(grid_x, grid_z, LFP,
                                   zorder=-2, colors=colors_from_map,
                                   levels=levels_norm, extend='both')

    ax_LFP.contour(grid_x, grid_z, LFP, colors='k', linewidths=(1), zorder=-2,
                   levels=levels_norm)

    ax_stim.plot(tvec_dict[freq], syni_dict[freq])
    ax_stim.axvline(tvec_dict[freq][max_idx_dict[freq]], ls='--', c='gray')

    tstop = 1000. / freq

    ax_stim.plot([tstop/2, tstop], [-0.01, -0.01], lw=2, c='k')
    dur_marker = "{:1.1f} ms".format(tstop/2) if tstop/2 < 1 else "{:d} ms".format(int(tstop/2))
    ax_stim.text(tstop - tstop/4, -0.02, dur_marker, ha='center', va='top',
                 fontsize=12)
    ax_stim.plot([tstop * 1.01, tstop * 1.01], [0, 0.1], lw=2, c='k')
    ax_stim.text(tstop * 1.05, 0.05, "0.1 nA", ha='left', fontsize=12)
    plotting_convention.mark_subplots(ax_stim, "ABC"[i], xpos=0.1)

cbar = fig.colorbar(ep_intervals, cax=cax, orientation='horizontal')

cbar.set_label('$\phi$ ($\mu$V)', labelpad=0)
cbar.set_ticks([-10, -1, -0.1, 0.1, 1, 10, 100])


plt.savefig(join(fig_folder, "intrinsic_dend_filt.png"), dpi=100)
plt.savefig(join(fig_folder, "intrinsic_dend_filt.pdf"), dpi=300)
