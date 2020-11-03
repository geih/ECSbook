
import os
from os.path import join
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import neuron
from neuron import h
import LFPy
from plotting_convention import mark_subplots, simplify_axes

def return_freq_and_psd(tvec, sig):
    """ Returns the power and freqency of the input signal"""
    import scipy.fftpack as ff
    sig = np.array(sig)
    if len(sig.shape) == 1:
        sig = np.array([sig])
    elif len(sig.shape) == 2:
        pass
    else:
        raise RuntimeError("Not compatible with given array shape!")
    timestep = (tvec[1] - tvec[0])/1000. if type(tvec) in [list, np.ndarray] else tvec
    sample_freq = ff.fftfreq(sig.shape[1], d=timestep)
    pidxs = np.where(sample_freq >= 0)
    freqs = sample_freq[pidxs]

    Y = ff.fft(sig, axis=1)[:, pidxs[0]]

    power = np.abs(Y)**2/Y.shape[1]
    return freqs, power


def make_WN_input(cell, max_freq):
    """ White Noise input ala Linden 2010 is made """
    tot_ntsteps = round((cell.tstop - cell.tstart) / cell.dt + 1)
    I = np.zeros(tot_ntsteps)
    tvec = np.arange(tot_ntsteps) * cell.dt
    for freq in range(1, max_freq + 1):
        I += np.sin(2 * np.pi * freq * tvec/1000. + 2*np.pi*np.random.random())
    return I


def make_white_noise_stimuli(cell, input_idx, weight=None):

    input_scaling = 0.005
    max_freq = 1000
    np.random.seed(1234)
    input_array = input_scaling * (make_WN_input(cell, max_freq))
    print(1000 * np.std(input_array))

    noise_vec = (neuron.h.Vector(input_array) if weight is None
                 else neuron.h.Vector(input_array * weight))

    i = 0
    syn = None
    for sec in cell.allseclist:
        for seg in sec:
            if i == input_idx:
                print("Input inserted in ", sec.name())
                syn = neuron.h.ISyn(seg.x, sec=sec)
                # print "Dist: ", nrn.distance(seg.x)
            i += 1
    if syn is None:
        raise RuntimeError("Wrong stimuli index")
    syn.dur = 1E9
    syn.delay = 0
    noise_vec.play(syn._ref_amp, cell.dt)
    return cell, syn, noise_vec


def simulate_white_noise_synapse(elec_x_pos):

    tstop = 1000
    dt = 2**-6
    cell = return_cell(tstop, dt)
    num_tsteps = int(tstop / dt + 1)
    t = np.arange(num_tsteps) * dt
    stim_idx = 0
    cell, syn, noise_vec = make_white_noise_stimuli(cell, stim_idx)
    cell.simulate(rec_vmem=True, rec_imem=True)
    syn_i = np.array(noise_vec)

    sigma = 0.3
    # Define electrode parameters
    lateral_elec_params = {
        'sigma': sigma,      # extracellular conductivity
        'x': np.array(elec_x_pos),  # electrode positions
        'y': np.zeros(2),
        'z': np.zeros(2),
        'method': 'linesource'
    }
    lateral_electrode = LFPy.RecExtElectrode(cell, **lateral_elec_params)
    lateral_electrode.calc_lfp()

    lat_LFP = lateral_electrode.LFP
    freq_LFP, LFP_psd = return_freq_and_psd(cell.tvec, lat_LFP)
    freq_syn, syn_i_psd = return_freq_and_psd(cell.tvec, syn_i[-len(cell.tvec):])

    return freq_LFP, LFP_psd, syn_i_psd

def return_cell(tstop, dt):

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
                'dt': dt,
                'tstart': 0.,
                'tstop': tstop,
            }
    cell = LFPy.Cell(**cell_params)
    cell.set_pos(x=-cell.xstart[0])
    return cell

frequencies = np.array([1, 1000])
fig_folder = os.path.abspath('.')

grid_LFP_dict = {}
lat_LFP_dict = {}
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
grid_elec_params = {
    'sigma': sigma,      # extracellular conductivity
    'x': grid_x.flatten(),  # electrode positions
    'y': grid_y.flatten(),
    'z': grid_z.flatten(),
    'method': 'linesource'
}

# Define electrode parameters
lateral_elec_params = {
    'sigma': sigma,      # extracellular conductivity
    'x': np.linspace(2, 1000, 100),  # electrode positions
    'y': np.zeros(100),
    'z': np.zeros(100),
    'method': 'linesource'
}


for freq in frequencies:

    tstop = 1000. / freq
    cell = return_cell(tstop, dt=tstop / 500)

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
    grid_electrode = LFPy.RecExtElectrode(cell, **grid_elec_params)
    grid_electrode.calc_lfp(t_indices=max_idx)

    lateral_electrode = LFPy.RecExtElectrode(cell, **lateral_elec_params)
    lateral_electrode.calc_lfp(t_indices=max_idx)

    imem_dict[freq] = cell.imem[:, max_idx]
    somav_dict[freq] = [cell.tvec.copy(), cell.vmem[0].copy()]
    grid_LFP_dict[freq] = 1000 * grid_electrode.LFP[:].reshape(grid_x.shape)
    lat_LFP_dict[freq] = 1000 * lateral_electrode.LFP[:]
    syni_dict[freq] = syn.i
    max_idx_dict[freq] = max_idx
    tvec_dict[freq] = cell.tvec
    cell.__del__()

elec_x_pos = np.array([10, 1000])
freq_LFP, LFP_psd, syn_i_psd = simulate_white_noise_synapse(elec_x_pos)

num = 11
levels = np.logspace(-2.5, 0, num=num)

scale_max = 10

levels_norm = scale_max * np.concatenate((-levels[::-1], levels))
rainbow_cmap = plt.cm.get_cmap('PRGn')  # rainbow, spectral, RdYlBu

colors_from_map = [rainbow_cmap(i*np.int(255/(len(levels_norm) - 2)))
                   for i in range(len(levels_norm) -1)]
colors_from_map[num - 1] = (1.0, 1.0, 1.0, 1.0)

fig = plt.figure(figsize=[5, 4])
fig.subplots_adjust(bottom=0.15, top=0.93, right=0.88, left=0.05, wspace=0.5,
                    hspace=0.05)

cax = fig.add_axes([0.04, 0.125, 0.5, 0.02], frameon=False)

ax_dict = {}

ax_dec = fig.add_axes([0.7, 0.7, 0.25, 0.17], xticks=[1, 10, 100], zorder=10,
                      xticklabels=["1", "10", "100"],
                      xlabel="distance (µm)", #ylabel="µV",
                      xlim=[1, 500], ylim=[1e-3, 2e1],
                      yscale="log", xscale="log")

ax_wn = fig.add_axes([0.7, 0.14, 0.25, 0.17], xticks=[1, 10, 100],
                      xticklabels=["1", "10", "100"],
                      xlabel="frequency (Hz)", #ylabel=,
                      xlim=[1, 500], ylim=[1e-3, 2e1],
                      yscale="log", xscale="log")

ax_wn.set_ylabel("norm. PSD", labelpad=-3)
ax_dec.set_ylabel("µV", labelpad=-3)

ax_dec.grid(True)

ax_wn.grid(True)

l1, = ax_wn.plot(freq_LFP, LFP_psd[0] / LFP_psd[0, 1])
l2, = ax_wn.plot(freq_LFP, LFP_psd[1] / LFP_psd[1, 1])
l3, = ax_wn.plot(freq_LFP, syn_i_psd[0] / syn_i_psd[0][1])

fig.legend([l3, l1, l2],
           ["WN stim",
            "%d µm" % elec_x_pos[0],
            "%d µm" % elec_x_pos[1]],
           frameon=False, loc=[0.72, 0.3], fontsize=12)

lines = []
line_names = []
for i, freq in enumerate(frequencies):

    imem = imem_dict[freq]
    LFP = grid_LFP_dict[freq]
    ax_LFP = fig.add_axes([0.05 + i * 0.3, 0.1, 0.2, 0.8], aspect=1,
                          frameon=False,  xticks=[], yticks=[],
                         ylim=[np.min(grid_electrode.z),
                               np.max(grid_electrode.z)],
                          xlim=[np.min(grid_electrode.x),
                                np.max(grid_electrode.x)])

    ax_stim = fig.add_axes([0.01 + i * 0.3, 0.77, 0.12, 0.1],
                           frameon=False, xticks=[], yticks=[])
    ax_stim.set_title("{:d} Hz".format(freq))

    [ax_LFP.plot([cell.xstart[idx], cell.xend[idx]],
                 [cell.zstart[idx], cell.zend[idx]], lw=2, c='b')
     for idx in range(cell.totnsegs)]

    ax_LFP.plot(cell.xmid[stim_idx], cell.zmid[stim_idx], '*',
                c='y', ms=14, mec='k')

    ep_intervals = ax_LFP.contourf(grid_x, grid_z, LFP,
                                   zorder=-2, colors=colors_from_map,
                                   levels=levels_norm, extend='both')

    ax_LFP.contour(grid_x, grid_z, LFP, colors='k', linewidths=(1), zorder=-2,
                   levels=levels_norm)

    ax_stim.plot(tvec_dict[freq], syni_dict[freq])
    ax_stim.axvline(tvec_dict[freq][max_idx_dict[freq]], ls='--', c='gray')

    tstop = 1000. / freq

    if i == 1:
        ax_LFP.plot([-200, -200], [50, 250], c='k', lw=2)
        ax_LFP.text(-220, 100, "200 µm", ha='right')

    ax_stim.plot([tstop/2 + tstop/10, tstop + tstop/10], [-0.02, -0.02], lw=2, c='k')
    dur_marker = "{:1.1f} ms".format(tstop/2) if tstop/2 < 1 else "{:d} ms".format(int(tstop/2))
    ax_stim.text(tstop - tstop/2, -0.05, dur_marker, ha='left', va='top',
                 fontsize=12)
    ax_stim.plot([tstop * 1.1, tstop * 1.1], [-0.01, 0.09], lw=2, c='k')
    ax_stim.text(tstop * 1.15, 0.02, "0.1 nA", ha='left', fontsize=12)
    mark_subplots(ax_stim, "BC"[i], xpos=0.1, ypos=2)

    l, = ax_dec.plot(lateral_elec_params["x"], np.abs(lat_LFP_dict[freq]),
                     c='k', lw=2, ls="-:"[i])
    lines.append(l)
    line_names.append("{:d} Hz".format(freq))


mark_subplots(ax_dec, "D")
mark_subplots(ax_wn, "E")
fig.legend(lines, line_names, frameon=False, loc=(0.73, 0.85), fontsize=12)
simplify_axes([ax_dec, ax_wn])
cbar = fig.colorbar(ep_intervals, cax=cax, orientation='horizontal')

cbar.set_label('$\phi$ (µV)', labelpad=0)
cbar.set_ticks([-10, -1, -0.1, 0.1, 1, 10, 100])

plt.savefig(join(fig_folder, "intrinsic_dend_filt.png"), dpi=100)
plt.savefig(join(fig_folder, "intrinsic_dend_filt.pdf"), dpi=300)
