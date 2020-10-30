
import os
from os.path import join
import sys
import numpy as np
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import neuron
from neuron import h
import LFPy
from plotting_convention import mark_subplots, simplify_axes


if not os.path.isfile('L5bPCmodelsEH/morphologies/cell1.asc'):
    print("Downloading Hay model")
    if sys.version < '3':
        from urllib2 import urlopen
    else:
        from urllib.request import urlopen
    import ssl
    from warnings import warn
    import zipfile
    #get the model files:
    u = urlopen('http://senselab.med.yale.edu/ModelDB/eavBinDown.asp?o=139653&a=23&mime=application/zip',
                context=ssl._create_unverified_context())
    localFile = open('L5bPCmodelsEH.zip', 'wb')
    localFile.write(u.read())
    localFile.close()
    #unzip:
    myzip = zipfile.ZipFile('L5bPCmodelsEH.zip', 'r')
    myzip.extractall('.')
    myzip.close()

    #compile mod files every time, because of incompatibility with Mainen96 files:
    if "win32" in sys.platform:
        pth = "L5bPCmodelsEH/mod/"
        warn("no autompile of NMODL (.mod) files on Windows.\n"
             + "Run mknrndll from NEURON bash in the folder L5bPCmodelsEH/mod and rerun example script")
        if not pth in neuron.nrn_dll_loaded:
            neuron.h.nrn_load_dll(pth+"nrnmech.dll")
        neuron.nrn_dll_loaded.append(pth)
    else:
        os.system('''
                  cd L5bPCmodelsEH/mod/
                  nrnivmodl
                  ''')
        neuron.load_mechanisms('L5bPCmodelsEH/mod/')

def return_cell(tstop, dt):
    cellParameters = {
        'morphology': 'L5bPCmodelsEH/morphologies/cell1.asc',
        # 'templatefile'  : ['L5bPCmodelsEH/models/L5PCbiophys3.hoc',
        #                    'L5bPCmodelsEH/models/L5PCtemplate.hoc'],
        # 'templatename'  : 'L5PCtemplate',
        # 'templateargs'  : 'L5bPCmodelsEH/morphologies/cell1.asc',
        'passive': True,
        'nsegs_method': None,
        'dt': dt,
        'tstart': -1,
        'tstop': tstop,
        'v_init': -60,
    }

    cell = LFPy.Cell(**cellParameters)
    cell.set_rotation(x=4.729, y=-3.166)

    return cell

frequencies = np.array([100])
fig_folder = os.path.abspath('.')

grid_LFP_dict = {}
lat_LFP_dict = {}
imem_dict = {}
somav_dict = {}
syni_dict = {}
max_idx_dict = {}
tvec_dict = {}
color_dict = {frequencies[0]: 'gray',
              }
sigma = 0.3


# Define electrode parameters
lateral_elec_params = {
    'sigma': sigma,      # extracellular conductivity
    'x': np.array([10, 1000]),  # electrode requires 1d vector of positions
    'y': np.zeros(2),
    'z': np.zeros(2),
    'method': 'linesource'
}

freq = 100

tstop = 100
dt = 2**-4
cell = return_cell(tstop, dt)
num_tsteps = int(tstop / dt + 1)
t = np.arange(num_tsteps) * dt

syn_list = []

for f in np.arange(1000):
    stim_idx = 0
    stim_params = {
                 'idx': stim_idx,
                 'record_current': True,
                 'syntype': 'SinSyn',
                 'del': 0.,
                 'dur': 1e9,
                 'phase': np.random.uniform() * 2 * np.pi,
                 'pkamp': 0.05,
                 'freq': f,
                 'weight': 1.0,  # not in use, but required for Synapse
                }

    syn_list.append(LFPy.Synapse(cell, **stim_params))


# i_noise.play(syn._ref_i, t)

cell.simulate(rec_vmem=True, rec_imem=True)

syn_i = np.sum(np.array([syn.i for syn in syn_list]), axis=0)


lateral_electrode = LFPy.RecExtElectrode(cell, **lateral_elec_params)
lateral_electrode.calc_lfp()

lat_LFP = 1000 * lateral_electrode.LFP

fig = plt.figure(figsize=[6, 4])
fig.subplots_adjust(bottom=0.15, top=0.93, right=0.88, left=0.07, wspace=0.5,
                    hspace=0.3)

ax_dict = {}

ax_dec = fig.add_axes([0.70, 0.33, 0.2, 0.4], xticks=[1, 10, 100],
                      xticklabels=["1", "10", "100"],
                      xlabel="distance ($\mu$m)", ylabel="µV",
                      yscale="log", xscale="log")
ax_dec.grid(True)

lines = []
line_names = []

ax_stim = fig.add_subplot(2, 4, 1, )
ax_vmem = fig.add_subplot(2, 4, 5, )
# ax_stim.set_title("{:d} Hz".format(freq), fontsize=16)

ax_LFP = fig.add_axes([0.2, 0.1, 0.3, 0.8], aspect=1,
                      frameon=False,  xticks=[], yticks=[],
                     ylim=[-300, 1300],
                      xlim=[-300, 300])

[ax_LFP.plot([cell.xstart[idx], cell.xend[idx]],
             [cell.zstart[idx], cell.zend[idx]], lw=2, c='k')
 for idx in range(cell.totnsegs)]

ax_LFP.plot(cell.xmid[stim_idx], cell.zmid[stim_idx], '*', c='y', ms=14, mec='k')

ax_stim.plot(cell.tvec, syn_i)
ax_vmem.plot(cell.tvec, cell.somav)

# ax_stim.axvline(tvec_dict[freq][max_idx_dict[freq]], ls='--', c='gray')

tstop = 1000. / freq

# ax_stim.plot([tstop/2, tstop], [-0.01, -0.01], lw=2, c='k')
# dur_marker = "{:1.1f} ms".format(tstop/2) if tstop/2 < 1 else "{:d} ms".format(int(tstop/2))
# ax_stim.text(tstop - tstop/4, -0.02, dur_marker, ha='center', va='top',
#              fontsize=12)
# ax_stim.plot([tstop * 1.01, tstop * 1.01], [0, 0.1], lw=2, c='k')
# ax_stim.text(tstop * 1.05, 0.05, "0.1 nA", ha='left', fontsize=12)

# mark_subplots(ax_stim, "BC"[i], xpos=0.0)

l, = ax_dec.plot(lateral_elec_params["x"], np.abs(lat_LFP),
                 c='k', lw=2, ls="-:"[0])
lines.append(l)
line_names.append("{:d} Hz".format(freq))

mark_subplots(ax_dec, "D")
fig.legend(lines, line_names, frameon=False, loc=(0.7, 0.73))
simplify_axes(ax_dec)

plt.savefig(join(fig_folder, "intrinsic_dend_filt_hay.png"), dpi=100)
# plt.savefig(join(fig_folder, "intrinsic_dend_filt_hay.pdf"), dpi=300)
