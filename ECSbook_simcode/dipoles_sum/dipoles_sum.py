import os
import sys
from os.path import join
import numpy as np
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import neuron
from neuron import h
import LFPy
from plotting_convention import mark_subplots, simplify_axes

np.random.seed(12345)

# Create a grid of measurement locations, in (um)
grid_x, grid_z = np.mgrid[-500:501:20, -620:1400:20]
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

ax_lfp_dict = dict(aspect=1, frameon=False, xticks=[], yticks=[],
                   ylim=[np.min(grid_z), np.max(grid_z)],
                   xlim=[np.min(grid_x), np.max(grid_x)])

def return_cell(tstop, dt):
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
    cellParameters = {
        'morphology': 'L5bPCmodelsEH/morphologies/cell1.asc',
        'passive': True,
        'nsegs_method': "lambda_f",
        "lambda_f": 100,
        'dt': dt,
        'tstart': -1,
        'tstop': tstop,
        'v_init': -70,
    }

    cell = LFPy.Cell(**cellParameters)
    cell.set_rotation(x=4.729, y=-3.166)

    return cell


def insert_synaptic_input(idx, cell):

    synapse_parameters = {'e': 0., # reversal potential
                          'weight': 0.002, # 0.001, # synapse weight
                          'record_current': True, # record synapse current
                          'syntype': 'Exp2Syn',
                          'tau1': 0.1, #Time constant, rise
                          'tau2': 0.1, #Time constant, decay
                          }
    synapse_parameters['idx'] = idx
    synapse = LFPy.Synapse(cell, **synapse_parameters)
    synapse.set_spike_times(np.array([1.]))
    return synapse, cell


def plot_grid_LFP(cell, grid_elec_params, ax, synapses):

    grid_electrode = LFPy.RecExtElectrode(cell, **grid_elec_params)
    grid_electrode.calc_lfp()

    max_amp_elec_idx = np.argmax(np.max(np.abs(grid_electrode.LFP), axis=1))
    max_amp_t_idx = np.argmax(np.abs(grid_electrode.LFP[max_amp_elec_idx, :]))

    max_amp_LFP = np.max(np.abs(grid_electrode.LFP))
    if not max_amp_LFP == np.abs(grid_electrode.LFP[max_amp_elec_idx, max_amp_t_idx]):
        raise RuntimeError("Wrong with chosen max value")

    LFP = 1000 * grid_electrode.LFP[:, max_amp_t_idx].reshape(grid_x.shape)

    num = 11
    levels = np.logspace(-2.0, 0, num=num)

    scale_max = np.max(np.abs(LFP))

    levels_norm = scale_max * np.concatenate((-levels[::-1], levels))
    rainbow_cmap = plt.cm.get_cmap('PRGn')  # rainbow, spectral, RdYlBu

    colors_from_map = [rainbow_cmap(i*np.int(255/(len(levels_norm) - 2)))
                       for i in range(len(levels_norm) -1)]
    colors_from_map[num - 1] = (1.0, 1.0, 1.0, 1.0)

    [ax.plot([cell.xstart[idx], cell.xend[idx]],
                 [cell.zstart[idx], cell.zend[idx]], lw=1, c='gray')
     for idx in range(cell.totnsegs)]

    [ax.plot(cell.xmid[syn.idx], cell.zmid[syn.idx], marker='o', c='cyan',
                 ms=3, mec='k')
     for syn in synapses]

    ep_intervals = ax.contourf(grid_x, grid_z, LFP,
                                   zorder=2, colors=colors_from_map,
                                   levels=levels_norm, extend='both')

    ax.contour(grid_x, grid_z, LFP, colors='k', linewidths=(1), zorder=2,
                   levels=levels_norm)


def make_all_basal_dipoles():

    basal_dipole_fig_folder = join('.', "all_basal_dipoles")
    os.makedirs(basal_dipole_fig_folder, exist_ok=True)
    cell = return_cell(tstop=5, dt=2**-4)
    idxs = cell.get_rand_idx_area_norm(section='allsec', z_max=100,
                                       z_min=-1e9, nidx=1000)
    cell.__del__()

    for syn_idx in set(idxs):
        synapses = []
        print(syn_idx)
        cell = return_cell(tstop=5, dt=2**-4)
        syn, cell = insert_synaptic_input(syn_idx, cell)
        synapses.append(syn)
        cell.simulate(rec_imem=True)

        plt.close("all")
        fig = plt.figure(figsize=[3, 4])

        ax = fig.add_axes([0.01, 0.01, 0.9, 0.9], **ax_lfp_dict)
        plot_grid_LFP(cell, grid_elec_params, ax, synapses)

        fig.savefig(join(basal_dipole_fig_folder, "fig_dipole_%d.png" % syn_idx), dpi=100)
        cell.__del__()


def make_figure():
    chosen_basal_idxs = np.array([129, 184, 95]) # 118 221,

    plt.close("all")
    fig = plt.figure(figsize=[4, 5])
    fig.subplots_adjust(bottom=0.01, top=0.98, right=0.99,
                        left=0.0, wspace=-0.0, hspace=0.0)
    num_cols = 3

    # Plot single-synapse dipoles
    for i, syn_idx in enumerate(chosen_basal_idxs):
        synapses = []
        cell = return_cell(tstop=5, dt=2**-4)
        syn, cell = insert_synaptic_input(syn_idx, cell)
        synapses.append(syn)
        cell.simulate(rec_imem=True)

        ax = fig.add_subplot(2, num_cols, i + 1, **ax_lfp_dict)

        plot_grid_LFP(cell, grid_elec_params, ax, synapses)

        cell.__del__()

    # Plot compound basal input
    cell = return_cell(tstop=5, dt=2**-4)
    idxs = cell.get_rand_idx_area_norm(section='allsec', z_max=100,
                                       z_min=-1e9, nidx=1000)
    synapses = []
    for syn_idx in idxs:
        syn, cell = insert_synaptic_input(syn_idx, cell)
        synapses.append(syn)
    cell.simulate(rec_imem=True)

    ax = fig.add_subplot(2, num_cols, 4, **ax_lfp_dict)

    plot_grid_LFP(cell, grid_elec_params, ax, synapses)
    cell.__del__()

    # Plot compound apical input
    cell = return_cell(tstop=5, dt=2**-4)
    idxs = cell.get_rand_idx_area_norm(section='allsec', z_max=1e9,
                                       z_min=700, nidx=1000)
    synapses = []
    for syn_idx in idxs:
        syn, cell = insert_synaptic_input(syn_idx, cell)
        synapses.append(syn)
    cell.simulate(rec_imem=True)

    ax = fig.add_subplot(2, num_cols, 5, **ax_lfp_dict)

    plot_grid_LFP(cell, grid_elec_params, ax, synapses)
    cell.__del__()

    # Plot compound uniform input
    cell = return_cell(tstop=5, dt=2**-4)
    idxs = cell.get_rand_idx_area_norm(section='allsec', z_max=1e9,
                                       z_min=-1e9, nidx=1000)
    synapses = []
    for syn_idx in idxs:
        syn, cell = insert_synaptic_input(syn_idx, cell)
        synapses.append(syn)
    cell.simulate(rec_imem=True)

    ax = fig.add_subplot(2, num_cols, 6, **ax_lfp_dict)

    plot_grid_LFP(cell, grid_elec_params, ax, synapses)
    cell.__del__()

    mark_subplots(fig.axes, xpos=0.15, ypos=0.95)

    fig.savefig("fig_chosen_dipoles.png", dpi=300)
    fig.savefig("fig_chosen_dipoles.pdf", dpi=300)


if __name__ == '__main__':
    # make_all_basal_dipoles()
    make_figure()

