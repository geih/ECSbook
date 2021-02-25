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
from ECSbook_simcode.plotting_convention import mark_subplots, simplify_axes
from ECSbook_simcode.neural_simulations import return_hay_cell
import ECSbook_simcode.neural_simulations as ns

neuron.load_mechanisms(ns.cell_models_folder)

np.random.seed(1534)
# Create a grid of measurement locations, in (um)

sigma = 0.3

num_elecs = 1
# Define electrode parameters
elec_params = {
    'sigma': sigma,      # extracellular conductivity
    'x': np.array([200]),  # electrode positions
    'y': np.zeros(num_elecs),
    'z': np.array([0]),
    'method': 'pointsource'
}

synapse_params = {
    'syntype' : 'Exp2Syn',      #conductance based exponential synapse
    'e': 0.,
    'tau1' : 0.1,                #Time constant, rise
    'tau2' : 1,                #Time constant, rise
    'weight' : 0.0005,           #Synaptic weight
    'record_current' : False,    #record synaptic currents
}


def insert_synapses(cell, synapse_params, syn_idxs,  spiketimes):
    ''' Find n compartments to insert synapses onto '''

    for num, idx in enumerate(syn_idxs):
        synapse_params.update({'idx' : idx})
        s = LFPy.Synapse(cell, **synapse_params)
        s.set_spike_times(np.array([spiketimes[num]]))


def return_up_down_states_spiketimes(dur, tvec, num_spikes):

    num_background_spikes = int(num_spikes / 4)
    num_state_spikes = num_spikes - num_background_spikes
    num_repeats = int(tvec[-1] / dur)
    num_spikes_per_repeat = int(num_state_spikes / num_repeats)
    spiketimes = []

    for idx in range(num_repeats):
        t0 = idx * dur
        t_mid = (idx + 0.5) * dur
        t1 = (idx + 1) * dur
        spiketimes.extend(np.random.uniform(t0, t_mid, num_spikes_per_repeat))
        # print(t0, t1)
    # print(spiketimes)

    spiketimes.extend(np.random.uniform(0, tvec[-1], num_background_spikes))
    if len(spiketimes) != num_spikes:
        print(len(spiketimes), num_spikes)
        raise RuntimeError("Wrong spike count!")

    return spiketimes


def return_wave_spiketimes(tvec, num_spikes):

    num_background_spikes = int(num_spikes / 4)
    num_state_spikes = num_spikes - num_background_spikes

    spiketimes = []

    spiketimes.extend(np.random.uniform(0, tvec[-1], num_background_spikes))
    spiketimes.extend(np.random.normal(tvec[-1] / 2, 30, num_state_spikes))

    if len(spiketimes) != num_spikes:
        raise RuntimeError("Wrong spike count!")

    return spiketimes


def make_input_dynamics_figure():

    num_syns = 4000
    tstop = 5000
    dt = 2**-4
    num_tsteps = int(tstop / dt + 1)
    tvec = np.arange(num_tsteps) * dt
    divide_into_welch = 8
    welch_dict = {'Fs': 1000 / dt,
                  'NFFT': int(num_tsteps/divide_into_welch),
                  'noverlap': int(num_tsteps/divide_into_welch/2.),
                  # 'window': np.window_hanning,
                  'detrend': 'mean',
                  'scale_by_freq': True,
                  }

    cell = return_hay_cell(tstop=tstop, dt=dt, make_passive=True)   
    syn_idxs = cell.get_rand_idx_area_norm(section="allsec", nidx=num_syns,
                                       z_min=-100, z_max=100)
    cell.__del__()
    spiketime_dict = {}
    depth_clrs = {}

    # spiketime_dict["wave"] = return_wave_spiketimes(tvec, num_syns)
    # depth_clrs["wave"] = 'r'

    spiketime_dict["up_down_states"] = return_up_down_states_spiketimes(500,
                                                        tvec, num_syns)
    depth_clrs["up_down_states"] = 'gray'

    spiketime_dict["stationary"] = np.random.uniform(0, tstop, num_syns)
    depth_clrs["stationary"] = 'k'

    legend_dict = {"stationary": "stationary",
                   "up_down_states": "up-and-down states"}

    # depth_clrs = {spiketimes: plt.cm.rainbow(i / (len(spiketime_dict)))
    #               for i, spiketimes in enumerate(spiketime_dict)}

    plt.close("all")
    fig = plt.figure(figsize=[6, 3])

    ax_morph = fig.add_axes([0.01, 0.01, 0.12, 0.98], aspect=1, xticks=[],
                            yticks=[], frameon=False)
    ax_fr = fig.add_axes([0.27, 0.67, 0.25, 0.22], xlim=[0, tstop], xlabel="time (ms)", ylabel="#",
                            title="synaptic input")
    ax_lfp = fig.add_axes([0.27, 0.17, 0.25, 0.22], xlim=[0, tstop], xlabel="time (ms)", ylabel="µV", title="LFP")
    ax_psd = fig.add_axes([0.7, 0.17, 0.25, 0.65], xlabel="Hz", ylabel="µV²/Hz",
                          xlim=[1, 5000], ylim=[1e-8, 1e-3])
    lfp_dict = {}
    psd_dict = {}

    for i, (name, spiketimes) in enumerate(spiketime_dict.items()):
        # print(i, spiketimes)

        cell = return_hay_cell(tstop=tstop, dt=dt, make_passive=True)
        # for sec in cell.allseclist:
        #     sec.cm = 2.0
        insert_synapses(cell, synapse_params, syn_idxs, spiketimes)
        cell.simulate(rec_imem=True, rec_vmem=True)
        print("Max vmem: ", np.max(cell.vmem))

        electrode = LFPy.RecExtElectrode(cell, **elec_params)
        lfp = electrode.get_transformation_matrix() @ cell.imem * 1000
        print("Max LFP: ", np.max(np.abs(lfp)))

        cell.__del__()
        lfp_dict[name] = lfp
        freq, psd = ns.return_freq_and_psd_welch(lfp, welch_dict)
        psd_dict[name] = [freq, psd]

    cell = return_hay_cell(tstop=tstop, dt=dt, make_passive=True)
    ax_morph.plot(cell.x.T,
                   cell.z.T, c='k', lw=1, zorder=-1)

    ax_morph.plot(cell.x.mean(axis=1)[syn_idxs], cell.z.mean(axis=1)[syn_idxs],
                  'go', ms=1)
    ax_morph.plot(elec_params["x"], elec_params["z"], 'rD')
    # lfp_normalize = np.max([np.max(np.abs(lfp)) for lfp in lfp_dict.values()])

    lines = []
    line_names = []
    for i, name in enumerate(spiketime_dict):

        ax_fr.hist(spiketime_dict[name], bins=100, range=[0, tstop], fc=depth_clrs[name])
        ax_psd.loglog(psd_dict[name][0], psd_dict[name][1][0], c=depth_clrs[name])
        l, = ax_lfp.plot(tvec, lfp_dict[name][0], c=depth_clrs[name])
        lines.append(l)
        line_names.append(legend_dict[name])

    fig.legend(lines, line_names, frameon=False, ncol=1, loc=(0.6, 0.8))

    mark_subplots(ax_morph, "A", xpos=0.1, ypos=1.0)
    mark_subplots([ax_fr, ax_lfp], "BC")
    mark_subplots(ax_psd, "D", ypos=0.9)
    simplify_axes(fig.axes)
    ax_psd.set_xticks([1, 10, 100, 1000])

    fig.savefig("fig_LFP_powerlaws_input_dynamics_.png", dpi=300)
    # plt.show()


def make_cm_figure():

    num_syns = 4000
    tstop = 4000
    dt = 2**-4
    num_tsteps = int(tstop / dt + 1)
    tvec = np.arange(num_tsteps) * dt
    divide_into_welch = 8
    welch_dict = {'Fs': 1000 / dt,
                  'NFFT': int(num_tsteps/divide_into_welch),
                  'noverlap': int(num_tsteps/divide_into_welch/2.),
                  # 'window': np.window_hanning,
                  'detrend': 'mean',
                  'scale_by_freq': True,
                  }

    cell = return_hay_cell(tstop=tstop, dt=dt, make_passive=True)
    syn_idxs = cell.get_rand_idx_area_norm(section="allsec", nidx=num_syns,
                                           z_min=-100, z_max=100)
    cell.__del__()
    spiketime_dict = {}
    depth_clrs = {}

    # spiketime_dict["wave"] = return_wave_spiketimes(tvec, num_syns)
    # depth_clrs["wave"] = 'r'


    spiketimes = np.random.uniform(0, tstop, num_syns)

    c_ms = [0.5, 2]

    legend_dict = {0.5: "c$_m$ = 0.5 µF/cm²",
                   1: "c$_m$ = 1.0 µF/cm²",
                   2: "c$_m$ = 2.0 µF/cm²",
    }

    # depth_clrs = {spiketimes: plt.cm.rainbow(i / (len(spiketime_dict)))
    #               for i, spiketimes in enumerate(spiketime_dict)}

    depth_clrs = {0.5: 'k', 1: 'gray', 2:'blue'}

    plt.close("all")
    fig = plt.figure(figsize=[6, 3])

    ax_morph = fig.add_axes([0.01, 0.01, 0.12, 0.98], aspect=1, xticks=[],
                            yticks=[], frameon=False)
    ax_fr = fig.add_axes([0.27, 0.67, 0.25, 0.22], xlim=[0, tstop],
                         xlabel="time (ms)", ylabel="#",
                         title="synaptic input")
    ax_lfp = fig.add_axes([0.27, 0.17, 0.25, 0.22], xlim=[0, tstop],
                          xlabel="time (ms)", ylabel="µV", title="LFP")
    ax_psd = fig.add_axes([0.7, 0.17, 0.25, 0.65], xlabel="Hz", ylabel="µV²/Hz",
                          xlim=[1, 5000], ylim=[1e-10, 1e-5])
    lfp_dict = {}
    psd_dict = {}

    for i, c_m in enumerate(c_ms):
        # print(i, spiketimes)

        cell = return_hay_cell(tstop=tstop, dt=dt, make_passive=True)
        for sec in cell.allseclist:
            sec.cm = c_m
            # sec.g_pas = g_pass[i]
        insert_synapses(cell, synapse_params, syn_idxs, spiketimes)
        cell.simulate(rec_imem=True, rec_vmem=True)
        print("Max vmem: ", np.max(cell.vmem))

        electrode = LFPy.RecExtElectrode(cell, **elec_params)
        lfp = electrode.get_transformation_matrix() @ cell.imem * 1000
        print("Max LFP: ", np.max(np.abs(lfp)))

        cell.__del__()
        lfp_dict[c_m] = lfp
        freq, psd = ns.return_freq_and_psd_welch(lfp, welch_dict)
        psd_dict[c_m] = [freq, psd]

    cell = return_hay_cell(tstop=tstop, dt=dt, make_passive=True)
    ax_morph.plot(cell.x.T,
                  cell.z.T, c='k', lw=1, zorder=-1)

    ax_morph.plot(cell.x.mean(axis=1)[syn_idxs], cell.z.mean(axis=1)[syn_idxs],
                  'go', ms=1)
    ax_morph.plot(elec_params["x"], elec_params["z"], 'rD')
    # lfp_normalize = np.max([np.max(np.abs(lfp)) for lfp in lfp_dict.values()])

    lines = []
    line_names = []
    ax_fr.hist(spiketimes, bins=100, range=[0, tstop], fc='k')
    for i, name in enumerate(c_ms):

        ax_psd.loglog(psd_dict[name][0], psd_dict[name][1][0], c=depth_clrs[name])
        l, = ax_lfp.plot(tvec, lfp_dict[name][0], c=depth_clrs[name])
        lines.append(l)
        line_names.append(legend_dict[name])

    fig.legend(lines, line_names, frameon=False, ncol=1, loc=(0.6, 0.8))

    mark_subplots(ax_morph, "A", xpos=0.1, ypos=1.0)
    mark_subplots([ax_fr, ax_lfp], "BC")
    mark_subplots(ax_psd, "D", ypos=0.9)
    simplify_axes(fig.axes)
    ax_psd.set_xticks([1, 10, 100, 1000])

    fig.savefig("fig_LFP_powerlaws_cm.png", dpi=300)
    # plt.show()

def make_syn_tau_figure():

    num_syns = 4000
    tstop = 4000
    dt = 2**-4
    num_tsteps = int(tstop / dt + 1)
    tvec = np.arange(num_tsteps) * dt
    divide_into_welch = 8
    welch_dict = {'Fs': 1000 / dt,
                  'NFFT': int(num_tsteps/divide_into_welch),
                  'noverlap': int(num_tsteps/divide_into_welch/2.),
                  # 'window': np.window_hanning,
                  'detrend': 'mean',
                  'scale_by_freq': True,
                  }

    cell = return_hay_cell(tstop=tstop, dt=dt, make_passive=True)
    syn_idxs = cell.get_rand_idx_area_norm(section="allsec", nidx=num_syns,
                                           z_min=-100, z_max=100)
    cell.__del__()
    spiketime_dict = {}
    depth_clrs = {}

    # spiketime_dict["wave"] = return_wave_spiketimes(tvec, num_syns)
    # depth_clrs["wave"] = 'r'


    spiketimes = np.random.uniform(0, tstop, num_syns)

    tau_ss = [1, 2]

    legend_dict = {0.1: r"$\tau_s$ = 0.1 ms",
                   1: r"$\tau_s$ = 1 ms",
                   2: r"$\tau_s$ = 2 ms",
                   10: r"$\tau_s$ = 10.0 ms",
    }

    # depth_clrs = {spiketimes: plt.cm.rainbow(i / (len(spiketime_dict)))
    #               for i, spiketimes in enumerate(spiketime_dict)}

    depth_clrs = {1: 'k', 2 : 'gray',
                  10:'blue'}

    plt.close("all")
    fig = plt.figure(figsize=[6, 3])

    ax_morph = fig.add_axes([0.01, 0.01, 0.12, 0.98], aspect=1, xticks=[],
                            yticks=[], frameon=False)
    ax_fr = fig.add_axes([0.27, 0.67, 0.25, 0.22], xlim=[0, tstop],
                         xlabel="time (ms)", ylabel="#",
                         title="synaptic input")
    ax_lfp = fig.add_axes([0.27, 0.17, 0.25, 0.22], xlim=[0, tstop],
                          xlabel="time (ms)", ylabel="µV", title="LFP")
    ax_psd = fig.add_axes([0.7, 0.17, 0.25, 0.65], xlabel="Hz", ylabel="µV²/Hz",
                          xlim=[1, 5000], ylim=[1e-8, 1e-4])
    lfp_dict = {}
    psd_dict = {}

    for i, tau_s in enumerate(tau_ss):
        print(i, tau_s)

        cell = return_hay_cell(tstop=tstop, dt=dt, make_passive=True)
        synapse_params["tau2"] = tau_s
        # for sec in cell.allseclist:
        #     sec.cm = c_m
            # sec.g_pas = g_pass[i]
        insert_synapses(cell, synapse_params, syn_idxs, spiketimes)
        cell.simulate(rec_imem=True, rec_vmem=True)
        print("Max vmem: ", np.max(cell.vmem))

        electrode = LFPy.RecExtElectrode(cell, **elec_params)
        lfp = electrode.get_transformation_matrix() @ cell.imem * 1000
        print("Max LFP: ", np.max(np.abs(lfp)))

        cell.__del__()
        lfp_dict[tau_s] = lfp
        freq, psd = ns.return_freq_and_psd_welch(lfp, welch_dict)
        psd_dict[tau_s] = [freq, psd]

    cell = return_hay_cell(tstop=tstop, dt=dt, make_passive=True)
    ax_morph.plot(cell.x.T,
                  cell.z.T, c='k', lw=1, zorder=-1)

    ax_morph.plot(cell.x.mean(axis=1)[syn_idxs], cell.z.mean(axis=1)[syn_idxs],
                  'go', ms=1)
    ax_morph.plot(elec_params["x"], elec_params["z"], 'rD')
    # lfp_normalize = np.max([np.max(np.abs(lfp)) for lfp in lfp_dict.values()])

    lines = []
    line_names = []
    ax_fr.hist(spiketimes, bins=100, range=[0, tstop], fc='k')
    for i, name in enumerate(tau_ss):

        ax_psd.loglog(psd_dict[name][0], psd_dict[name][1][0], c=depth_clrs[name])
        l, = ax_lfp.plot(tvec, lfp_dict[name][0], c=depth_clrs[name])
        lines.append(l)
        line_names.append(legend_dict[name])

    fig.legend(lines, line_names, frameon=False, ncol=1, loc=(0.6, 0.8))

    mark_subplots(ax_morph, "A", xpos=0.1, ypos=1.0)
    mark_subplots([ax_fr, ax_lfp], "BC")
    mark_subplots(ax_psd, "D", ypos=0.9)
    simplify_axes(fig.axes)
    ax_psd.set_xticks([1, 10, 100, 1000])

    fig.savefig("fig_LFP_powerlaws_tau_s.png", dpi=300)
    # plt.show()


def make_intrinsic_comp_hay_bns_2c_figure():

    tstop = 1000
    dt = 2**-5

    stim_idx = 0

    cell_model_dict = {}
    depth_clrs = {}

    cell_model_dict["hay"] = ns.return_hay_cell
    depth_clrs["hay"] = 'gray'

    cell_model_dict["ball_and_stick"] = ns.return_ball_and_stick_cell
    depth_clrs["ball_and_stick"] = 'k'

    cell_model_dict["two_compartment"] = ns.return_two_comp_cell
    depth_clrs["two_compartment"] = 'r'

    legend_dict = {"hay": "reconstructed morphology",
                   "ball_and_stick": "ball and stick",
                   "two_compartment": "two-compartment"}

    # depth_clrs = {return_cell_func: plt.cm.rainbow(i / (len(cell_model_dict)))
    #               for i, return_cell_func in enumerate(cell_model_dict)}

    plt.close("all")
    fig = plt.figure(figsize=[4, 4])
    fig.subplots_adjust(left=0.2, bottom=0.15)

    # ax_morph = fig.add_axes([0.01, 0.01, 0.12, 0.98], aspect=1, xticks=[],
    #                         yticks=[], frameon=False)
    # ax_fr = fig.add_axes([0.27, 0.67, 0.25, 0.22], xlim=[0, tstop], xlabel="time (ms)", ylabel="#",
    #                         title="synaptic input")
    # ax_lfp = fig.add_axes([0.27, 0.17, 0.25, 0.22], xlim=[0, tstop], xlabel="time (ms)", ylabel="µV", title="LFP")
    ax_psd = fig.add_subplot(111, xlabel="frequency (Hz)", ylabel="normalized LFP (µV²/Hz)",
                          xlim=[1, 1000], ylim=[1e-3, 4e0])
    lfp_dict = {}
    psd_dict = {}

    for i, (name, return_cell_func) in enumerate(cell_model_dict.items()):
        # print(i, return_cell_func)

        cell = return_cell_func(tstop=tstop, dt=dt)#, make_passive=True)
        cell, syn, noise_vec = ns.make_white_noise_stimuli(cell, stim_idx)

        cell.simulate(rec_imem=True, rec_vmem=True)
        print("Max vmem: ", np.max(cell.vmem))

        electrode = LFPy.RecExtElectrode(cell, **elec_params)
        lfp = electrode.get_transformation_matrix() @ cell.imem * 1000
        print("Max LFP: ", np.max(np.abs(lfp)))

        lfp_dict[name] = lfp
        # freq, psd = ns.return_freq_and_psd_welch(lfp, welch_dict)
        freq, psd = ns.return_freq_and_psd(cell.tvec, lfp)
        psd_dict[name] = [freq, psd]
        cell.__del__()


    lines = []
    line_names = []
    for i, name in enumerate(cell_model_dict):

        # ax_fr.hist(cell_model_dict[name], bins=100, range=[0, tstop], fc=depth_clrs[name])
        l, = ax_psd.loglog(psd_dict[name][0], psd_dict[name][1][0] / psd_dict[name][1][0][1], c=depth_clrs[name])
        # l, = ax_lfp.plot(tvec, lfp_dict[name][0], c=depth_clrs[name])
        lines.append(l)
        line_names.append(legend_dict[name])

    fig.legend(lines, line_names, frameon=False, ncol=1, loc=(0.25, 0.8))

    # mark_subplots(ax_morph, "A", xpos=0.1, ypos=1.0)
    # mark_subplots([ax_fr, ax_lfp], "BC")
    # mark_subplots(ax_psd, "D", ypos=0.9)
    simplify_axes(fig.axes)
    ax_psd.set_xticks([1, 10, 100, 1000])

    fig.savefig("fig_LFP_powerlaws_morphology.png", dpi=300)
    # plt.show()

if __name__ == '__main__':
    # make_input_dynamics_figure()
    # make_cm_figure()
    # make_syn_tau_figure()
    make_intrinsic_comp_hay_bns_2c_figure()