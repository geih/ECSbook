
import numpy as np
import matplotlib.pyplot as plt
import LFPy
import elephant
from ECSbook_simcode.plotting_convention import mark_subplots, simplify_axes
from ECSbook_simcode.neural_simulations import return_hay_cell

np.random.seed(13734)
dt = 2**-5

filt_dict_mua_lp = {'highpass_freq': None,
             'lowpass_freq': 50,
             'order': 4,
             'filter_function': 'filtfilt',
             'fs': 1 / dt * 1000,
             'axis': -1}

filt_dict_high_pass = {'highpass_freq': 300,
             'lowpass_freq': None,
             'order': 4,
             'filter_function': 'filtfilt',
             'fs': 1 / dt * 1000,
             'axis': -1}

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


def simulate_spikes_on_grid():

    tstop = 15
    cell = return_hay_cell(tstop=tstop, dt=dt, make_passive=False)
    syn, cell = insert_synaptic_input(cell)
    cell.simulate(rec_imem=True, rec_vmem=True)

    xmin, xmax = [-70, 70]
    ymin, ymax = [-70, 70]
    zmin, zmax = [-70, 70]

    dx = 10
    x_grid, y_grid, z_grid = np.mgrid[xmin:xmax+dx:dx, ymin:ymax+dx:dx, zmin:zmax+dx:dx]

    elec_grid_params = dict(
                sigma = 0.3,      # extracellular conductivity
                x = x_grid.flatten(),
                y = y_grid.flatten(),
                z = z_grid.flatten(),
                method = 'root_as_point',
            )

    elec = LFPy.RecExtElectrode(cell, **elec_grid_params)
    M_elec = elec.get_transformation_matrix()
    eaps = M_elec @ cell.imem

    # fig = plt.figure()
    # [plt.plot(cell.tvec, eap) for eap in eaps]
    # plt.show()

    return eaps


def insert_burst_of_spikes(sig, tvec, spike_burst_center, burst_width,
                           num_spikes_to_insert):
    eaps = simulate_spikes_on_grid()
    num_spikes_on_grid = eaps.shape[0]
    eap_len = eaps.shape[1]

    insert_times = np.random.uniform(spike_burst_center - burst_width/2,
                                    spike_burst_center + burst_width/2,
                                    size=num_spikes_to_insert)

    insert_time_idxs = np.array([np.argmin(np.abs(insert_time - tvec))
                            for insert_time in insert_times])
    chosen_eap_idxs = np.random.randint(0, num_spikes_on_grid,
                                                num_spikes_to_insert)

    for s_idx in range(num_spikes_to_insert):
        t0_idx = insert_time_idxs[s_idx] - int(eap_len / 2)
        t1_idx = t0_idx + eap_len
        sig[t0_idx:t1_idx] += eaps[chosen_eap_idxs[s_idx]]
    return sig, insert_times


def make_figure():

    raw_sig_tstop = 500
    num_spikes_to_insert = 1000
    spike_burst_center = 250  # ms
    burst_width = 50

    num_tsteps = int(raw_sig_tstop / dt + 1)
    tvec = np.arange(num_tsteps) * dt
    sig_raw = np.zeros(num_tsteps)
    sig_raw += np.cumsum(np.random.normal(0, 0.02, size=num_tsteps))
    sig_raw, spike_times = insert_burst_of_spikes(sig_raw, tvec,
                                     spike_burst_center, burst_width,
                                     num_spikes_to_insert)

    sig_hp = elephant.signal_processing.butter(sig_raw, **filt_dict_high_pass)
    sig_hp_rect = np.abs(sig_hp)
    sig_hp_rect_lp = elephant.signal_processing.butter(sig_hp_rect,
                                                       **filt_dict_mua_lp)

    plt.close("all")
    fig = plt.figure(figsize=[10, 3])
    fig.subplots_adjust(left=0.07, bottom=0.18, top=0.65, right=0.98,
                        wspace=0.5)

    ax_dict = dict(xlabel="time (ms)", ylabel="mV",
                   xticks=[0, 250, 500], xlim=[-10, 510])

    ax_spiketimes = fig.add_axes([0.07, 0.80, 0.17, 0.1], ylabel="#",
                                 xticks=[0, 250, 500], xticklabels=[],
                                 title="spike-time histogram",
                                 xlim=[-10, 510])

    ax_raw = fig.add_subplot(141, title="raw signal", **ax_dict)
    ax_hp = fig.add_subplot(142, title="HP filtered", **ax_dict)
    ax_hp_rect = fig.add_subplot(143, title="HP filtered\n+rectified",
                                 **ax_dict)
    ax_hp_rect_lp = fig.add_subplot(144,
                                 title="HP filtered\n+rectified\n+LP filtered",
                                 **ax_dict)

    ax_spiketimes.hist(spike_times, facecolor='k')

    ax_raw.plot(tvec, sig_raw, c='k')
    ax_hp.plot(tvec, sig_hp, c='k')
    ax_hp_rect.plot(tvec, sig_hp_rect, c='k')
    ax_hp_rect_lp.plot(tvec[100:], sig_hp_rect_lp[100:], c='k')

    for ax in [ax_raw, ax_hp, ax_hp_rect, ax_hp_rect_lp]:
        ax.axvspan(spike_burst_center - burst_width/2,
                 spike_burst_center + burst_width/2, facecolor='0.8',)

    mark_subplots(fig.axes[0], 'A', ypos=1.5, xpos=-0.2)
    mark_subplots(fig.axes[1:], 'BCDE', xpos=-0.2)
    simplify_axes(fig.axes)

    fig.savefig("fig_hay_MUA.png", dpi=300)
    fig.savefig("fig_hay_MUA.pdf", dpi=300)


if __name__ == '__main__':
    make_figure()

