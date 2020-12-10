
import numpy as np
import matplotlib
# matplotlib.use("AGG")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Ellipse
import LFPy
import elephant
from ECSbook_simcode.plotting_convention import mark_subplots, simplify_axes
from ECSbook_simcode.neural_simulations import return_hay_cell
import ECSbook_simcode.neural_simulations as ns

np.random.seed(12345)

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
    synapse.set_spike_times(np.array([3.]))
    return synapse, cell



def make_figure():

    tstop = 50
    dt = 2**-5

    filt_dict_lf = {'highpass_freq': None,
                     'lowpass_freq': 300,
                     'order': 4,
                     'filter_function': 'filtfilt',
                     'fs': 1 / dt * 1000,
                     'axis': -1}

    filt_dict_hf = {'highpass_freq': 300,
                     'lowpass_freq': None,
                     'order': 4,
                     'filter_function': 'filtfilt',
                     'fs': 1 / dt * 1000,
                     'axis': -1}


    #

    cell = return_hay_cell(tstop=tstop, dt=dt, make_passive=False)
    syn, cell = insert_synaptic_input(cell)
    cell.simulate(rec_imem=True, rec_vmem=True)

    elec_params = dict(
                sigma = 0.3,      # extracellular conductivity
                x = np.array([30.]),
                y = np.array([0.0]),
                z = np.array([0.0]),
                method = 'root_as_point',
            )

    elec = LFPy.RecExtElectrode(cell, **elec_params)
    M_elec = elec.get_transformation_matrix()
    eap = M_elec @ cell.imem * 1000

    eap[0] -= np.linspace(eap[0, 0], eap[0, -1], len(eap[0]))
    print(eap[0])

    eap_lf = elephant.signal_processing.butter(eap, **filt_dict_lf)
    eap_hf = elephant.signal_processing.butter(eap, **filt_dict_hf)

    freqs_vm, vm_psd = ns.return_freq_and_amplitude(cell.tvec, cell.vmem[0, :])
    freqs_eap, eap_psd = ns.return_freq_and_amplitude(cell.tvec, eap)
    freqs_eap, eap_psd_lf = ns.return_freq_and_amplitude(cell.tvec, eap_lf)
    freqs_eap, eap_psd_hf = ns.return_freq_and_amplitude(cell.tvec, eap_hf)

    plt.close("all")
    fig = plt.figure(figsize=[10, 4])
    fig.subplots_adjust(left=0.1, bottom=0.25, top=0.87, right=0.98,
                        hspace=0.5)

    ax_vm = fig.add_subplot(131, title="membrane\npotential",
                            xlabel="time (ms)", ylabel="mV",
                            xlim=[0, 15])

    ax_eap = fig.add_subplot(132, title="extracellular spike",
                             xlabel="time (ms)", ylabel="µV", xlim=[0, 15])
    ax_eap_psd = fig.add_subplot(133, title="extracellular spike\nPSD",
                                 xlim=[0, 2000],
                                 xlabel="frequency (Hz)", ylabel="µV")#$^2$ / Hz")
    ax_eap_psd.grid(True)

    ax_vm.plot(cell.tvec, cell.vmem[0, :], c='gray', lw=2)
    l, = ax_eap.plot(cell.tvec, eap[0], c='k', lw=2)
    l_hf, = ax_eap.plot(cell.tvec, eap_hf[0], c='b', lw=2)
    l_lf, = ax_eap.plot(cell.tvec, eap_lf[0], c='r', lw=2)

    lines = [l, l_lf, l_hf]
    line_names = ["original", "low-pass filtered (< 300 Hz)",
                  "high-pass filtered  (> 300 Hz)"]

    ax_eap_psd.plot(freqs_eap, eap_psd[0], 'k')
    ax_eap_psd.plot(freqs_eap, eap_psd_hf[0], 'b')
    ax_eap_psd.plot(freqs_eap, eap_psd_lf[0], 'r')
    # l_vm, = ax_eap_psd.plot(freqs_eap, vm_psd[0], c='gray', lw=2)

    fig.legend(lines, line_names, loc="lower right", frameon=False, ncol=3)
    mark_subplots(fig.axes)
    simplify_axes(fig.axes)

    fig.savefig("fig_spike_freq_content_amp.png", dpi=300)


if __name__ == '__main__':
    make_figure()

