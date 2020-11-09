import os
import sys
from os.path import join
import numpy as np
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import neuron
import LFPy
from ECSbook_simcode.plotting_convention import mark_subplots, simplify_axes
from ECSbook_simcode.neural_simulations import return_hay_cell

np.random.seed(12345)

def insert_current_stimuli(amp, cell, delay):

    stim_params = {'amp': amp,
                   'idx': 0,
                   'pptype': "IClamp",
                   'dur': 1e9,
                   'delay': delay}
    synapse = LFPy.StimIntElectrode(cell, **stim_params)
    return synapse, cell


def make_figure():
    current_amps = np.linspace(-0.45, 0.45, 15)  # 118 221,

    plt.close("all")
    fig = plt.figure(figsize=[4, 5])
    fig.subplots_adjust(bottom=0.13, top=0.98, right=0.99,
                        left=0.0, wspace=-0.0, hspace=0.2)
    tstop = 50
    dt = 2**-5
    delay = 5
    dVs = np.zeros(len(current_amps))
    somavs = []
    tvec = None

    for i, amp in enumerate(current_amps):
        cell = return_hay_cell(tstop=tstop, dt=dt, make_passive=False)
        syn, cell = insert_current_stimuli(amp, cell, delay=delay)
        cell.simulate(rec_imem=False, rec_vmem=True)
        v = cell.vmem[0, :]
        t0_idx = np.argmin(np.abs(cell.tvec - delay)) - 1
        dV_max_idx = np.argmax(np.abs(v - v[t0_idx]))
        dVs[i] = v[dV_max_idx] - v[t0_idx]
        somavs.append(v.copy())
        tvec = cell.tvec.copy()
        cell.__del__()

    fig = plt.figure(figsize=[7, 3])
    fig.subplots_adjust(left=0.14, bottom=0.18, top=0.85, right=0.98)

    ax1 = fig.add_subplot(121, xlabel="time (ms)",
                          ylabel="membrane potential (mV)")
    ax2 = fig.add_subplot(122, xlabel="I (nA)", ylabel="$\Delta$V$_m$")
    for i, amp in enumerate(current_amps):
        ax1.plot(tvec, somavs[i], 'k')

    ax2.plot(current_amps, dVs, 'k-x')
    ax2.plot([current_amps[0], current_amps[-2]], [dVs[0], dVs[-2]], 'r--')

    mark_subplots(fig.axes)
    simplify_axes(fig.axes)
    fig.savefig("fig_hay_subthresh_linearity_test.png", dpi=300)
    fig.savefig("fig_hay_subthresh_linearity_test.pdf", dpi=300)


if __name__ == '__main__':
    # make_all_basal_dipoles()
    make_figure()

