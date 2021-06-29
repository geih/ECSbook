import os
from os.path import join
import numpy as np
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import LFPy
import neuron
from neuron import h
import ECSbook_simcode.neural_simulations as ns
from ECSbook_simcode.plotting_convention import mark_subplots, simplify_axes
from ECSbook_simcode import cell_models
neuron.load_mechanisms(ns.cell_models_folder)

np.random.seed(12345)

tstop = 30
dt = 2**-6
sigma = 0.3

cell_models_folder = os.path.abspath(cell_models.__path__[0])
hay_folder = join(cell_models_folder, "L5bPCmodelsEH")

neuron.load_mechanisms(join(hay_folder, 'mod'))

def return_hay_cell(tstop, dt):

    # cell_params = {
    #     'morphology': join(hay_folder, 'morphologies', 'cell1.asc'),
    #     'passive': True,
    #     'nsegs_method': "lambda_f",
    #     "lambda_f": 100,
    #     'dt': dt,
    #     'tstart': -1,
    #     'tstop': tstop,
    #     'v_init': -70,
    #     'pt3d': True,
    #     'extracellular': True,
    # }
    #
    # cell = LFPy.Cell(**cell_params)
    # cell.set_rotation(x=4.729, y=-3.166)
    # h = neuron.h
    #
    cell_params = {
        'morphology': join(hay_folder, "morphologies", "cell1.asc"),
        'templatefile': [join(hay_folder, 'models', 'L5PCbiophys3.hoc'),
                         join(hay_folder, 'models', 'L5PCtemplate.hoc')],
        'templatename': 'L5PCtemplate',
        'templateargs': join(hay_folder, 'morphologies', 'cell1.asc'),
        'passive': False,
        'nsegs_method': None,
        'dt': dt,
        'tstart': -200,
        'tstop': tstop,
        'v_init': -75,
        'celsius': 34,
        'pt3d': True,
        'extracellular': True,
    }

    cell = LFPy.TemplateCell(**cell_params)
    cell.set_rotation(x=4.729, y=-3.166)
    return cell

def return_ball_and_stick_cell(tstop, dt):

    h("forall delete_section()")
    h("""
    proc celldef() {
      topol()
      subsets()
      geom()
      biophys()
      geom_nseg()
    }

    create soma[1], dend[1]

    proc topol() { local i
      basic_shape()
      connect dend(0), soma(1)
    }
    proc basic_shape() {
      soma[0] {pt3dclear()
      pt3dadd(0, 0, -10., 20.)
      pt3dadd(0, 0, 10., 20.)}
      dend[0] {pt3dclear()
      pt3dadd(0, 0, 10., 5)
      pt3dadd(0, 0, 1000, 5)}
    }

    objref all
    proc subsets() { local i
      objref all
      all = new SectionList()
        soma[0] all.append()
        dend[0] all.append()

    }
    proc geom() {
    }
    proc geom_nseg() {
    soma[0] {nseg = 1}
    dend[0] {nseg = 200}
    }
    proc biophys() {
    }
    celldef()

    Ra = 150.
    cm = 1.
    Rm = 30000.

    soma[0] {
        insert hh
        }
    dend[0] {
        insert pas // 'pas' for passive, 'hh' for Hodgkin-Huxley
        g_pas = 1 / Rm
        }
    """)
    cell_params = {
                'morphology': h.all,
                'delete_sections': False,
                'v_init': -65.,
                'passive': False,
                'nsegs_method': None,
                'dt': dt,
                'tstart': -150.,
                'tstop': tstop,
                'pt3d': True,
                'extracellular': True,
            }
    cell = LFPy.Cell(**cell_params)
    # cell.set_pos(x=-cell.xstart[0])
    return cell


def insert_synaptic_input(cell, synidx):

    synapse_parameters = dict(
                          idx = synidx,
                          e = 0., # reversal potential
                          weight = 0.15, # synapse weight
                          record_current = True, # record synapse current
                          syntype = 'Exp2Syn',
                          tau1 = 0.1, #Time constant, rise
                          tau2 = 1.0, #Time constant, decay
                          )
    synapse = LFPy.Synapse(cell, **synapse_parameters)
    synapse.set_spike_times(np.array([3.]))
    return synapse, cell


def test_autoephaptic():

    # return_cell_method = return_ball_and_stick_cell
    return_cell_method = return_hay_cell

    cell = return_cell_method(tstop, dt)
    plot_idxs = [cell.get_closest_idx(0, 0, np.max(cell.z)),
                 cell.get_closest_idx(0, 0, np.max(cell.z) / 2),
                 cell.get_closest_idx(0, 0, 0),
                 ]
    plot_idx_clrs = ['b', 'gray', 'r']

    cell_x = cell.x.copy()
    cell_z = cell.z.copy()
    # h.dt = dt
    elec_params = dict(
                sigma = sigma,      # extracellular conductivity
                x = cell.x.mean(axis=1),
                y = cell.y.mean(axis=1),
                z = cell.z.mean(axis=1),
                method = 'pointsource',
            )
    elec = LFPy.RecExtElectrode(cell, **elec_params)
    M_elec = elec.get_transformation_matrix().copy()

    synidx = cell.get_closest_idx(x=0, y=0, z=0)

    # Original un-ephaptic simulation:
    syn, cell = insert_synaptic_input(cell, synidx)
    cell.simulate(rec_imem=True, rec_vmem=True)
    t = cell.tvec.copy()
    v_orig = cell.vmem.copy()
    self_caused_pot = M_elec @ cell.imem
    self_caused_pot_orig = self_caused_pot.copy()
    cell.__del__()

    for iteration in range(5):
        cell = return_cell_method(tstop, dt)
        syn, cell = insert_synaptic_input(cell, synidx)
        cell.insert_v_ext(self_caused_pot, t)

        cell.simulate(rec_imem=True, rec_vmem=True)

        self_caused_pot_ = M_elec @ cell.imem
        vmem = cell.vmem.copy()
        max_error = np.max(np.abs((self_caused_pot_ - self_caused_pot)) /
                     np.max(np.abs(self_caused_pot)))
        print("Max relative error: {:1.5f}".format(max_error))
        self_caused_pot = self_caused_pot_
        cell.__del__()
        cell = None
        syn = None


    fig = plt.figure(figsize=[10, 6])
    fig.subplots_adjust(left=0.07, wspace=0.5, hspace=0.5, right=0.98,
                        bottom=0.15)
    ax1 = fig.add_subplot(131, aspect=1, xlim=[-300, 300],
                          title="morphology", xlabel="x (µm)",
                          ylabel="z (µm)")

    ax1.plot(cell_x.T, cell_z.T, 'k')
    ax1.plot(cell_x[0].mean(), cell_z[0].mean(), 'ko', ms=13)

    for i, comp in enumerate(plot_idxs):
        c = plot_idx_clrs[i]
        ax1.plot(cell_x[comp].mean(), cell_z[comp].mean(), 'o', c=c)
        ax_v = fig.add_subplot(3, len(plot_idxs), i * 3 + 2,
                               ylabel="mV",
                               xlabel="time (ms)")
        ax_ecp = fig.add_subplot(3, len(plot_idxs), i * 3 + 3,
                               ylabel="mV",
                                 xlabel="time (ms)")

        if i == 0:
            ax_v.set_title("membrane potential")
            ax_ecp.set_title("extracellular potential ")
        ax_v.plot(t, v_orig[comp], c=c)
        ax_v.plot(t, vmem[comp], c='k', ls='--')


        l_orig, = ax_ecp.plot(t, self_caused_pot_orig[comp], c=c)
        l_efap, = ax_ecp.plot(t, self_caused_pot[comp], c='k', ls='--')

    mark_subplots(fig.axes, ypos=1.1)
    simplify_axes(fig.axes)
    fig.legend([l_orig, l_efap], ["control", "auto-ephaptic"], ncol=2,
               loc="lower right", frameon=False)
    # plt.plot(cell.tvec, cell.somav, 'r')
    # plt.plot(cell.tvec, cell.somav - v_orig, 'blue')
    plt.savefig("fig_ephaptic_hay_active.png")
    # plt.show()


if __name__ == '__main__':
    test_autoephaptic()