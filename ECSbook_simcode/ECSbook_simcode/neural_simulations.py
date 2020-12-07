import os
import sys
from os.path import join
import numpy as np
import neuron
import LFPy
from ECSbook_simcode import cell_models

h = neuron.h
cell_models_folder = os.path.abspath(cell_models.__path__[0])
hay_folder = join(cell_models_folder, "L5bPCmodelsEH")

def download_hay_model():

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
    localFile = open(join(cell_models_folder, 'L5bPCmodelsEH.zip'), 'wb')
    localFile.write(u.read())
    localFile.close()
    #unzip:
    myzip = zipfile.ZipFile(join(cell_models_folder, 'L5bPCmodelsEH.zip'), 'r')
    myzip.extractall(cell_models_folder)
    myzip.close()

    #compile mod files every time, because of incompatibility with Mainen96 files:
    mod_pth = join(hay_folder, "mod/")

    if "win32" in sys.platform:
        warn("no autompile of NMODL (.mod) files on Windows.\n"
             + "Run mknrndll from NEURON bash in the folder "
               "L5bPCmodelsEH/mod and rerun example script")
        if not mod_pth in neuron.nrn_dll_loaded:
            neuron.h.nrn_load_dll(join(mod_pth, "nrnmech.dll"))
        neuron.nrn_dll_loaded.append(mod_pth)
    else:
        os.system('''
                  cd {}
                  nrnivmodl
                  '''.format(mod_pth))
        neuron.load_mechanisms(mod_pth)


def return_hay_cell(tstop, dt, make_passive=False):
    if not os.path.isfile(join(hay_folder, 'morphologies', 'cell1.asc')):
        download_hay_model()

    if make_passive:
        cell_params = {
            'morphology': join(hay_folder, 'morphologies', 'cell1.asc'),
            'passive': True,
            'nsegs_method': "lambda_f",
            "lambda_f": 100,
            'dt': dt,
            'tstart': -1,
            'tstop': tstop,
            'v_init': -70,
            'pt3d': True,
        }

        cell = LFPy.Cell(**cell_params)
        cell.set_rotation(x=4.729, y=-3.166)

        return cell
    else:
        neuron.load_mechanisms(join(hay_folder, 'mod'))
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
        }

        #Initialize cell instance, using the LFPy.Cell class
        cell = LFPy.TemplateCell(**cell_params)

        cell.set_rotation(x=4.729, y=-3.166)
        return cell

def return_stick_cell(tstop, dt):

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
                'tstart': -10.,
                'tstop': tstop,
                'pt3d': True,
            }
    cell = LFPy.Cell(**cell_params)
    # cell.set_pos(x=-cell.xstart[0])
    return cell

def return_two_comp_cell(tstop, dt):

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
      pt3dadd(0, 0, -10.0, 20)
      pt3dadd(0, 0, 10., 20)}
      dend[0] {pt3dclear()
      pt3dadd(0, 0, 10.0, 20)
      pt3dadd(0, 0, 30.0, 20)}
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
    dend[0] {nseg = 1}
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
                'tstart': -10.,
                'tstop': tstop,
                'pt3d': True,
            }
    cell = LFPy.Cell(**cell_params)
    # cell.set_pos(x=-cell.xstart[0])
    return cell

def remove_active_mechanisms(remove_list, cell):
    # remove_list = ["Nap_Et2", "NaTa_t", "NaTs2_t", "SKv3_1",
    # "SK_E2", "K_Tst", "K_Pst",
    # "Im", "Ih", "CaDynamics_E2", "Ca_LVAst", "Ca", "Ca_HVA"]
    mt = h.MechanismType(0)
    for sec in h.allsec():
        for seg in sec:
            for mech in remove_list:
                mt.select(mech)
                mt.remove(sec=sec)
    return cell

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

    noise_vec = (neuron.h.Vector(input_array) if weight is None
                 else neuron.h.Vector(input_array * weight))

    i = 0
    syn = None
    for sec in cell.allseclist:
        for seg in sec:
            if i == input_idx:
                print("Input inserted in ", sec.name())
                syn = neuron.h.ISyn(seg.x, sec=sec)
            i += 1
    if syn is None:
        raise RuntimeError("Wrong stimuli index")
    syn.dur = 1E9
    syn.delay = 0
    noise_vec.play(syn._ref_amp, cell.dt)
    return cell, syn, noise_vec
