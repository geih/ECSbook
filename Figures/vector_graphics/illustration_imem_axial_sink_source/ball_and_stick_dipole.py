
import os
from os.path import join
import sys
import numpy as np
import matplotlib.pyplot as plt

import neuron
from neuron import h
import LFPy

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
  pt3dadd(0, 0, 300, 1)}
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
forall {nseg = 3}
}
proc biophys() {
}
celldef()

Ra = 100.
cm = 1.
Rm = 30000

forall {
    insert pas // 'pas' for passive, 'hh' for Hodgkin-Huxley
    }
""")

dt = 2**-3
print("dt: ", dt)
cell_params = {          # various cell parameters,
            'morphology': h.all,
            'delete_sections': False,
            'v_init' : -70.,    # initial crossmembrane potential
            'passive' : False,   # switch on passive mechs
            'nsegs_method' : None,
            'dt' : dt,   # [ms] dt's should be in powers of 2 for both,
            'tstart' : 0.,    # start time of simulation, recorders start at t=0
            'tstop' : 5.,   # stop simulation at 200 ms. These can be overridden
        }


cell = LFPy.Cell(**cell_params)
cell.set_pos(x=-cell.xstart[0])

stim_idx = 0
stim_params = {
             'idx' : stim_idx,
             'record_current' : True,
             'syntype' : 'Exp2Syn',
             'tau1': 0.1,
             'tau2': 0.2,
             'weight' : 0.01,
            }

syn = LFPy.Synapse(cell, **stim_params)
syn.set_spike_times(np.array([1]))
cell.simulate(rec_vmem=True, rec_imem=True)

cell.imem = np.array([[-1], [0.75], [0.25]])

# Create a grid of measurement locations, in (um)
grid_x, grid_z = np.mgrid[-250:251:3, -250:450:2]
grid_y = np.zeros(grid_x.shape)

sigma = 0.3

# Define electrode parameters
grid_electrode_parameters = {
    'sigma' : sigma,      # extracellular conductivity
    'x' : grid_x.flatten(),  # electrode requires 1d vector of positions
    'y' : grid_y.flatten(),
    'z' : grid_z.flatten(),
    'method': 'pointsource'
}


grid_electrode = LFPy.RecExtElectrode(cell, **grid_electrode_parameters)
grid_electrode.calc_lfp()

fig = plt.figure(figsize=[12, 5])
fig.subplots_adjust(bottom=0.3, right=0.98, left=0.05)
ax1 = fig.add_subplot(151, aspect=1, frameon=False, xlabel=r"x ($\mu$m)",
                      ylabel=r"z ($\mu$m)", title="Morphology",
                      xlim=[-200, 200], ylim=[-250, 450])

ax_comp_based = fig.add_subplot(152, aspect=1, xlabel='x ($\mu$m)', ylabel='z ($\mu$m)', title="Sigma: %s S/m" % str(sigma),
                                frameon=False, xticks=[], yticks=[],
                                ylim=[np.min(grid_electrode.z), np.max(grid_electrode.z)],
                                xlim=[np.min(grid_electrode.x), np.max(grid_electrode.x)])

ax_dip_multi = fig.add_subplot(153, aspect=1, xlabel='x ($\mu$m)', ylabel='z ($\mu$m)', title="Sigma: %s S/m" % str(sigma),
                               frameon=False, xticks=[], yticks=[],
                               ylim=[np.min(grid_electrode.z), np.max(grid_electrode.z)],
                               xlim=[np.min(grid_electrode.x), np.max(grid_electrode.x)])

ax_dip_single = fig.add_subplot(154, aspect=1, xlabel='x ($\mu$m)', ylabel='z ($\mu$m)', title="Sigma: %s S/m" % str(sigma),
                                frameon=False, xticks=[], yticks=[],
                                ylim=[np.min(grid_electrode.z), np.max(grid_electrode.z)],
                                xlim=[np.min(grid_electrode.x), np.max(grid_electrode.x)])

cax = fig.add_axes([0.3, 0.11, 0.3, 0.01], frameon=False)


ax2_ = fig.add_subplot(255, ylabel="Injected current")
ax2 = fig.add_subplot(2,5,10, ylabel="Membrane potential (mV)")

[ax1.plot([cell.xstart[idx], cell.xend[idx]],
          [cell.zstart[idx], cell.zend[idx]], lw=4, c='k') for idx in range(cell.totnsegs)]

# ls, = ax1.plot(cell.xmid[stim_idx] - 10, cell.zmid[stim_idx] , '>', ms=10, c='g')
ax1.plot([cell.xmid[stim_idx] - 40, cell.xmid[stim_idx]],
         [cell.zmid[stim_idx], cell.zmid[stim_idx]], c='g')

max_idx = np.argmax(np.abs(cell.imem[0, :]))

# print(iaxial)
# print(d_axial)
# print(pos)
# p_axial = iaxial * d_axial

# inf_dip = LFPy.InfiniteVolumeConductor(sigma=sigma)

print(cell.xmid, cell.ymid, cell.zmid)

imem = cell.imem[:, max_idx]

LFP = 1000 * grid_electrode.LFP[:, max_idx].reshape(grid_x.shape)


i_a = np.array([-imem[0], imem[2]])
d_a = np.diff(np.array([cell.xmid, cell.ymid, cell.zmid]).T, axis=0)
pos_a = np.array([[0, 0],
                 [0, 0],
                 [50, 150]]).T
print(d_a)
print(imem)
p = (d_a.T * i_a).T
print(p)
electrode_locs = np.array([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()]).T

LFP_dp_multi = np.zeros(LFP.shape)
for idx in range(2):
    r = electrode_locs - pos_a[idx]
    LFP_dp_multi += 1000 * 1. / (4 * np.pi * sigma) * (np.dot(r, p[idx].T) / np.linalg.norm(r, axis=1) ** 3).reshape(grid_x.shape)

LFP_dp_single = np.zeros(LFP.shape)
p_sum = np.sum(p, axis=0)
r_mean = electrode_locs - np.mean(pos_a, axis=0)
LFP_dp_single = 1000 * 1. / (4 * np.pi * sigma) * (np.dot(r_mean, p_sum.T) / np.linalg.norm(r_mean, axis=1) ** 3).reshape(grid_x.shape)

num = 7
levels = np.logspace(-2., 0, num=num)
scale_max = 10**np.ceil(np.log10(np.max(np.abs(LFP)))) / 20

levels_norm = scale_max * np.concatenate((-levels[::-1], levels))
rainbow_cmap = plt.cm.get_cmap('PRGn') # rainbow, spectral, RdYlBu

colors_from_map = [rainbow_cmap(i*np.int(255/(len(levels_norm) - 2))) for i in range(len(levels_norm) -1)]
colors_from_map[num - 1] = (1.0, 1.0, 1.0, 1.0)

ep_intervals = ax_comp_based.contourf(grid_x, grid_z, LFP,
                                      zorder=-2, colors=colors_from_map,
                                      levels=levels_norm, extend='both')

ax_comp_based.contour(grid_x, grid_z, LFP, colors='k', linewidths=(1), zorder=-2,
                      levels=levels_norm)

ep_intervals_dp = ax_dip_multi.contourf(grid_x, grid_z, LFP_dp_multi,
                                        zorder=-2, colors=colors_from_map,
                                        levels=levels_norm, extend='both')

ax_dip_multi.contour(grid_x, grid_z, LFP_dp_multi, colors='k', linewidths=(1), zorder=-2,
                     levels=levels_norm)

ep_intervals_single_do = ax_dip_single.contourf(grid_x, grid_z, LFP_dp_single,
                                              zorder=-2, colors=colors_from_map,
                                              levels=levels_norm, extend='both')

ax_dip_single.contour(grid_x, grid_z, LFP_dp_single, colors='k', linewidths=(1), zorder=-2,
                     levels=levels_norm)

cbar = fig.colorbar(ep_intervals, cax=cax, orientation='horizontal',
                    )#format='%.0E')

cbar.set_label('$\phi$ (nV)', labelpad=0)

c = lambda idx: plt.cm.viridis(idx / (cell.totnsegs - 1))

ax2_.plot(cell.tvec, syn.i, c='g')
ax2.axhline(-70, c='gray', ls=':')
ax2.plot(cell.tvec, cell.vmem[0,:])

plt.savefig("comp_vs_multi_single_dp_based.png", dpi=300)
plt.savefig("comp_vs_multi_single_dp_based.pdf", dpi=300)
