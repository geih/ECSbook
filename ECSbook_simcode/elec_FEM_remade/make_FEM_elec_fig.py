import numpy as np
import os
from os.path import join
import sys
import dolfin as df
import matplotlib.pyplot as plt
import LFPykit
from plotting_convention import *
from matplotlib.colors import LogNorm
import ConductivityClass
from params import *


plt.seed(1234)

def electrode_fem_recip(solver_params, charges_pos, elec_pos,
                                magnitudes, mesh_dict, elec_name):
    """ The potential from charges in the electrode is found by FEM"""

    print("Loading meshes ...")
    mesh = df.Mesh(join(mesh_dict['folder'], "mesh_%s.xml" % elec_name))
    subdomains = df.MeshFunction("size_t", mesh,
                                 join(mesh_dict['folder'], "mesh_%s_physical_region.xml" % elec_name))
    boundaries = df.MeshFunction("size_t", mesh,
                                 join(mesh_dict['folder'], "mesh_%s_facet_region.xml" % elec_name))
    V = df.FunctionSpace(mesh, "CG", solver_params["LagrangeP"])
    Vs = df.FunctionSpace(mesh, "DG", 0)
    values = np.zeros(len(charges_pos))
    readout_pos = elec_pos[0], elec_pos[1], elec_pos[2] - mesh_dict['elec_depth']/2.
    print("Readout pos ", readout_pos)
    # sigma_class = ConductivityClass.ElectrodeConductivity(**set_electrode)
    sigma_class = ConductivityClass.SigmaClass()

    sigma = sigma_class.return_conductivity_tensor(Vs)

    ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)
    dx = df.Measure("dx", domain=mesh, subdomain_data=subdomains)

    v = df.TestFunction(V)
    u = df.TrialFunction(V)
    a = df.inner(sigma * df.grad(u), df.grad(v)) * dx(1)
    L = df.Constant(0) * v * df.dx
    A = df.assemble(a)
    b = df.assemble(L)

    bc = df.DirichletBC(V, df.Constant(0), boundaries, 9)
    bc.apply(A, b)
    phi = df.Function(V)
    point = df.Point(readout_pos[0], readout_pos[1], readout_pos[2])
    delta = df.PointSource(V, point, 1.0)
    delta.apply(b)

    df.solve(A, phi.vector(), b, solver_params["linear_solver"], solver_params["preconditioner"])

    for idx, pos in enumerate(charges_pos):
        values[idx] = phi(pos)
    return values


def moi_value(charges_pos, elec_r, elec_pos, npts=100):
    """ Find the value at the electrode using the Method of Images (MoI) with electrode
    """

    xmid = charges_pos[:, 0]
    ymid = charges_pos[:, 1]
    zmid = charges_pos[:, 2]
    include_elec = elec_r != None
    ext_sim_dict = {'include_elec': include_elec,
                    'elec_x': np.array([0.]),
                    'elec_y': np.array([0.]),
                    'elec_z': elec_pos[2] + 150,
                    'n_avrg_points': npts,
                    'elec_radius': elec_r,
                    'use_line_source': False,
                    }

    from lfpykit import CellGeometry, PointSourcePotential, RecExtElectrode
    mappings = []
    for idx in range(len(xmid)):
        x = np.array([[xmid[idx], xmid[idx]]])
        y = np.array([[ymid[idx], ymid[idx]]])
        z = np.array([[zmid[idx], zmid[idx]]])

        cell = CellGeometry(x=x,
                            y=y,
                            z=z,
                            d=np.array([1]))

        psp = RecExtElectrode(cell,
                               x=np.array([0.]),
                               y=np.array([0.]),
                               z=np.array([elec_pos[2]]),
                               sigma=0.3,
                               method="pointsource",
                              r=elec_r,
                              n=npts,
                              N=[0, 0, 1])

        M = psp.get_transformation_matrix()
        print(M)
        mappings.append([M[0, 0]])
        # imem = np.array([[1.]])
        # V_ex = M @ imem
    # print(mappings)

    # mapping = moi.make_mapping_cython(ext_sim_dict, xmid=xmid, ymid=ymid, zmid=zmid + moi.h / 2)
    # ground = 0
    # moi_value_cython = M - ground
    return np.array(mappings)


def moi_value_recip(charges_pos, elec_r, elec_pos, npts=100):
    """ Find the value at the electrode using the Method of Images (MoI) with electrode
    """

    xmid = charges_pos[:, 0]
    ymid = charges_pos[:, 1]
    zmid = charges_pos[:, 2]

    from lfpykit import CellGeometry, PointSourcePotential, RecExtElectrode

    if elec_r is None:
        x = np.array([[elec_pos[0], elec_pos[0]]])
        y = np.array([[elec_pos[1], elec_pos[1]]])
        z = np.array([[elec_pos[2], elec_pos[2]]])
        d = np.array([1])
    else:
        x = np.zeros((npts, 2))
        y = np.zeros((npts, 2))
        z = np.ones((npts, 2)) * elec_pos[2]
        d = np.ones(npts)
        for idx in range(npts):
            x_ = np.random.uniform(-elec_r, elec_r)
            y_ = np.random.uniform(-elec_r, elec_r)
            while np.sqrt(x_**2 + y_**2) > elec_r:
                x_ = np.random.uniform(-elec_r, elec_r)
                y_ = np.random.uniform(-elec_r, elec_r)
            x[idx, :] = [x_, x_]
            y[idx, :] = [y_, y_]

    cell = CellGeometry(x=x,
                        y=y,
                        z=z,
                        d=d)

    psp = PointSourcePotential(cell,
                           x=xmid,
                           y=ymid,
                           z=zmid,
                           sigma=0.3,)


    M = psp.get_transformation_matrix()
    ecp = M @ np.ones((npts, 1)) / npts
    ecp -= 1 / (4 * np.pi * 0.3 * 10000)
    return ecp


def elec_impact_simulations():
    
    solver_params = {'linear_solver': 'cg',
                     'preconditioner': 'ilu',
                     'LagrangeP': 2,
                     }

    df.parameters["krylov_solver"]["relative_tolerance"] = 1e-10

    out_folder = 'electrode_results'
    if not os.path.isdir(out_folder): os.mkdir(out_folder)
    elec_pos = np.array([0., 0., elec_z + epsilon], dtype='float')

    #moi = ViMEAPy.MoI(**set_control)
    electrodes = ['E1']#, 'E2', 'E3']

    for elec_name in electrodes:
        print(elec_name)
        elec_r = mesh_dict['elec_r%s' % elec_name[1]]
        charge_z_pos = np.linspace(0.5, 10, 10) * elec_r + elec_z
        charges_pos = np.zeros((len(charge_z_pos), 3), dtype='float')
        magnitudes = np.ones(len(charges_pos))
        charges_pos[:, 2] = charge_z_pos
        np.save(join(out_folder, 'positions_%s.npy' % elec_name), charges_pos)

        elec_pts_to_average = 500
        moi_values = 2 * moi_value_recip(charges_pos, elec_r, elec_pos, elec_pts_to_average)[:, 0]
        np.save(join(out_folder, 'moi_%s_%d.npy' % (elec_name, elec_pts_to_average)), moi_values)
        
        elec_pts_to_average = 1
        moi_values = 2 * moi_value_recip(charges_pos, None, elec_pos, elec_pts_to_average)[:, 0]
        np.save(join(out_folder, 'moi_%s_%d.npy' % (elec_name, elec_pts_to_average)), moi_values)

        fem_values = electrode_fem_recip(solver_params, charges_pos, elec_pos, magnitudes, mesh_dict, elec_name)
        np.save(join(out_folder, 'fem_recip_%s.npy' % elec_name), fem_values)


def make_FEM_elec_fig():

    folder = 'electrode_results'
    n_avrg_points = 500
    
    plt.close('all')
    fig = plt.figure(figsize=[10, 2.7])
    fig.subplots_adjust(wspace=0.33, hspace=0.5, bottom=0.25, top=0.84, left=0.05, right=0.97)
    elec_names = ['E1']
    ax0 = fig.add_subplot(144, yscale='log',
                       xlabel='Source height [e.r.]',
                      ylabel='Relative error')
    ax0.axis([0, 10, 1e-4, 1e0])

    # elec_name_pos = [5e-1, 1.5e-1, 5e-2]
    
    for elec_idx, elec_name in enumerate(elec_names):
        charges_pos = np.load(join(folder, 'positions_%s.npy' % elec_name))[:, 2]
        clr = 'kbg'[elec_idx]
        elec_r = float(mesh_dict['elec_r%s' % elec_name[1]])

        charge_heights = charges_pos - elec_z
        moi_avrg = np.load(join(folder, 'moi_%s_%d.npy' % (elec_name, n_avrg_points)))
        moi_1 = np.load(join(folder, 'moi_%s_%d.npy' % (elec_name, 1)))
        fem = np.load(join(folder, 'fem_recip_%s.npy' % elec_name))
        normalize_factor = np.max([fem])

        fem /= normalize_factor
        moi_avrg /= normalize_factor
        moi_1 /= normalize_factor

        rel_error_fem_elec_moi_no_elec = np.abs(fem - moi_1)/fem
        rel_error_elec = np.abs(fem - moi_avrg)/fem
        ax0.plot(charge_heights/elec_r, rel_error_elec, '-', lw=2, color=clr)
        ax0.plot(charge_heights/elec_r, rel_error_fem_elec_moi_no_elec, '--', lw=2, color=clr)
        # ax0.text(10.1, elec_name_pos[elec_idx], '%d $\mu m$' % elec_r, color=clr, size=10)

    ax0.plot(1000, 1000, '-', color='k', lw=2, label='disc electrode')
    ax0.plot(1000, 1000, '--', color='k', lw=2, label='point electrode')
    simplify_axes(ax0)
    ax0.grid(True)
    ax0.legend(bbox_to_anchor=(0.1, 1.3), handlelength=3, frameon=False)

    positions = np.load(join(folder, 'lateral_grid_positions.npy'))

    elec_dists = 5
    elec_r = 5

    x = positions[:, 0].reshape(30, 30).T / elec_r
    z = (positions[:, 2].reshape(30, 30).T + slice_thickness/2.) / elec_r
    moi_1 = np.load(join(folder, 'moi_lateral_grid_1.npy'))
    moi_avrg = np.load(join(folder, 'moi_lateral_grid_%s.npy' % n_avrg_points))
    fem = np.load(join(folder, 'fem_lateral_recip_grid.npy'))

    normalize_const = np.max(fem)
    moi_1 = moi_1.reshape(30, 30).T / normalize_const
    moi_avrg = moi_avrg.reshape(30, 30).T / normalize_const
    fem = fem.reshape(30, 30).T / normalize_const
    error_1 = np.abs(fem - moi_1)
    error_avrg = np.abs(fem - moi_avrg)

    ax_kwargs = {'frameon': False,
                 'xlim': [0, elec_dists],
                 'ylim': [0, elec_dists],
                 'xticks': np.arange(6),
                 'aspect': '1'
                 }

    ax1 = fig.add_subplot(141, title='FEM', xlabel='Lateral position [e.r.]',
                 ylabel='Source height [e.r.]', **ax_kwargs)
    ax2 = fig.add_subplot(142, title='point electrode\napproximation', **ax_kwargs)
    ax5 = fig.add_subplot(143, title='disc electrode\napproximation', **ax_kwargs)
    # ax3 = fig.add_subplot(324, title='Error', **ax_kwargs)
    # ax6 = fig.add_subplot(326, title='Error', **ax_kwargs)

    extent = [0, np.max(x)/elec_r, 0, np.max(z)/elec_r]

    contourf_kwargs = {'origin': 'lower',
                     'cmap': plt.cm.hot,
                     #'norm': LogNorm()
                     }

    contour_kwargs = {'origin': 'lower',
                     'extent': extent,
                     'colors': 'k',
                      "linewidths": 0.5,
                     }


    # clevels_fields = [0.01, 0.1, 0.25, 0.5, 0.75, 1., 3.]
    clevels_fields = np.linspace(0, 1, 10)#np.logspace(-1., 0, 15)
    print(clevels_fields)
    # clevels_errors = [1e-5, 1e-2, 1e-1, 1e0]
    clevels_errors =  [1e-5, 1e-2, 1e-1, 1e0]#np.logspace(-5, 0, 10)
    print(np.max(fem))

    im1 = ax1.contourf(x, z, fem, levels=clevels_fields,  **contourf_kwargs)
    im2 = ax2.contourf(x, z, moi_1, levels=clevels_fields, **contourf_kwargs)
    im5 = ax5.contourf(x, z, moi_avrg, levels=clevels_fields,  **contourf_kwargs)

    ax1.contour(x, z, fem, levels=clevels_fields, **contour_kwargs)
    ax2.contour(x, z, moi_1, levels=clevels_fields, **contour_kwargs)

    ax5.contour(x, z, moi_avrg, levels=clevels_fields, **contour_kwargs)

    ax_list = [ax1, ax2, ax5, ax0,]
    #
    mark_subplots(ax_list)
    [ax.plot([0, 1], [-.1, -.1], '-', lw=2, color='k', clip_on=False) for ax in ax_list]

    fields = [[ax1, im1], [ax2, im2], [ax5, im5]]
    # for ax, im in fields:
    cax = fig.add_axes([0.3, 0.12, 0.17, 0.02])
    cbar = plt.colorbar(im1, cax=cax, orientation="horizontal")
        # cbar.set_ticks(clevels_fields)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
        # cbar.set_ticklabels(['%1.2f' % tick for tick in clevels_fields])
        # cbar.set_ticklabels([])
    cbar.solids.set_rasterized(True)

    # for ax, im in [[ax3, im3], [ax6, im6]]:
    #     cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    #     cbar.solids.set_rasterized(True)

    plt.savefig('fig_FEM_elec.pdf', dpi=300)
    plt.savefig('fig_FEM_elec.png', dpi=300)

def simulate_lateral_grid():
    
    solver_params = {'linear_solver': 'cg',
                     'preconditioner': 'ilu',
                     'LagrangeP': 2,
                     }
    df.parameters["krylov_solver"]["relative_tolerance"] = 1e-10

    out_folder = join('electrode_results')
    os.makedirs(out_folder, exist_ok=True)
    epsilon = 1e-15
    elec_pos = [0., 0., elec_z + epsilon]

    elec_name = 'E1'
    elec_r = 5.
    
    charge_z_pos = np.linspace(1, 5 * elec_r, 30) + elec_z
    charge_x_pos = np.linspace(0, 5 * elec_r, 30)
    
    charges_pos = np.zeros((len(charge_x_pos)*len(charge_z_pos), 3))
    idx = 0
    idx_matrix = np.zeros((len(charge_z_pos), len(charge_x_pos)))
    for col, xpos in enumerate(charge_x_pos):
        for row, zpos in enumerate(charge_z_pos):
            charges_pos[idx, :] = [xpos, 0, zpos]
            idx_matrix[row, col] = idx
            idx += 1

    np.save(join(out_folder, 'lateral_grid_positions.npy'), charges_pos)
    magnitudes = np.ones(len(charges_pos))

    #moi = ViMEAPy.MoI(set_control)
    n_avrg_points = 1
    moi_values = 2 * moi_value_recip(charges_pos,  None, elec_pos, n_avrg_points)[:, 0]
    np.save(join(out_folder, 'moi_lateral_grid_%d.npy' % n_avrg_points), moi_values)

    n_avrg_points = 500
    moi_values = 2 * moi_value_recip(charges_pos, elec_r, elec_pos, n_avrg_points)[:, 0]

    np.save(join(out_folder, 'moi_lateral_grid_%d.npy' % n_avrg_points), moi_values)

    lateral_values_recip = electrode_fem_recip(solver_params, charges_pos, elec_pos, magnitudes,
                                               mesh_dict, elec_name)
    np.save(join(out_folder, 'fem_lateral_recip_grid.npy'), lateral_values_recip)

if __name__ == '__main__':

    # elec_impact_simulations()
    # simulate_lateral_grid()
    make_FEM_elec_fig()
