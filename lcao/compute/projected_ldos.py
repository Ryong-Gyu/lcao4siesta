import math
import time

import numpy as np

from lcao.compute.kernels import accumulate_overlap_weight, compute_projection_factor
from lcao.core.model import phi_tolerance
from lcao.selection.orbital_selector import mask_to_pointer, orbital_mask


def orbital_projected_local_density_of_state(projector, select, energys, cell, mesh):
    """Compute orbital-projected local density of states on a real-space grid.

    Contract: for identical inputs, the output `pldos` keeps the same shape and
    iteration order as before (`(ntarget, nenergy, na, nb, nc)`).
    """
    projector.load_context(need_wfsx_hsx=True, need_struct_supercell=True)
    projector._target = []

    energys = np.array(energys)
    nenergy = len(energys)

    if isinstance(select, str):
        select = [select]
    orbital_mask(projector, select)
    mask_to_pointer(projector)

    xgrid, ygrid, zgrid = projector.unit_cell_grid(cell, mesh)

    gamma = projector.gamma
    wf = projector.wavefunction
    wk = projector.kweight
    kpt = projector.kpoints
    eig = projector.eigenvalue

    index = projector.atom_index
    symbol = projector.atom_species

    target = projector._target
    list_io = projector._projection_io
    list_ptr = projector._projection_ptr

    Sover = projector.Sover
    xij = projector.xij

    cell = projector.cell
    atoms = projector.atoms
    supercell_vectors = projector._supercell_vector_list
    nvectors = len(supercell_vectors)

    na = int(mesh[0])
    nb = int(mesh[1])
    nc = int(mesh[2])

    nwavefunctions = eig.shape[0]
    nspin = eig.shape[1]
    nkpoints = eig.shape[2]
    ntarget = len(target)

    spatial = np.zeros((2, nvectors), dtype=float)
    pldos = np.zeros((ntarget, nenergy, na, nb, nc), dtype=float)

    for itar in range(ntarget):
        start = time.time()
        tar = target[itar]

        for ix in range(na):
            for iy in range(nb):
                for iz in range(nc):
                    _ = time.time() - start
                    position_vector = np.zeros((3), dtype=float)
                    position_vector[0] = xgrid[0][ix][iy][iz]
                    position_vector[1] = ygrid[0][ix][iy][iz]
                    position_vector[2] = zgrid[0][ix][iy][iz]
                    list_target_io = tar['orbital_index']
                    number_of_target = len(list_target_io)
                    buff0 = np.zeros((nenergy, number_of_target), dtype=float)
                    iio1 = 0
                    for io1 in list_target_io:
                        atom_symbol = symbol[io1]
                        atom_index = index[io1] - 1
                        target_position = atoms[atom_index]
                        target_vector = target_position - position_vector
                        target_n = tar['principle_quantum_number'][iio1]
                        target_l = tar['angular_quantum_number'][iio1]
                        target_m = tar['magnetic_quantum_nuber'][iio1]
                        target_z = tar['zeta'][iio1]

                        buff1 = np.zeros((nenergy, 2, nkpoints, nspin, nwavefunctions), dtype=float)
                        buff = np.zeros((nenergy, nkpoints), dtype=float)
                        for ik in range(nkpoints):
                            for isp in range(nspin):
                                for iw in range(nwavefunctions):
                                    eigenvalue = eig[iw][isp][ik]
                                    nprojection = len(list_io[itar])
                                    buff2 = np.zeros((2, nprojection), dtype=float)
                                    iio2 = 0
                                    for io2 in list_io[itar]:
                                        ind = list_ptr[itar][iio2]
                                        qcos, qsin = compute_projection_factor(gamma, wf, io1, io2, iw, isp, ik)
                                        alfa = np.inner(kpt[ik], xij[ind])
                                        real_weight, imag_weight = accumulate_overlap_weight(Sover, xij, kpt[ik], ind, qcos, qsin)

                                        if Sover[ind] != 0:
                                            buff2[0][iio2] = real_weight / Sover[ind]
                                            buff2[1][iio2] = imag_weight / Sover[ind]
                                        else:
                                            buff2[0][iio2] = 0
                                            buff2[1][iio2] = 0
                                        buff2 *= Sover[ind]
                                        iio2 += 1

                                    rval = (buff2[0]).sum()
                                    ival = (buff2[1]).sum()
                                    for ie in range(nenergy):
                                        factor = projector.delta(energys[ie] - eigenvalue)
                                        buff1[ie][0][ik][isp][iw] = rval * factor
                                        buff1[ie][1][ik][isp][iw] = ival * factor

                            buff1 = wk[ik] * buff1

                            for iv in range(nvectors):
                                xji = -(target_vector + supercell_vectors[iv])
                                phase = np.inner(kpt[ik], xji)
                                r = np.sqrt(xji.dot(xji))
                                phir = projector.Rnl(
                                    atom_symbol,
                                    target_n,
                                    target_l,
                                    target_z,
                                    r,
                                    io=io1 + 1,
                                    ia=index[io1],
                                )

                                if phir < phi_tolerance:
                                    factor = 0
                                else:
                                    spherical = projector.Yml(xji, target_m, target_l)
                                    factor = phir * spherical

                                spatial[0][iv] = factor.real * math.cos(phase) - factor.imag * math.sin(phase)
                                spatial[1][iv] = factor.real * math.sin(alfa) + factor.imag * math.cos(phase)
                            rspatial = spatial[0].sum()
                            ispatial = spatial[1].sum()

                            for ie in range(nenergy):
                                rphi_orbital = buff1[ie][0][ik].sum()
                                iphi_orbital = buff1[ie][1][ik].sum()
                                buff[ie][ik] = rphi_orbital * rspatial - iphi_orbital * ispatial

                        for ie in range(nenergy):
                            buff0[ie][iio1] = buff[ie].sum()
                        iio1 += 1

                    for ie in range(nenergy):
                        pldos[itar][ie][ix][iy][iz] = buff0[ie].sum()

    projector.pldos = pldos
    return pldos
