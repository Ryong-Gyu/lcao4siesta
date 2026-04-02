import numpy as np

from lcao.compute.kernels import accumulate_overlap_weight, compute_projection_factor
from lcao.selection.orbital_selector import mask_to_pointer, orbital_mask


def orbital_projected_denstiy_of_state(projector, select, energys):
    """Compute orbital-projected density of states.

    Contract: for identical inputs, the output `pdos` keeps the same shape and
    iteration order as before (`(ntarget, nenergy)`).
    """
    projector.load_context(need_wfsx_hsx=True)
    projector._target = []

    energys = np.array(energys)
    nenergy = len(energys)

    if isinstance(select, str):
        select = [select]
    orbital_mask(projector, select)
    mask_to_pointer(projector)

    gamma = projector.gamma
    wf = projector.wavefunction
    wk = projector.kweight
    kpt = projector.kpoints
    eig = projector.eigenvalue
    target = projector._target
    list_io = projector._projection_io
    list_ptr = projector._projection_ptr
    Sover = projector.Sover
    xij = projector.xij

    nwavefunctions = eig.shape[0]
    nspin = eig.shape[1]
    nkpoints = eig.shape[2]
    ntarget = len(target)

    pdos = np.zeros((ntarget, nenergy), dtype=float)
    buff0 = np.zeros((nenergy, nkpoints), dtype=float)
    buff1 = np.zeros((ntarget, nkpoints, nwavefunctions), dtype=float)

    for itar in range(ntarget):
        tar = target[itar]
        for ik in range(nkpoints):
            for isp in range(nspin):
                for iw in range(nwavefunctions):
                    nprojection = len(list_io[itar])
                    buff2 = np.zeros((nprojection), dtype=float)
                    list_target_io = tar['orbital_index']
                    for io1 in list_target_io:
                        iio = 0
                        for io2 in list_io[itar]:
                            ind = list_ptr[itar][iio]
                            qcos, qsin = compute_projection_factor(gamma, wf, io1, io2, iw, isp, ik)
                            real_weight, _ = accumulate_overlap_weight(Sover, xij, kpt[ik], ind, qcos, qsin)
                            buff2[iio] = real_weight
                            iio += 1
                    buff1[itar][ik][iw] = buff2.sum()

                    eigenvalue = eig[iw][isp][ik]
                    for ie in range(nenergy):
                        factor = projector.delta(energys[ie] - eigenvalue)
                        buff0[ie][ik] = factor * buff1[itar][ik].sum() * wk[ik]

        for ie in range(nenergy):
            pdos[itar][ie] = buff0[ie].sum()

    projector.pdos = pdos
    return pdos
