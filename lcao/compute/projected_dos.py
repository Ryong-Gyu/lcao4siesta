import numpy as np

from lcao.compute.kernels import sum_projection_real_weights
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
        list_target_io = tar['orbital_index']
        target_projection_io = list_io[itar]
        target_projection_ptr = list_ptr[itar]
        for ik in range(nkpoints):
            for isp in range(nspin):
                for iw in range(nwavefunctions):
                    buff1[itar][ik][iw] = sum_projection_real_weights(
                        gamma,
                        wf,
                        kpt[ik],
                        list_target_io,
                        target_projection_io,
                        target_projection_ptr,
                        Sover,
                        xij,
                        iw,
                        isp,
                        ik,
                    )

                    eigenvalue = eig[iw][isp][ik]
                    for ie in range(nenergy):
                        factor = projector.delta(energys[ie] - eigenvalue)
                        buff0[ie][ik] = factor * buff1[itar][ik].sum() * wk[ik]

        for ie in range(nenergy):
            pdos[itar][ie] = buff0[ie].sum()

    projector.pdos = pdos
    return pdos
