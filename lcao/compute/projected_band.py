import numpy as np

from lcao.selection.orbital_selector import mask_to_pointer, orbital_mask


def orbital_projected_bandstructure(projector, select):
    projector.load_context(need_wfsx_hsx=True)
    projector._target = []
    if isinstance(select, str):
        select = [select]
    orbital_mask(projector, select)
    mask_to_pointer(projector)

    gamma = projector.gamma
    wf = projector.wavefunction
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

    fat = np.zeros((ntarget, nkpoints, nwavefunctions), dtype=float)

    for itar in range(ntarget):
        tar = target[itar]
        for ik in range(nkpoints):
            for isp in range(nspin):
                for iw in range(nwavefunctions):
                    nprojection = len(list_io[itar])
                    buff = np.zeros((nprojection), dtype=float)
                    list_target_io = tar['orbital_index']
                    for io1 in list_target_io:
                        iio = 0
                        for io2 in list_io[itar]:
                            ind = list_ptr[itar][iio]
                            if gamma == 1:
                                qcos = wf[0][io1] * wf[0][io2]
                                qsin = 0
                            else:
                                qcos = wf[0][io1][iw][isp][ik] * wf[0][io2][iw][isp][ik] + wf[1][io1][iw][isp][ik] * wf[1][io2][iw][isp][ik]
                                qsin = wf[0][io1][iw][isp][ik] * wf[1][io2][iw][isp][ik] - wf[1][io1][iw][isp][ik] * wf[0][io2][iw][isp][ik]
                            phase = (kpt[ik] * xij[ind]).sum()
                            factor = qcos * np.cos(phase) - qsin * np.sin(phase)
                            buff[iio] = Sover[ind] * factor
                            iio += 1
                    fat[itar][ik][iw] = buff.sum()

    projector.fat = fat
    return fat
