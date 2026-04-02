import numpy as np

from lcao.core.model import overlap_tolerance


def orbital_mask(projector, select):
    label = projector._atom_symbol
    za = projector.atom_index
    zn = projector.orbital_n
    zl = projector.orbital_l
    zx = projector.orbital_ml
    zz = projector.orbital_zeta

    for index in select:
        indexes = index.split('_')
        islabel = 0

        try:
            atom_index = int(indexes[0])
        except ValueError:
            atom_label = indexes[0]
            islabel = 1

        if islabel:
            ibuff = np.where(label == atom_label)[0]
        else:
            ibuff = np.where(za == atom_index)[0]

        if len(indexes) >= 2:
            atom_n = int(indexes[1])
            ibuff = [i for i in ibuff if zn[i] == atom_n]

            if len(indexes) >= 3:
                atom_l = int(indexes[2])
                ibuff = [i for i in ibuff if zl[i] == atom_l]

                if len(indexes) >= 4:
                    atom_m = int(indexes[3])
                    ibuff = [i for i in ibuff if zx[i] == atom_m + zl[i] + 1]

                    if len(indexes) == 5:
                        atom_zeta = int(indexes[4])
                        ibuff = [i for i in ibuff if zz[i] == atom_zeta]

        m_iao = ibuff
        nao = len(m_iao)
        m_ia = np.zeros((nao), dtype=int)
        m_izn = np.zeros((nao), dtype=int)
        m_izl = np.zeros((nao), dtype=int)
        m_izm = np.zeros((nao), dtype=int)
        m_izz = np.zeros((nao), dtype=int)

        for i in range(nao):
            idx = m_iao[i]
            m_ia[i] = za[idx]
            m_izn[i] = zn[idx]
            m_izl[i] = zl[idx]
            m_izm[i] = zx[idx] - zl[idx] - 1
            m_izz[i] = zz[idx]

        projector._target.append({
            'number_of_components': nao,
            'atomic_index': m_ia,
            'orbital_index': m_iao,
            'principle_quantum_number': m_izn,
            'angular_quantum_number': m_izl,
            'magnetic_quantum_nuber': m_izm,
            'zeta': m_izz,
        })


def mask_to_pointer(projector):
    target = projector._target
    numh = projector.numh
    listhptr = projector.listhptr
    listh = projector.listh
    indxuo = projector.indxuo
    Sover = projector.Sover

    list_ptr = []
    list_io = []

    for selected in target:
        list_ptr.append([])
        list_io.append([])
        iao_list = selected['orbital_index']

        for j1 in iao_list:
            for k1 in range(numh[j1]):
                ind = listhptr[j1] + k1
                io = indxuo[listh[ind] - 1]
                if abs(Sover[ind]) >= overlap_tolerance:
                    list_ptr[-1].append(ind)
                    list_io[-1].append(io - 1)

    projector._projection_ptr = np.array(list_ptr)
    projector._projection_io = np.array(list_io)
