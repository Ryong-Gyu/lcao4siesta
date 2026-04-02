import numpy as np

from lcao.core.model import phi_tolerance


def _normalize_atom_symbol(projector, atom_symbol):
    if atom_symbol in projector.ions:
        return atom_symbol
    if len(projector.ions) == 1:
        return next(iter(projector.ions.keys()))
    raise ValueError(
        f'Cannot map atom species "{atom_symbol}" to an ion basis key. '
        'Provide HSX metadata or use a single-species ion_files mapping.'
    )


def _prepare_density_orbital_metadata(projector):
    if hasattr(projector, 'atom_index') and hasattr(projector, 'orbital_n'):
        return

    if len(projector.ions) != 1:
        raise ValueError(
            'electron_density without HSX currently supports only single-species ion_files. '
            'For multi-species systems, provide HSX metadata.'
        )

    symbol = next(iter(projector.ions.keys()))
    basis = projector.ions[symbol]

    atom_index = []
    atom_species = []
    orbital_n = []
    orbital_l = []
    orbital_ml = []
    orbital_zeta = []

    for ia in range(len(projector.atoms)):
        for n in sorted(basis.keys()):
            for l in sorted(basis[n].keys()):
                for zeta in sorted(basis[n][l].keys()):
                    for m in range(1, 2 * l + 2):
                        atom_index.append(ia + 1)
                        atom_species.append(symbol)
                        orbital_n.append(n)
                        orbital_l.append(l)
                        orbital_ml.append(m)
                        orbital_zeta.append(zeta)

    if len(atom_index) != projector.dm_nb:
        raise ValueError(
            f'Inferred orbital count ({len(atom_index)}) does not match DM basis size ({projector.dm_nb}). '
            'Provide HSX metadata if basis ordering differs.'
        )

    projector.atom_index = np.array(atom_index, dtype=int)
    projector.atom_species = np.array(atom_species)
    projector.orbital_n = np.array(orbital_n, dtype=int)
    projector.orbital_l = np.array(orbital_l, dtype=int)
    projector.orbital_ml = np.array(orbital_ml, dtype=int)
    projector.orbital_zeta = np.array(orbital_zeta, dtype=int)


def _orbital_value_at_position(projector, io, position_vector, supercell_vectors):
    atom_symbol = _normalize_atom_symbol(projector, projector.atom_species[io])
    atom_index = projector.atom_index[io] - 1

    target_n = projector.orbital_n[io]
    target_l = projector.orbital_l[io]
    target_m = projector.orbital_ml[io]
    target_z = projector.orbital_zeta[io]

    target_position = projector.atoms[atom_index]

    value = 0.0 + 0.0j
    for vector in supercell_vectors:
        xji = -(target_position - position_vector + vector)
        radius = np.sqrt(xji.dot(xji))

        phir = projector.Rnl(atom_symbol, target_n, target_l, target_z, radius)
        if abs(phir) < phi_tolerance:
            continue

        spherical = projector.Yml(xji, target_m, target_l)
        value += phir * spherical

    return value


def electron_density(projector, cell, mesh):
    """Compute real-space electron density from the density matrix.

    The density on each grid point is evaluated as
    ``rho(r) = sum_{mu,nu} DM_{mu,nu} * phi_mu(r) * phi_nu(r)``.
    """
    projector.load_context(need_struct_supercell=True)
    _prepare_density_orbital_metadata(projector)

    xgrid, ygrid, zgrid = projector.unit_cell_grid(cell, mesh)

    na = int(mesh[0])
    nb = int(mesh[1])
    nc = int(mesh[2])

    nbasis = projector.dm_nb
    nspin = projector.dm_ns

    rho = np.zeros((nspin, na, nb, nc), dtype=float)
    supercell_vectors = projector._supercell_vector_list

    for ix in range(na):
        for iy in range(nb):
            for iz in range(nc):
                position_vector = np.array(
                    [
                        xgrid[0][ix][iy][iz],
                        ygrid[0][ix][iy][iz],
                        zgrid[0][ix][iy][iz],
                    ],
                    dtype=float,
                )

                phi = np.zeros((nbasis), dtype=np.complex128)
                for io in range(nbasis):
                    phi[io] = _orbital_value_at_position(projector, io, position_vector, supercell_vectors)

                for isp in range(nspin):
                    density_value = 0.0
                    for io1 in range(nbasis):
                        row_start = projector.dm_listdptr[io1]
                        row_end = row_start + projector.dm_numd[io1]
                        for ind in range(row_start, row_end):
                            io2 = projector.dm_listd[ind] - 1
                            density_value += projector.dm[ind][isp] * (phi[io1] * phi[io2]).real

                    rho[isp][ix][iy][iz] = density_value

    projector.rho = rho
    return rho
