import numpy as np

from lcao.core.model import phi_tolerance
from lcao.core.orbital_m import normalize_orbital_m, validate_signed_orbital_m


def _orbital_value_at_position(projector, io, position_vector, supercell_vectors):
    atom_symbol = projector.atom_species[io]
    atom_index = projector.atom_index[io] - 1

    target_n = projector.orbital_n[io]
    target_l = projector.orbital_l[io]
    # ORB_INDX magnetic quantum number encoding:
    # - signed: m in [-l, ..., +l] (use directly)
    # - legacy: ml in [1, ..., 2l+1] (convert by ml - l - 1)
    target_m = normalize_orbital_m(
        projector.orbital_ml[io],
        target_l,
        source='ORB_INDX',
        orbital_index=io + 1,
        file_path=f'{projector._system}.ORB_INDX',
    )
    target_z = projector.orbital_zeta[io]

    target_position = projector.atoms[atom_index]

    value = 0.0 + 0.0j
    for vector in supercell_vectors:
        xji = -(target_position - position_vector + vector)
        radius = np.sqrt(xji.dot(xji))

        phir = projector.Rnl(atom_symbol, target_n, target_l, target_z, radius)
        if abs(phir) < phi_tolerance:
            continue

        validate_signed_orbital_m(
            target_m,
            target_l,
            source='ORB_INDX',
            orbital_index=io + 1,
            file_path=f'{projector._system}.ORB_INDX',
        )
        spherical = projector.Yml(xji, target_m, target_l)
        value += phir * spherical

    return value


def electron_density(projector, cell, mesh):
    """Compute real-space electron density from the density matrix.

    The density on each grid point is evaluated as
    ``rho(r) = sum_{mu,nu} DM_{mu,nu} * phi_mu(r) * phi_nu(r)``.
    """
    projector.load_context(need_struct_supercell=True, need_orbital_metadata=True)

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
