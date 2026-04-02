import numpy as np

from lcao.core.model import phi_tolerance
from lcao.core.orbital_m import normalize_orbital_m, validate_signed_orbital_m


def _build_unique_dm_pairs(projector):
    """Build unique (mu, nu) DM pairs in upper-triangular form.

    Returns arrays ``mu``, ``nu`` and ``dm_unique`` where ``mu <= nu``.
    If both (mu,nu) and (nu,mu) are present in the sparse DM rows, their
    values are averaged to make the representation order-independent.
    """
    nspin = projector.dm_ns
    pair_values = {}
    pair_counts = {}

    for io1 in range(projector.dm_nb):
        row_start = projector.dm_listdptr[io1]
        row_end = row_start + projector.dm_numd[io1]
        for ind in range(row_start, row_end):
            io2 = projector.dm_listd[ind] - 1
            mu = io1 if io1 <= io2 else io2
            nu = io2 if io1 <= io2 else io1
            key = (mu, nu)

            if key not in pair_values:
                pair_values[key] = projector.dm[ind].copy()
                pair_counts[key] = 1
            else:
                pair_values[key] += projector.dm[ind]
                pair_counts[key] += 1

    keys = sorted(pair_values.keys())
    mu = np.array([k[0] for k in keys], dtype=int)
    nu = np.array([k[1] for k in keys], dtype=int)
    dm_unique = np.zeros((len(keys), nspin), dtype=float)
    for idx, key in enumerate(keys):
        dm_unique[idx] = pair_values[key] / pair_counts[key]

    return mu, nu, dm_unique


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

        phir = projector.Rnl(
            atom_symbol,
            target_n,
            target_l,
            target_z,
            radius,
            io=io + 1,
            ia=projector.atom_index[io],
        )
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
    dm_mu, dm_nu, dm_unique = _build_unique_dm_pairs(projector)
    pair_factor = np.where(dm_mu == dm_nu, 1.0, 2.0)

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

                pair_product = (phi[dm_mu] * phi[dm_nu]).real
                for isp in range(nspin):
                    density_value = np.sum(dm_unique[:, isp] * pair_factor * pair_product)
                    rho[isp][ix][iy][iz] = float(density_value)

    projector.rho = rho
    return rho
