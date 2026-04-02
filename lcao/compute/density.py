import numpy as np

from lcao.core.model import phi_tolerance
from lcao.core.orbital_m import normalize_orbital_m, validate_signed_orbital_m


def _build_unique_dm_pairs(projector, dm_columns):
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
            io2 = dm_columns[ind]
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


def _dm_columns_zero_based(projector):
    """Return sparse-DM column indices normalized to Python 0-based indexing.

    SIESTA Fortran files are 1-based, but some Python readers may already convert
    to 0-based. Accept both conventions, then normalize here.
    """
    dm_cols = projector.dm_listd

    is_one_based = np.all((dm_cols >= 1) & (dm_cols <= projector.dm_nb))
    if is_one_based:
        return dm_cols - 1

    is_zero_based = np.all((dm_cols >= 0) & (dm_cols < projector.dm_nb))
    if is_zero_based:
        return dm_cols

    if hasattr(projector, 'supercell_orbital_io') and hasattr(projector, 'supercell_orbital_iuo'):
        io_map = {
            int(io): int(iuo) - 1
            for io, iuo in zip(projector.supercell_orbital_io, projector.supercell_orbital_iuo)
        }

        can_map_fortran = np.all(dm_cols >= 1) and np.all([int(col) in io_map for col in dm_cols])
        if can_map_fortran:
            mapped = np.array([io_map[int(col)] for col in dm_cols], dtype=int)
            if np.all((mapped >= 0) & (mapped < projector.dm_nb)):
                return mapped

        can_map_zero = np.all(dm_cols >= 0) and np.all([int(col + 1) in io_map for col in dm_cols])
        if can_map_zero:
            mapped = np.array([io_map[int(col + 1)] for col in dm_cols], dtype=int)
            if np.all((mapped >= 0) & (mapped < projector.dm_nb)):
                return mapped

    bad_pos = np.where((dm_cols < 0) | (dm_cols > projector.dm_nb))[0]
    first_pos = int(bad_pos[0]) if len(bad_pos) else 0
    bad_value = int(dm_cols[first_pos])
    row = int(np.searchsorted(projector.dm_listdptr, first_pos, side='right') - 1)
    row = max(0, min(row, projector.dm_nb - 1))

    raise ValueError(
        'DM connectivity contains invalid orbital indices after checking both '
        f'Fortran(1-based) and Python(0-based) conventions: '
        f'dm_listd[{first_pos}]={bad_value} (row io={row + 1}), basis size={projector.dm_nb}. '
        f'Likely cause: inconsistent files ({projector._dm_file} vs {projector._system}.ORB_INDX).'
    )


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
    dm_columns = _dm_columns_zero_based(projector)

    xgrid, ygrid, zgrid = projector.unit_cell_grid(cell, mesh)

    na = int(mesh[0])
    nb = int(mesh[1])
    nc = int(mesh[2])

    nbasis = projector.dm_nb
    nspin = projector.dm_ns
    dm_mu, dm_nu, dm_unique = _build_unique_dm_pairs(projector, dm_columns)
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
