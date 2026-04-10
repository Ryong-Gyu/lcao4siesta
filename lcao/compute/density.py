import numpy as np

from lcao.compute.kernels import build_mesh_positions
from lcao.core.orbital_m import normalize_orbital_m, validate_signed_orbital_m

try:
    from numba import njit
except ImportError:  # pragma: no cover - optional dependency
    def njit(*args, **kwargs):
        def _decorator(func):
            return func

        return _decorator


def _io_cutoff_radius_from_pao(projector, io):
    """Return PAO cutoff radius for an ORB_INDX io via io->iuo metadata."""
    io_int = int(io)
    if hasattr(projector, 'io_to_iuo'):
        iuo = int(projector.io_to_iuo[io_int])
    else:
        iuo = int(projector.dm_orbital_iuo[io_int])
    atom_symbol = projector.iuo_to_symbol[iuo]
    target_n = projector.iuo_to_n[iuo]
    target_l = projector.iuo_to_l[iuo]
    target_z = projector.iuo_to_zeta[iuo]
    return projector.ions[atom_symbol][target_n][target_l][target_z]['cutoff']


def _dm_columns_zero_based(projector):
    """Return sparse-DM column indices as Python 0-based indexes.

    Convention:
    - DM/HSX/ORB_INDX orbital ids are treated as Fortran 1-based ids.
    - Conversion to Python indexing is done exactly once here via ``- 1``.
    """
    dm_cols = np.asarray(projector.dm_listd, dtype=np.int64)

    known_io_max = None
    if hasattr(projector, 'io_to_iuo') and len(projector.io_to_iuo) > 0:
        known_io_max = max(projector.io_to_iuo.keys())

    # 0-based io indexing
    if known_io_max is not None and np.all((dm_cols >= 0) & (dm_cols <= known_io_max)):
        return dm_cols

    # 1-based io indexing (standard SIESTA DM encoding)
    if known_io_max is not None and np.all((dm_cols >= 1) & (dm_cols <= (known_io_max + 1))):
        return dm_cols - 1

    # Fallback compatibility: unit-cell-only domains.
    if np.all((dm_cols >= 0) & (dm_cols < projector.dm_nb)):
        return dm_cols
    if np.all((dm_cols >= 1) & (dm_cols <= projector.dm_nb)):
        return dm_cols - 1

    upper_bound = projector.dm_nb if known_io_max is None else (known_io_max + 1)
    bad_pos = np.where((dm_cols < 1) | (dm_cols > upper_bound))[0]
    first_pos = int(bad_pos[0]) if len(bad_pos) else 0
    bad_value = int(dm_cols[first_pos])
    row = int(np.searchsorted(projector.dm_listdptr, first_pos, side='right') - 1)
    row = max(0, min(row, projector.dm_nb - 1))

    raise ValueError(
        'DM connectivity contains invalid orbital indices for Fortran 1-based '
        'convention. '
        f'dm_listd[{first_pos}]={bad_value} (row io={row}), basis size={projector.dm_nb}. '
        f'Likely cause: inconsistent files ({projector._dm_file} vs {projector._system}.ORB_INDX).'
    )


def _orbital_value_at_position(projector, io, center_io, position_vector, supercell_vectors):
    io_int = int(io)
    if hasattr(projector, 'io_to_iuo'):
        iuo = int(projector.io_to_iuo[io_int])
    else:
        iuo = int(projector.dm_orbital_iuo[io_int])
    atom_symbol = projector.iuo_to_symbol[iuo]

    target_n = projector.iuo_to_n[iuo]
    target_l = projector.iuo_to_l[iuo]
    # ORB_INDX magnetic quantum number encoding:
    # - signed: m in [-l, ..., +l] (use directly)
    # - legacy: ml in [1, ..., 2l+1] (convert by ml - l - 1)
    target_m = normalize_orbital_m(
        projector.iuo_to_m[iuo],
        target_l,
        source='ORB_INDX',
        orbital_index=iuo,
        file_path=f'{projector._system}.ORB_INDX',
    )
    target_z = projector.iuo_to_zeta[iuo]
    validate_signed_orbital_m(
        target_m,
        target_l,
        source='ORB_INDX',
        orbital_index=iuo,
        file_path=f'{projector._system}.ORB_INDX',
    )

    value = 0.0 + 0.0j
    for vector in supercell_vectors:
        xji = -(center_io - position_vector + vector)
        radius = np.sqrt(xji.dot(xji))
        phir = projector.Rnl(
            atom_symbol,
            target_n,
            target_l,
            target_z,
            radius,
            io=io,
            ia=None,
        )
        # Rnl already applies the orbital cutoff radius explicitly.
        # Keep only exact zeros from the cutoff path, without an extra
        # tolerance-based truncation.
        if phir == 0.0:
            continue

        spherical = projector.Yml(xji, target_m, target_l)
        value += phir * spherical

    return value


def build_active_io_per_mesh(projector, positions, io_domain, supercell_vectors):
    """Build active-io list for each mesh point from io-center and PAO cutoff."""
    nio = io_domain.shape[0]
    npoint = positions.shape[0]

    io_cutoff = np.zeros((nio), dtype=float)
    io_centers = np.zeros((nio, 3), dtype=float)
    for idx in range(nio):
        io = int(io_domain[idx])
        io_cutoff[idx] = _io_cutoff_radius_from_pao(projector, io)
        if hasattr(projector, 'io_to_center_io'):
            io_centers[idx] = projector.io_to_center_io[io]
        else:
            io_centers[idx] = projector.dm_center_io[io]
    io_cutoff2 = io_cutoff * io_cutoff

    active_io_per_mesh = []
    for ip in range(npoint):
        position_vector = positions[ip]
        active_io = []
        for idx in range(nio):
            io = int(io_domain[idx])
            center_io = io_centers[idx]
            cutoff2 = io_cutoff2[idx]
            is_active = False
            for vector in supercell_vectors:
                xji = -(center_io - position_vector + vector)
                if xji.dot(xji) < cutoff2:
                    is_active = True
                    break
            if is_active:
                active_io.append(io)
        active_io_per_mesh.append(np.array(active_io, dtype=np.int64))

    return active_io_per_mesh


def evaluate_phi_for_active_io(projector, active_io, position_vector, supercell_vectors):
    """Evaluate phi(io, ip) only for active io on the current mesh point."""
    nactive = active_io.shape[0]
    phi_active = np.zeros((nactive), dtype=np.complex128)
    for idx in range(nactive):
        io = int(active_io[idx])
        if hasattr(projector, 'io_to_center_io'):
            center_io = projector.io_to_center_io[io]
        else:
            center_io = projector.dm_center_io[io]
        phi_active[idx] = _orbital_value_at_position(
            projector,
            io,
            center_io,
            position_vector,
            supercell_vectors,
        )
    return phi_active


@njit(cache=True)
def _accumulate_density_from_pairs(
    dm, dm_listdptr, dm_numd, dm_columns, active_io, phi_active, nspin, row_basis_size, io_domain_size
):
    density_value = np.zeros((nspin), dtype=np.float64)
    if active_io.shape[0] == 0:
        return density_value

    active_mask = np.zeros((io_domain_size), dtype=np.bool_)
    active_phi_pos = np.full((io_domain_size), -1, dtype=np.int64)
    for idx in range(active_io.shape[0]):
        io = int(active_io[idx])
        if io < 0 or io >= io_domain_size:
            continue
        active_mask[io] = True
        active_phi_pos[io] = idx

    for io1 in range(row_basis_size):
        if not active_mask[io1]:
            continue
        idx1 = int(active_phi_pos[io1])
        row_start = dm_listdptr[io1]
        row_end = row_start + dm_numd[io1]
        phi1 = phi_active[idx1]

        for ind in range(row_start, row_end):
            io2 = int(dm_columns[ind])
            if io2 < 0 or io2 >= io_domain_size:
                continue
            if not active_mask[io2]:
                continue
            if io1 > io2:
                continue

            idx2 = int(active_phi_pos[io2])
            phi2 = phi_active[idx2]
            # With real spherical harmonics, phi values are real and the
            # density contribution uses a plain product.
            pair_product = (phi1 * phi2).real
            factor = 1.0 if io1 == io2 else 2.0
            weighted_pair = factor * pair_product
            for isp in range(nspin):
                density_value[isp] += dm[ind, isp] * weighted_pair

    return density_value


def accumulate_rho_from_sparse_dm(projector, dm_columns, active_io, phi_active, io_domain_size):
    """Accumulate rho from sparse DM, limited to active io pairs only.

    Symmetry handling follows rhoofd.F90 upper-triangular accumulation:
    keep io1 <= io2 pairs, apply factor 2.0 for off-diagonal terms.
    """
    return _accumulate_density_from_pairs(
        projector.dm,
        projector.dm_listdptr,
        projector.dm_numd,
        dm_columns,
        active_io,
        phi_active,
        projector.dm_ns,
        projector.dm_nb,
        io_domain_size,
    )


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

    nspin = projector.dm_ns
    io_domain_size = projector.dm_nb
    if hasattr(projector, 'io_all') and len(projector.io_all) > 0:
        io_domain_size = int(np.max(projector.io_all)) + 1

    rho = np.zeros((nspin, na, nb, nc), dtype=float)
    supercell_vectors = projector._supercell_vector_list

    positions, grid_indices = build_mesh_positions(xgrid[0], ygrid[0], zgrid[0])
    npoint = positions.shape[0]
    io_domain = np.unique(np.concatenate((np.arange(projector.dm_nb, dtype=np.int64), dm_columns)))
    # Representation modes:
    # - unit-cell io domain: DM connectivity only references 0..dm_nb-1,
    #   and periodic images are generated by explicit translation summation.
    # - explicit supercell io domain: DM connectivity includes io>=dm_nb
    #   already carrying ORB_INDX isc shifts; do not add translation images again.
    uses_explicit_supercell_io = np.any(io_domain >= projector.dm_nb)
    if uses_explicit_supercell_io:
        supercell_vectors = np.zeros((1, 3), dtype=float)

    active_io_per_mesh = build_active_io_per_mesh(projector, positions, io_domain, supercell_vectors)

    for ip in range(npoint):
        ix = int(grid_indices[ip, 0])
        iy = int(grid_indices[ip, 1])
        iz = int(grid_indices[ip, 2])
        position_vector = positions[ip]

        active_io = active_io_per_mesh[ip]
        phi_active = evaluate_phi_for_active_io(projector, active_io, position_vector, supercell_vectors)
        density_value = accumulate_rho_from_sparse_dm(projector, dm_columns, active_io, phi_active, io_domain_size)
        for isp in range(nspin):
            rho[isp][ix][iy][iz] = float(density_value[isp])

    projector.rho = rho
    return rho
