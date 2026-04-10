import numpy as np
from lcao.core.orbital_m import normalize_orbital_m, validate_signed_orbital_m
from lcao.compute.kernels import build_mesh_positions

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


def _build_species_r2cut_coarse(projector, io_domain):
    """Build species-level max(cutoff^2) map used as a coarse gate.

    This mirrors rhoofd.F90 `r2cut(is)` intent: a fast species-level rejection
    before evaluating detailed orbital contributions.
    """
    r2cut_species = {}
    for io in io_domain:
        io_int = int(io)
        if hasattr(projector, 'io_to_iuo'):
            iuo = int(projector.io_to_iuo[io_int])
        else:
            iuo = int(projector.dm_orbital_iuo[io_int])
        atom_symbol = projector.iuo_to_symbol[iuo]
        cutoff = _io_cutoff_radius_from_pao(projector, io_int)
        cutoff2 = cutoff * cutoff
        previous = r2cut_species.get(atom_symbol, 0.0)
        if cutoff2 > previous:
            r2cut_species[atom_symbol] = cutoff2
    return r2cut_species


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


def _orbital_value_at_position(
    projector,
    io,
    center_io,
    position_vector,
    supercell_vectors,
    r2cut_species,
):
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
    coarse_r2cut = float(r2cut_species.get(atom_symbol, 0.0))
    orbital_cutoff = _io_cutoff_radius_from_pao(projector, io_int)
    orbital_cutoff2 = orbital_cutoff * orbital_cutoff
    for vector in supercell_vectors:
        xji = -(center_io - position_vector + vector)
        r2 = xji.dot(xji)
        # Stage-1 coarse filter (rhoofd.F90 r2cut-like species gate):
        # skip expensive per-orbital radial evaluation when the point is
        # outside the maximum cutoff radius of this species.
        if r2 >= coarse_r2cut:
            continue
        # Stage-2 detailed orbital filter:
        # preserve exact per-orbital cutoff behavior for numerical fidelity.
        if r2 >= orbital_cutoff2:
            continue

        radius = np.sqrt(r2)
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


def build_meshphi_active_index(projector, positions, io_domain, supercell_vectors):
    """Build meshphi-like sparse active-orbital index (endpht/lstpht/listp2)."""
    npoint = int(positions.shape[0])
    nio = int(io_domain.shape[0])
    counts = np.zeros((npoint,), dtype=np.int64)
    active_per_point = []

    for ip in range(npoint):
        position_vector = positions[ip]
        active_io = []
        for idx in range(nio):
            io = int(io_domain[idx])
            cutoff = _io_cutoff_radius_from_pao(projector, io)
            cutoff2 = cutoff * cutoff
            if hasattr(projector, 'io_to_center_io'):
                center_io = projector.io_to_center_io[io]
            else:
                center_io = projector.dm_center_io[io]

            for vector in supercell_vectors:
                xji = -(center_io - position_vector + vector)
                if xji.dot(xji) < cutoff2:
                    active_io.append(io)
                    break
        active_array = np.asarray(active_io, dtype=np.int64)
        active_per_point.append(active_array)
        counts[ip] = int(active_array.shape[0])

    endpht = np.zeros((npoint + 1,), dtype=np.int64)
    if npoint > 0:
        endpht[1:] = np.cumsum(counts)

    nlist = int(endpht[npoint])
    lstpht = np.zeros((nlist,), dtype=np.int64)
    listp2 = np.zeros((nlist,), dtype=np.int64)

    for ip in range(npoint):
        start = int(endpht[ip])
        stop = int(endpht[ip + 1])
        nactive = stop - start
        if nactive <= 0:
            continue
        lstpht[start:stop] = active_per_point[ip]
        listp2[start:stop] = np.arange(start, stop, dtype=np.int64)

    return {'endpht': endpht, 'lstpht': lstpht, 'listp2': listp2}


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
            projector._r2cut_species,
        )
    return phi_active


@njit(cache=True)
def _idx_ijl(ilocal, jlocal):
    i = ilocal
    j = jlocal
    if i > j:
        i, j = j, i
    return (j * (j + 1)) // 2 + i


def _init_local_dm_cache(io_domain_size, max_active, nspin):
    ntri = (max_active * (max_active + 1)) // 2
    return {
        # Fortran rhoofd.F90 ilocal equivalent: io -> local active index
        'ilocal': np.full((io_domain_size,), -1, dtype=np.int64),
        # Fortran rhoofd.F90 iorb equivalent: local active index -> io
        'iorb': np.full((max_active,), -1, dtype=np.int64),
        # Fortran rhoofd.F90 Dlocal equivalent: triangular local DM cache
        'Dlocal': np.full((ntri, nspin), np.nan, dtype=np.float64),
        # Current active size for iorb/Dlocal view
        'nactive': 0,
    }


def _build_local_dm_cache(projector, dm_columns, active_io, cache):
    """Update local triangular DM cache (ilocal/iorb/Dlocal/idx_ijl style).

    Only rows newly required by an active-set update are scanned from sparse DM.
    Retained active-orbital pairs are copied from the previous cache.
    """
    nspin = int(projector.dm_ns)
    nactive = int(active_io.shape[0])
    ilocal = cache['ilocal']
    iorb_prev = cache['iorb'].copy()
    dlocal_prev = cache['Dlocal'].copy()
    nactive_prev = int(cache['nactive'])

    ilocal.fill(-1)
    iorb = cache['iorb']
    iorb[:nactive] = active_io
    if nactive < iorb.shape[0]:
        iorb[nactive:] = -1
    for idx in range(nactive):
        io = int(active_io[idx])
        if 0 <= io < ilocal.shape[0]:
            ilocal[io] = idx

    dlocal = cache['Dlocal']
    dlocal.fill(np.nan)

    # Reuse overlapping pairs from previous local cache.
    ilocal_prev = np.full((ilocal.shape[0],), -1, dtype=np.int64)
    for idx in range(nactive_prev):
        io = int(iorb_prev[idx])
        if 0 <= io < ilocal_prev.shape[0]:
            ilocal_prev[io] = idx
    for j in range(nactive):
        io_j = int(active_io[j])
        pj = ilocal_prev[io_j] if 0 <= io_j < ilocal_prev.shape[0] else -1
        if pj < 0:
            continue
        for i in range(j + 1):
            io_i = int(active_io[i])
            pi = ilocal_prev[io_i] if 0 <= io_i < ilocal_prev.shape[0] else -1
            if pi < 0:
                continue
            idx_new = _idx_ijl(i, j)
            idx_old = _idx_ijl(pi, pj)
            for isp in range(nspin):
                dlocal[idx_new, isp] = dlocal_prev[idx_old, isp]

    # Determine sparse-DM rows needed for missing triangular entries.
    needed_rows = set()
    for j in range(nactive):
        io_j = int(active_io[j])
        for i in range(j + 1):
            idx_tri = _idx_ijl(i, j)
            if not np.isnan(dlocal[idx_tri, 0]):
                continue
            io_i = int(active_io[i])
            row_io = io_i if io_i <= io_j else io_j
            if 0 <= row_io < projector.dm_nb:
                needed_rows.add(int(row_io))

    if not needed_rows:
        cache['nactive'] = nactive
        return dlocal

    active_mask = np.zeros((ilocal.shape[0],), dtype=np.bool_)
    for idx in range(nactive):
        io = int(active_io[idx])
        if 0 <= io < active_mask.shape[0]:
            active_mask[io] = True

    for row_io in needed_rows:
        row_start = int(projector.dm_listdptr[row_io])
        row_end = row_start + int(projector.dm_numd[row_io])
        for ind in range(row_start, row_end):
            col_io = int(dm_columns[ind])
            if col_io < 0 or col_io >= ilocal.shape[0]:
                continue
            if not active_mask[col_io]:
                continue
            i_io = row_io if row_io <= col_io else col_io
            j_io = col_io if row_io <= col_io else row_io
            li = int(ilocal[i_io])
            lj = int(ilocal[j_io])
            if li < 0 or lj < 0:
                continue
            idx_tri = _idx_ijl(li, lj)
            for isp in range(nspin):
                dlocal[idx_tri, isp] = projector.dm[ind, isp]

    # Ensure uncoupled missing entries are exact zeros.
    dlocal[np.isnan(dlocal)] = 0.0
    cache['nactive'] = nactive
    return dlocal


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


@njit(cache=True)
def _accumulate_density_from_dlocal_tri(dlocal, phi_active, nspin):
    density_value = np.zeros((nspin), dtype=np.float64)
    nactive = phi_active.shape[0]
    if nactive == 0:
        return density_value
    for j in range(nactive):
        phi_j = phi_active[j]
        for i in range(j + 1):
            phi_i = phi_active[i]
            idx_tri = _idx_ijl(i, j)
            pair_product = (phi_i * phi_j).real
            factor = 1.0 if i == j else 2.0
            weighted_pair = factor * pair_product
            for isp in range(nspin):
                density_value[isp] += dlocal[idx_tri, isp] * weighted_pair
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


def accumulate_rho_from_sparse_dm_cached(projector, dm_columns, active_io, phi_active, local_dm_cache):
    dlocal = _build_local_dm_cache(projector, dm_columns, active_io, local_dm_cache)
    return _accumulate_density_from_dlocal_tri(dlocal, phi_active, projector.dm_ns)


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

    io_domain = np.unique(np.concatenate((np.arange(projector.dm_nb, dtype=np.int64), dm_columns)))
    projector._r2cut_species = _build_species_r2cut_coarse(projector, io_domain)
    # Representation modes:
    # - unit-cell io domain: DM connectivity only references 0..dm_nb-1,
    #   and periodic images are generated by explicit translation summation.
    # - explicit supercell io domain: DM connectivity includes io>=dm_nb
    #   already carrying ORB_INDX isc shifts; do not add translation images again.
    uses_explicit_supercell_io = np.any(io_domain >= projector.dm_nb)
    if uses_explicit_supercell_io:
        supercell_vectors = np.zeros((1, 3), dtype=float)

    positions, grid_indices = build_mesh_positions(xgrid[0], ygrid[0], zgrid[0])
    meshphi_context = build_meshphi_active_index(projector, positions, io_domain, supercell_vectors)
    projector.meshphi_active_context = meshphi_context
    endpht = meshphi_context['endpht']
    lstpht = meshphi_context['lstpht']

    npoint = int(positions.shape[0])
    max_active = int(np.max(endpht[1:] - endpht[:-1])) if npoint > 0 else 0
    local_dm_cache = _init_local_dm_cache(io_domain_size, max_active, nspin)
    for ip in range(npoint):
        ix = int(grid_indices[ip, 0])
        iy = int(grid_indices[ip, 1])
        iz = int(grid_indices[ip, 2])
        position_vector = positions[ip]
        active_count = int(endpht[ip + 1] - endpht[ip])
        if active_count > 0:
            imp_start = int(endpht[ip])
            imp_stop = int(endpht[ip + 1])
            active_io = lstpht[imp_start:imp_stop]
        else:
            active_io = np.zeros((0,), dtype=np.int64)
        phi_active = evaluate_phi_for_active_io(projector, active_io, position_vector, supercell_vectors)
        density_value = accumulate_rho_from_sparse_dm_cached(
            projector, dm_columns, active_io, phi_active, local_dm_cache
        )
        for isp in range(nspin):
            rho[isp][ix][iy][iz] = float(density_value[isp])

    projector.rho = rho
    return rho
