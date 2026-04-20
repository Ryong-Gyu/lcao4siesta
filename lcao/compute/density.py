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


def _orbital_value_at_position(
    projector,
    io,
    orbital_center,
    position_vector,
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
    xji = -(orbital_center - position_vector)
    r2 = xji.dot(xji)
    # Stage-1 coarse filter (rhoofd.F90 r2cut-like species gate):
    # skip expensive per-orbital radial evaluation when the point is
    # outside the maximum cutoff radius of this species.
    if r2 >= coarse_r2cut:
        return value
    # Stage-2 detailed orbital filter:
    # preserve exact per-orbital cutoff behavior for numerical fidelity.
    if r2 >= orbital_cutoff2:
        return value

    radius = np.sqrt(r2)
    phir = projector.Rnl(
        atom_symbol,
        target_n,
        target_l,
        target_z,
        radius,
    )
    # Rnl already applies the orbital cutoff radius explicitly.
    # Keep only exact zeros from the cutoff path, without an extra
    # tolerance-based truncation.
    if phir == 0.0:
        return value

    spherical = projector.Yml(xji, target_m, target_l)
    value += phir * spherical

    return value


def build_meshphi_active_index(projector, positions, io_domain, io_centers):
    """Build meshphi-like sparse active-orbital index (endpht/lstpht)."""
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
            orbital_center = io_centers[io]

            xji = -(orbital_center - position_vector)
            if xji.dot(xji) < cutoff2:
                active_io.append(io)
        active_array = np.asarray(active_io, dtype=np.int64)
        active_per_point.append(active_array)
        counts[ip] = int(active_array.shape[0])

    endpht = np.zeros((npoint + 1,), dtype=np.int64)
    if npoint > 0:
        endpht[1:] = np.cumsum(counts)

    nlist = int(endpht[npoint])
    lstpht = np.zeros((nlist,), dtype=np.int64)
    for ip in range(npoint):
        start = int(endpht[ip])
        stop = int(endpht[ip + 1])
        nactive = stop - start
        if nactive <= 0:
            continue
        lstpht[start:stop] = active_per_point[ip]
    return {'endpht': endpht, 'lstpht': lstpht}


def evaluate_phi_for_active_io(projector, active_io, position_vector, io_centers):
    """Evaluate phi(io, ip) only for active io on the current mesh point."""
    nactive = active_io.shape[0]
    phi_active = np.zeros((nactive), dtype=np.complex128)
    for idx in range(nactive):
        io = int(active_io[idx])
        orbital_center = io_centers[io]
        phi_active[idx] = _orbital_value_at_position(
            projector,
            io,
            orbital_center,
            position_vector,
            projector._r2cut_species,
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
    io_centers = projector.io_to_center_io
    dm_columns = np.asarray(projector.dm_listd, dtype=np.int64)

    xgrid, ygrid, zgrid = projector.unit_cell_grid(cell, mesh)

    na = int(mesh[0])
    nb = int(mesh[1])
    nc = int(mesh[2])

    nspin = projector.dm_ns
    io_domain_size = projector.dm_nb
    if hasattr(projector, 'io_all') and len(projector.io_all) > 0:
        io_domain_size = int(np.max(projector.io_all)) + 1

    rho = np.zeros((nspin, na, nb, nc), dtype=float)
    io_domain = np.unique(np.concatenate((np.arange(projector.dm_nb, dtype=np.int64), dm_columns)))
    projector._r2cut_species = _build_species_r2cut_coarse(projector, io_domain)

    positions, grid_indices = build_mesh_positions(xgrid[0], ygrid[0], zgrid[0])
    meshphi_context = build_meshphi_active_index(projector, positions, io_domain, io_centers)
    projector.meshphi_active_context = meshphi_context
    endpht = meshphi_context['endpht']
    lstpht = meshphi_context['lstpht']

    npoint = int(positions.shape[0])
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
        phi_active = evaluate_phi_for_active_io(projector, active_io, position_vector, io_centers)
        density_value = accumulate_rho_from_sparse_dm(projector, dm_columns, active_io, phi_active, io_domain_size)
        for isp in range(nspin):
            rho[isp][ix][iy][iz] = float(density_value[isp])

    projector.rho = rho
    return rho
