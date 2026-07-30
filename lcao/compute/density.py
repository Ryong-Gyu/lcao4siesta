import math

import numpy as np

from lcao.core.orbital_m import normalize_orbital_m

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover - numba is an optional accelerator
    prange = range

    def njit(*args, **kwargs):
        def decorator(function):
            return function

        return decorator


def _density_inputs(projector):
    """Pack object metadata into arrays suitable for the compiled kernel."""
    io_count = max(projector.io_to_iuo) + 1
    centers = np.empty((io_count, 3), dtype=np.float64)
    unit_orbital = np.empty(io_count, dtype=np.int64)
    shifts = np.empty((io_count, 3), dtype=np.int64)

    basis_size = projector.dm_nb
    radial_size = max(
        len(_pao_basis(projector, iuo)['r'])
        for iuo in range(basis_size)
    )
    radial_r = np.zeros((basis_size, radial_size), dtype=np.float64)
    radial_phi = np.zeros_like(radial_r)
    radial_lengths = np.empty(basis_size, dtype=np.int64)
    angular_l = np.empty(basis_size, dtype=np.int64)
    angular_m = np.empty(basis_size, dtype=np.int64)
    cutoff2 = np.empty(io_count, dtype=np.float64)

    for iuo in range(basis_size):
        l = projector.iuo_to_l[iuo]
        basis = _pao_basis(projector, iuo)
        size = len(basis['r'])
        radial_r[iuo, :size] = basis['r']
        radial_phi[iuo, :size] = basis['phi']
        radial_lengths[iuo] = size
        angular_l[iuo] = l
        angular_m[iuo] = normalize_orbital_m(
            projector.iuo_to_m[iuo],
            l,
            source='ORB_INDX',
            orbital_index=iuo,
            file_path=f'{projector._system}.ORB_INDX',
        )

    for io in range(io_count):
        iuo = projector.io_to_iuo[io]
        centers[io] = projector.io_to_center_io[io]
        unit_orbital[io] = iuo
        shifts[io] = projector.io_to_isc[io]
        cutoff2[io] = _pao_basis(projector, iuo)['cutoff'] ** 2

    dm_columns = np.asarray(projector.dm_listd, dtype=int)
    shift_limit = int(np.max(np.abs(shifts[dm_columns])))
    shift_width = 2 * shift_limit + 1
    dm_by_shift = np.zeros(
        (basis_size, basis_size, shift_width, shift_width, shift_width, projector.dm_ns),
        dtype=np.float64,
    )
    for row in range(basis_size):
        start = projector.dm_listdptr[row]
        stop = start + projector.dm_numd[row]
        for index in range(start, stop):
            column_io = int(projector.dm_listd[index])
            column = unit_orbital[column_io]
            sx, sy, sz = shifts[column_io] + shift_limit
            dm_by_shift[row, column, sx, sy, sz] = projector.dm[index]

    return (
        centers,
        cutoff2,
        unit_orbital,
        shifts,
        radial_r,
        radial_phi,
        radial_lengths,
        angular_l,
        angular_m,
        dm_by_shift,
        shift_limit,
    )


def _pao_basis(projector, iuo):
    """Return the radial table for one unit-cell orbital."""
    symbol = projector.iuo_to_symbol[iuo]
    return projector.ions[symbol][projector.iuo_to_n[iuo]][
        projector.iuo_to_l[iuo]
    ][projector.iuo_to_zeta[iuo]]


@njit(cache=True)
def _interpolate_radial(radius, grid, values, size):
    """Linear interpolation matching ``numpy.interp`` on an ion radial table."""
    if radius <= grid[0]:
        return values[0]
    if radius >= grid[size - 1]:
        return values[size - 1]

    low = 0
    high = size - 1
    while high - low > 1:
        middle = (low + high) // 2
        if grid[middle] <= radius:
            low = middle
        else:
            high = middle

    fraction = (radius - grid[low]) / (grid[high] - grid[low])
    return values[low] + fraction * (values[high] - values[low])


@njit(cache=True)
def _real_solid_harmonic(x, y, z, radius, l, m):
    """Return SIESTA's real ``r**l Y_lm`` convention."""
    if l == 0:
        return 1.0 / math.sqrt(4.0 * math.pi)
    if radius < 1.0e-20:
        return 0.0

    order = abs(m)
    cos_theta = z / radius
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))

    associated = 1.0
    factor = 1.0
    for _ in range(order):
        associated *= -factor * sin_theta
        factor += 2.0

    if l > order:
        previous = associated
        associated = cos_theta * (2 * order + 1) * previous
        for degree in range(order + 2, l + 1):
            older = previous
            previous = associated
            associated = (
                (2 * degree - 1) * cos_theta * previous
                - (degree + order - 1) * older
            ) / (degree - order)

    factorial_ratio = 1.0
    for value in range(l - order + 1, l + order + 1):
        factorial_ratio /= value
    normalization = math.sqrt((2 * l + 1) * factorial_ratio / (4.0 * math.pi))

    angular = normalization * associated
    if order:
        angle = order * math.atan2(y, x)
        angular *= math.sqrt(2.0) * (math.cos(angle) if m > 0 else math.sin(angle))
    return radius ** l * angular


@njit(cache=True, parallel=True)
def _density_on_points(
    positions,
    centers,
    cutoff2,
    unit_orbital,
    shifts,
    radial_r,
    radial_phi,
    radial_lengths,
    angular_l,
    angular_m,
    dm_by_shift,
    shift_limit,
):
    npoint = positions.shape[0]
    nio = centers.shape[0]
    nspin = dm_by_shift.shape[-1]
    density = np.zeros((nspin, npoint), dtype=np.float64)

    for point_index in prange(npoint):
        position = positions[point_index]
        active_io = np.empty(nio, dtype=np.int64)
        active_phi = np.empty(nio, dtype=np.float64)
        active_count = 0

        for io in range(nio):
            dx = position[0] - centers[io, 0]
            dy = position[1] - centers[io, 1]
            dz = position[2] - centers[io, 2]
            radius2 = dx * dx + dy * dy + dz * dz
            if radius2 >= cutoff2[io]:
                continue

            iuo = unit_orbital[io]
            radius = math.sqrt(radius2)
            radial = _interpolate_radial(
                radius,
                radial_r[iuo],
                radial_phi[iuo],
                radial_lengths[iuo],
            )
            solid_harmonic = _real_solid_harmonic(
                dx,
                dy,
                dz,
                radius,
                angular_l[iuo],
                angular_m[iuo],
            )
            active_io[active_count] = io
            active_phi[active_count] = radial * solid_harmonic
            active_count += 1

        for left in range(active_count):
            io1 = active_io[left]
            iuo1 = unit_orbital[io1]
            phi1 = active_phi[left]
            for right in range(left + 1):
                io2 = active_io[right]
                sx = shifts[io2, 0] - shifts[io1, 0]
                sy = shifts[io2, 1] - shifts[io1, 1]
                sz = shifts[io2, 2] - shifts[io1, 2]
                if (
                    abs(sx) > shift_limit
                    or abs(sy) > shift_limit
                    or abs(sz) > shift_limit
                ):
                    continue

                iuo2 = unit_orbital[io2]
                factor = 1.0 if io1 == io2 else 2.0
                product = factor * phi1 * active_phi[right]
                for spin in range(nspin):
                    density[spin, point_index] += (
                        dm_by_shift[
                            iuo1,
                            iuo2,
                            sx + shift_limit,
                            sy + shift_limit,
                            sz + shift_limit,
                            spin,
                        ]
                        * product
                    )

    return density


def electron_density(projector, cell, mesh):
    """Reconstruct the periodic real-space density from a SIESTA DM file."""
    projector.load_context(need_struct_supercell=True, need_orbital_metadata=True)
    positions = projector.unit_cell_positions(cell, mesh).reshape((-1, 3))
    density = _density_on_points(positions, *_density_inputs(projector))
    rho = density.reshape((projector.dm_ns, *map(int, mesh)))
    projector.rho = rho
    return rho
