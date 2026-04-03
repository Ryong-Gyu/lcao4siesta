import math
import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - optional dependency
    def njit(*args, **kwargs):
        def _decorator(func):
            return func

        return _decorator


@njit(cache=True)
def _compute_projection_components(gamma, wf0_io1, wf0_io2, wf1_io1, wf1_io2):
    if gamma == 1:
        qcos = wf0_io1 * wf0_io2
        qsin = 0.0
    else:
        qcos = wf0_io1 * wf0_io2 + wf1_io1 * wf1_io2
        qsin = wf0_io1 * wf1_io2 - wf1_io1 * wf0_io2
    return qcos, qsin


@njit(cache=True)
def _accumulate_overlap_phase(sover, phase, qcos, qsin):
    real_weight = sover * (qcos * math.cos(phase) - qsin * math.sin(phase))
    imag_weight = sover * (qcos * math.sin(phase) + qsin * math.sin(phase))
    return real_weight, imag_weight


def compute_projection_factor(gamma, wf, io1, io2, iw, isp, ik):
    """Compute projection components for one orbital pair.

    For identical inputs, this helper is deterministic and returns values in the
    same order `(qcos, qsin)`.
    """
    if gamma == 1:
        wf0_io1 = wf[0][io1]
        wf0_io2 = wf[0][io2]
        wf1_io1 = 0.0
        wf1_io2 = 0.0
    else:
        wf0_io1 = wf[0][io1][iw][isp][ik]
        wf0_io2 = wf[0][io2][iw][isp][ik]
        wf1_io1 = wf[1][io1][iw][isp][ik]
        wf1_io2 = wf[1][io2][iw][isp][ik]
    return _compute_projection_components(gamma, wf0_io1, wf0_io2, wf1_io1, wf1_io2)


def accumulate_overlap_weight(Sover, xij, kpt, ind, qcos, qsin):
    """Apply overlap and phase to projection components.

    For identical inputs, this helper is deterministic and returns values in the
    same order `(real_weight, imag_weight)`.
    """
    phase = (kpt * xij[ind]).sum()
    return _accumulate_overlap_phase(Sover[ind], phase, qcos, qsin)


def sum_projection_real_weights(gamma, wf, kpt_vector, list_target_io, list_io, list_ptr, Sover, xij, iw, isp, ik):
    """Return projection-buffer sum for one target on one (k, spin, wavefunction) state.

    Note: this intentionally preserves legacy write/overwrite behavior where
    the per-projection buffer is overwritten for each `io1` and only the last
    `io1` values contribute to the returned sum.
    """
    projection_buffer = np.zeros((len(list_io)), dtype=float)
    for io1 in list_target_io:
        for iio, io2 in enumerate(list_io):
            ind = list_ptr[iio]
            qcos, qsin = compute_projection_factor(gamma, wf, io1, io2, iw, isp, ik)
            real_weight, _ = accumulate_overlap_weight(Sover, xij, kpt_vector, ind, qcos, qsin)
            projection_buffer[iio] = real_weight
    return projection_buffer.sum()


@njit(cache=True)
def build_mesh_positions(xgrid0, ygrid0, zgrid0):
    """Flatten `(na, nb, nc)` mesh arrays into `(npoint, 3)` positions."""
    na, nb, nc = xgrid0.shape
    npoint = na * nb * nc
    positions = np.zeros((npoint, 3), dtype=np.float64)
    grid_indices = np.zeros((npoint, 3), dtype=np.int64)

    ip = 0
    for ix in range(na):
        for iy in range(nb):
            for iz in range(nc):
                positions[ip, 0] = xgrid0[ix, iy, iz]
                positions[ip, 1] = ygrid0[ix, iy, iz]
                positions[ip, 2] = zgrid0[ix, iy, iz]
                grid_indices[ip, 0] = ix
                grid_indices[ip, 1] = iy
                grid_indices[ip, 2] = iz
                ip += 1
    return positions, grid_indices
