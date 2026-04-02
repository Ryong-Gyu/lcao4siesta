import math

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
