import numpy as np
from dataclasses import dataclass


def volcel(cell):
    """Return cell volume from lattice vectors."""
    cell_arr = np.asarray(cell, dtype=float)
    return float(abs(np.linalg.det(cell_arr.T)))


def reclat(cell, with_2pi=False):
    """Return reciprocal lattice vectors as columns, matching SIESTA convention."""
    cell_arr = np.asarray(cell, dtype=float)
    recip = np.linalg.inv(cell_arr).T
    if with_2pi:
        recip = recip * (2.0 * np.pi)
    return recip


def modulo(i, n):
    """Fortran-like modulo for mesh indexing."""
    return int(i % n)


def dismin(cmesh, dx):
    """Minimum distance from vector dx to periodic images of one mesh cell."""
    cmesh_arr = np.asarray(cmesh, dtype=float)
    dx_arr = np.asarray(dx, dtype=float)

    best = np.inf
    for i1 in (-1, 0, 1):
        for i2 in (-1, 0, 1):
            for i3 in (-1, 0, 1):
                shifted = dx_arr - (cmesh_arr[:, 0] * i1 + cmesh_arr[:, 1] * i2 + cmesh_arr[:, 2] * i3)
                r = float(np.sqrt(np.dot(shifted, shifted)))
                if r < best:
                    best = r
    return best


@dataclass
class MeshPhiModule:
    DirectPhi: bool = False
    nphi: int = 0
    endpht: np.ndarray = None
    lstpht: np.ndarray = None
    listp2: np.ndarray = None
    phi: np.ndarray = None

    def resetMeshPhi(self):
        self.endpht = None
        self.lstpht = None
        self.listp2 = None
        self.phi = None


def resetMeshPhi(meshphi_module):
    meshphi_module.resetMeshPhi()


__all__ = ['volcel', 'reclat', 'modulo', 'dismin', 'MeshPhiModule', 'resetMeshPhi']
