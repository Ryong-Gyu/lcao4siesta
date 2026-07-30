from pathlib import Path

import numpy as np

from lcao import LcaoProjector
from lcao.io import read_rho


CASE = Path(__file__).resolve().parent


def test_density_matrix_reproduces_siesta_density():
    cell, mesh, expected = read_rho(CASE / 'Si.RHO')
    projector = LcaoProjector(
        system=str(CASE / 'Si'),
        dm_file=CASE / 'Si.DM',
        ion_files={'Si': CASE / 'Si.ion'},
        struct_file=CASE / 'STRUCT.fdf',
    )

    actual = projector.electron_density(cell, mesh)

    assert actual.shape == expected.shape
    assert np.max(np.abs(actual - expected)) < 3.0e-6

    volume_element = abs(np.linalg.det(cell)) / np.prod(mesh)
    np.testing.assert_allclose(
        actual.sum() * volume_element,
        8.0,
        atol=2.0e-4,
    )
