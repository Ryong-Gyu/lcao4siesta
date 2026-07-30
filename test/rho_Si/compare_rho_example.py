from pathlib import Path

import numpy as np

from lcao.io import read_rho, write_rho
from lcao import LcaoProjector


def main():
    case = Path(__file__).resolve().parent
    cell, mesh, reference = read_rho(case / 'Si.RHO')
    model = LcaoProjector(
        system=str(case / 'Si'),
        dm_file=case / 'Si.DM',
        ion_files={'Si': case / 'Si.ion'},
        struct_file=case / 'STRUCT.fdf',
    )
    generated = model.electron_density(cell, mesh)
    write_rho(case / 'OUT.RHO', cell, mesh, generated)

    max_error = float(np.max(np.abs(generated - reference)))
    volume_element = abs(np.linalg.det(cell)) / np.prod(mesh)
    electron_count = float(generated.sum() * volume_element)
    print('mesh:', tuple(mesh))
    print('max |generated - SIESTA|:', max_error)
    print('integrated electrons:', electron_count)

    if max_error > 3.0e-6:
        raise RuntimeError('Reconstructed density exceeds the 3e-6 validation tolerance')


if __name__ == '__main__':
    main()
