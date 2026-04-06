import glob

import siesta_io as siesta


def write_rho(path, cell, mesh, rho):
    return siesta.writeGrid(path, cell, mesh, rho)
