import glob

import siesta_io as siesta


def write_rho(path):
    return siesta.writeGrid(path, cell, mesh, rho)
