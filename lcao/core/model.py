import math

import numpy as np
import scipy
from scipy.interpolate import interp1d

from lcao.io import readers


pi = np.pi
smearing = 0.2
overlap_tolerance = 1.0e-10
phi_tolerance = 1.0e-10


class LcaoProjector:
    def __init__(self, system='Carbon', dm_file=None, ion_files=None):
        self._system = system
        self._dm_file = dm_file if dm_file is not None else self._system + '.DM'
        self._ion_files = ion_files

        self._target = []
        self._readDM()
        self._readIon()

    def _readDM(self):
        results = readers.read_dm(self._dm_file)
        self.dm_nb = results[0]
        self.dm_ns = results[1]
        self.dm_numd = results[2]
        self.dm_listdptr = results[3]
        self.dm_listd = results[4]
        self.dm = results[5]

    def _readWFSX(self):
        file_name = self._system + '.WFSX'
        results = readers.read_wfsx(file_name)
        self.gamma = results[0]
        self.kpoints = results[1]
        self.kweight = results[2]
        self.wavefunction = results[3]
        self.eigenvalue = results[4]
        self._atom_index = results[5]
        self._atom_symbol = results[6]
        self._orbital_index = results[7]
        self._orbital_n = results[8]
        self._orbital_symbol = results[9]

    def _readHSX(self):
        file_name = self._system + '.HSX'
        results = readers.read_hsx(file_name)
        self.numh = results[0]
        self.listhptr = results[1]
        self.listh = results[2]
        self.indxuo = results[3]
        self.Hamilt = results[4]
        self.Sover = results[5]
        self.xij = results[6]
        self.atom_index = results[7]
        self.atom_species = results[8]
        self.orbital_n = results[9]
        self.orbital_l = results[10]
        self.orbital_ml = results[11]
        self.orbital_zeta = results[12]

    def _readIon(self):
        self.ions = {}
        if self._ion_files is not None:
            for symbol, file_name in self._ion_files.items():
                self.ions.update({symbol: readers.read_ion(file_name)})
            return

        for file_name in readers.discover_ion_files():
            symbol = file_name.replace('.ion', '')
            self.ions.update({symbol: readers.read_ion(file_name)})

    def _readStruct(self):
        results = readers.read_struct()
        self.cell = results[0]
        self.atoms = results[1]
        self.species = results[2]

    def _build_supercell_vectors(self):
        max_cutoff = 0
        for species in self.ions:
            for qn in self.ions[species]:
                for ql in self.ions[species][qn]:
                    for qlm in self.ions[species][qn][ql]:
                        cutoff = self.ions[species][qn][ql][qlm]['cutoff']
                        if cutoff >= max_cutoff:
                            max_cutoff = cutoff

        length_a = self.length(self.cell[0])
        length_b = self.length(self.cell[1])
        length_c = self.length(self.cell[2])

        na = int(math.ceil(max_cutoff / length_a))
        nb = int(math.ceil(max_cutoff / length_b))
        nc = int(math.ceil(max_cutoff / length_c))

        nsc = (2 * na + 1) * (2 * nb + 1) * (2 * nc + 1)
        vectors = np.zeros((nsc, 3), dtype=float)

        index = 0
        for ia in range(2 * na + 1):
            for ib in range(2 * nb + 1):
                for ic in range(2 * nc + 1):
                    vectors[index] += (ia - na) * self.cell[0]
                    vectors[index] += (ib - nb) * self.cell[1]
                    vectors[index] += (ic - nc) * self.cell[2]
                    index += 1

        self._supercell_vector_list = vectors

    def load_context(self, need_wfsx_hsx=False, need_struct_supercell=False):
        if need_wfsx_hsx:
            if not hasattr(self, 'gamma'):
                self._readWFSX()
            if not hasattr(self, 'numh'):
                self._readHSX()
            for field in ('gamma', 'kpoints', 'kweight', 'wavefunction', 'eigenvalue', 'Hamilt', 'Sover', 'xij'):
                if not hasattr(self, field):
                    raise ValueError(f'Missing required field: {field}')

        if need_struct_supercell:
            if not hasattr(self, 'cell'):
                self._readStruct()
            if not hasattr(self, '_supercell_vector_list'):
                self._build_supercell_vectors()
            for field in ('cell', 'atoms', 'species', '_supercell_vector_list'):
                if not hasattr(self, field):
                    raise ValueError(f'Missing required field: {field}')

    def delta(self, x):
        if abs(x) > 8 * smearing:
            return 0
        return np.exp(-(x / smearing) ** 2) / (smearing * np.sqrt(pi))

    def length(self, vector):
        square = 0
        for ix in vector:
            square += ix ** 2
        return np.sqrt(square)

    def Rnl(self, symbol, n, l, zeta, r):
        pao_basis = self.ions[symbol][n][l][zeta]
        cutoff_radius = pao_basis['cutoff']
        if r < cutoff_radius:
            f_phi = interp1d(pao_basis['r'], pao_basis['phi'])
            return f_phi(r)
        return 0

    def Yml(self, vector, m, l):
        x, y, z = vector
        r2 = vector ** 2
        r = np.sqrt(sum(r2))
        phi = np.arccos(z / r)
        theta = np.arctan2(y, x) + pi
        return scipy.special.sph_harm(m, l, theta, phi)

    def unit_cell_grid(self, cell, mesh):
        cell = np.array(cell)
        na, nb, nc = int(mesh[0]), int(mesh[1]), int(mesh[2])

        ua = cell[0] if na == 1 else cell[0] / (na - 1)
        ub = cell[1] if nb == 1 else cell[1] / (nb - 1)
        uc = cell[2] if nc == 1 else cell[2] / (nc - 1)

        xgrid = np.zeros((1, na, nb, nc), dtype=float)
        ygrid = np.zeros((1, na, nb, nc), dtype=float)
        zgrid = np.zeros((1, na, nb, nc), dtype=float)

        for ia in range(na):
            for ib in range(nb):
                for ic in range(nc):
                    position_vector = ia * ua + ib * ub + ic * uc
                    xgrid[0, ia, ib, ic] = position_vector[0]
                    ygrid[0, ia, ib, ic] = position_vector[1]
                    zgrid[0, ia, ib, ic] = position_vector[2]
        return xgrid, ygrid, zgrid


__all__ = ['LcaoProjector', 'overlap_tolerance', 'phi_tolerance']
