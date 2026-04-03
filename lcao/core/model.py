import math
import os
import warnings

import numpy as np
import scipy
from scipy.interpolate import interp1d

from lcao.io import readers


pi = np.pi
smearing = 0.2
overlap_tolerance = 1.0e-10
phi_tolerance = 1.0e-10
INTERNAL_LENGTH_UNIT = 'bohr'


class LcaoProjector:
    def __init__(self, system='Carbon', dm_file=None, ion_files=None, strict_orbital_metadata=True):
        self._system = system
        self._dm_file = dm_file if dm_file is not None else self._system + '.DM'
        self._ion_files = ion_files
        self._strict_orbital_metadata = strict_orbital_metadata
        self._orbital_metadata_validation = {
            'status': 'not_checked',
            'key_type': None,
            'strict': strict_orbital_metadata,
            'matched_count': 0,
            'mismatched_count': 0,
            'first_mismatch': None,
        }
        self._length_unit = INTERNAL_LENGTH_UNIT

        self._target = []
        self._readDM()
        self._readORB_INDX()
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

    def _readORB_INDX(self):
        file_name = self._system + '.ORB_INDX'
        if not os.path.exists(file_name):
            return

        results = readers.read_orb_indx(file_name)
        self.orbital_io = results[0]
        self.atom_index = results[1]
        self.atom_species_index = results[2]
        self.atom_species = results[3]
        self.orbital_iao = results[4]
        self.orbital_n = results[5]
        self.orbital_l = results[6]
        self.orbital_ml = results[7]
        self.orbital_zeta = results[8]
        self.supercell_orbital_io = results[9]
        self.supercell_orbital_iuo = results[10]
        self.species_id_to_label = self._build_species_id_to_label()

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
        self._hsx_atom_index = results[7]
        self._hsx_atom_species = results[8]
        self._hsx_orbital_n = results[9]
        self._hsx_orbital_l = results[10]
        self._hsx_orbital_ml = results[11]
        self._hsx_orbital_zeta = results[12]
        self._validate_hsx_species_metadata()
        self._validate_orbital_metadata_consistency()

    def _build_species_id_to_label(self):
        mapping = {}
        for io, (ia, species_id, species_label) in enumerate(
            zip(self.atom_index, self.atom_species_index, self.atom_species),
            start=1,
        ):
            if species_id in mapping and mapping[species_id] != species_label:
                raise ValueError(
                    'Inconsistent ORB_INDX species metadata at '
                    f'io={io}, ia={ia}: species id={species_id}, '
                    f'label={species_label}, previous label={mapping[species_id]}'
                )
            mapping[species_id] = species_label
        return mapping

    def _validate_hsx_species_metadata(self):
        if not hasattr(self, '_hsx_atom_species'):
            return
        if not hasattr(self, 'atom_species_index'):
            return

        if len(self._hsx_atom_species) != len(self.atom_species_index):
            warnings.warn(
                'ORB_INDX/HSX species metadata length mismatch: '
                f'len(ORB_INDX species id)={len(self.atom_species_index)} '
                f'!= len(HSX species id)={len(self._hsx_atom_species)}',
                stacklevel=2,
            )
            return

        mismatch = np.where(self._hsx_atom_species != self.atom_species_index)[0]
        if len(mismatch) == 0:
            return

        first = int(mismatch[0])
        io = first + 1
        ia = int(self.atom_index[first]) if hasattr(self, 'atom_index') else None
        orb_species_id = int(self.atom_species_index[first])
        hsx_species_id = int(self._hsx_atom_species[first])
        label = self.species_id_to_label.get(orb_species_id, '<unknown>')
        warnings.warn(
            'ORB_INDX/HSX species metadata mismatch at '
            f'io={io}, ia={ia}: ORB_INDX species id/label={orb_species_id}/{label}, '
            f'HSX species id={hsx_species_id}',
            stacklevel=2,
        )

    def _normalize_atom_symbol(self, symbol, *, io=None, ia=None):
        if isinstance(symbol, (int, np.integer)):
            species_id = int(symbol)
            if not hasattr(self, 'species_id_to_label'):
                raise ValueError(
                    'Species id provided but ORB_INDX species mapping is unavailable '
                    f'at io={io}, ia={ia}: species id={species_id}'
                )
            if species_id not in self.species_id_to_label:
                raise ValueError(
                    'Unknown species id in ORB_INDX/HSX metadata at '
                    f'io={io}, ia={ia}: species id={species_id}'
                )
            return self.species_id_to_label[species_id]
        return symbol

    def _validate_orbital_metadata_consistency(self):
        required_orb = (
            'orbital_io',
            'atom_index',
            'orbital_iao',
            'orbital_n',
            'orbital_l',
            'orbital_ml',
            'orbital_zeta',
        )
        required_hsx = (
            '_hsx_atom_index',
            '_hsx_orbital_n',
            '_hsx_orbital_l',
            '_hsx_orbital_ml',
            '_hsx_orbital_zeta',
        )

        if not all(hasattr(self, field) for field in required_orb):
            return
        if not all(hasattr(self, field) for field in required_hsx):
            return

        orb_count = len(self.orbital_io)
        hsx_count = len(self._hsx_atom_index)
        nrows = min(orb_count, hsx_count)

        hsx_iao = np.zeros(hsx_count, dtype=int)
        atom_orbital_counter = {}
        for idx, ia in enumerate(self._hsx_atom_index):
            ia = int(ia)
            atom_orbital_counter[ia] = atom_orbital_counter.get(ia, 0) + 1
            hsx_iao[idx] = atom_orbital_counter[ia]

        orb_full_keys = [
            (
                int(self.orbital_io[i]),
                int(self.atom_index[i]),
                int(self.orbital_n[i]),
                int(self.orbital_l[i]),
                int(self.orbital_ml[i]),
                int(self.orbital_zeta[i]),
            )
            for i in range(nrows)
        ]
        hsx_full_keys = [
            (
                i + 1,
                int(self._hsx_atom_index[i]),
                int(self._hsx_orbital_n[i]),
                int(self._hsx_orbital_l[i]),
                int(self._hsx_orbital_ml[i]),
                int(self._hsx_orbital_zeta[i]),
            )
            for i in range(nrows)
        ]

        orb_min_keys = [
            (int(self.orbital_io[i]), int(self.atom_index[i]), int(self.orbital_iao[i]))
            for i in range(nrows)
        ]
        hsx_min_keys = [(i + 1, int(self._hsx_atom_index[i]), int(hsx_iao[i])) for i in range(nrows)]

        full_mismatch_indices = [i for i in range(nrows) if orb_full_keys[i] != hsx_full_keys[i]]
        if full_mismatch_indices:
            key_type = '(io, ia, iao)'
            mismatch_indices = [i for i in range(nrows) if orb_min_keys[i] != hsx_min_keys[i]]
            compared_orb = orb_min_keys
            compared_hsx = hsx_min_keys
        else:
            key_type = '(io, ia, n, l, m, z)'
            mismatch_indices = full_mismatch_indices
            compared_orb = orb_full_keys
            compared_hsx = hsx_full_keys

        first_mismatch = None
        if mismatch_indices:
            first = mismatch_indices[0]
            first_mismatch = {
                'index0': int(first),
                'io': int(first + 1),
                'orb_indx': compared_orb[first],
                'hsx': compared_hsx[first],
            }
        if orb_count != hsx_count and first_mismatch is None:
            first_mismatch = {
                'index0': int(nrows),
                'io': int(nrows + 1),
                'orb_indx': 'missing' if nrows >= orb_count else compared_orb[nrows],
                'hsx': 'missing' if nrows >= hsx_count else compared_hsx[nrows],
            }

        mismatched_count = len(mismatch_indices) + abs(orb_count - hsx_count)
        matched_count = nrows - len(mismatch_indices)
        self._orbital_metadata_validation = {
            'status': 'match' if mismatched_count == 0 else 'mismatch',
            'key_type': key_type,
            'strict': self._strict_orbital_metadata,
            'matched_count': int(matched_count),
            'mismatched_count': int(mismatched_count),
            'first_mismatch': first_mismatch,
        }

        if mismatched_count == 0:
            return

        mismatch_message = (
            'ORB_INDX/HSX orbital metadata mismatch: '
            f'key={key_type}, matched={matched_count}, mismatched={mismatched_count}, '
            f'first_mismatch={first_mismatch}. '
            'Falling back to ORB_INDX metadata.'
        )
        if self._strict_orbital_metadata:
            raise ValueError(mismatch_message)
        warnings.warn(mismatch_message, stacklevel=2)

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

    def load_context(self, need_wfsx_hsx=False, need_struct_supercell=False, need_orbital_metadata=False):
        if need_wfsx_hsx:
            if not hasattr(self, 'gamma'):
                self._readWFSX()
            if not hasattr(self, 'numh'):
                self._readHSX()
            for field in ('gamma', 'kpoints', 'kweight', 'wavefunction', 'eigenvalue', 'Hamilt', 'Sover', 'xij'):
                if not hasattr(self, field):
                    raise ValueError(f'Missing required field: {field}')
            need_orbital_metadata = True

        if need_struct_supercell:
            if not hasattr(self, 'cell'):
                self._readStruct()
            if not hasattr(self, '_supercell_vector_list'):
                self._build_supercell_vectors()
            for field in ('cell', 'atoms', 'species', '_supercell_vector_list'):
                if not hasattr(self, field):
                    raise ValueError(f'Missing required field: {field}')

        if need_orbital_metadata:
            if not hasattr(self, 'atom_index'):
                self._readORB_INDX()
            required = (
                'atom_index',
                'atom_species',
                'orbital_n',
                'orbital_l',
                'orbital_ml',
                'orbital_zeta',
                'orbital_io',
                'orbital_iao',
            )
            for field in required:
                if not hasattr(self, field):
                    raise ValueError(f'Missing required ORB_INDX field: {field}')
            if len(self.atom_index) != self.dm_nb:
                raise ValueError(
                    f'ORB_INDX orbital count ({len(self.atom_index)}) does not match DM basis size ({self.dm_nb})'
                )

    def delta(self, x):
        if abs(x) > 8 * smearing:
            return 0
        return np.exp(-(x / smearing) ** 2) / (smearing * np.sqrt(pi))

    def length(self, vector):
        square = 0
        for ix in vector:
            square += ix ** 2
        return np.sqrt(square)

    def Rnl(self, symbol, n, l, zeta, r, *, io=None, ia=None):
        if self._length_unit != INTERNAL_LENGTH_UNIT:
            raise ValueError(
                f'Unexpected internal length unit state: {self._length_unit}. '
                f'Expected {INTERNAL_LENGTH_UNIT}.'
            )
        atom_symbol = self._normalize_atom_symbol(symbol, io=io, ia=ia)
        if atom_symbol not in self.ions:
            raise KeyError(
                'Missing ion data for orbital metadata at '
                f'io={io}, ia={ia}: species id/label={symbol}/{atom_symbol}'
            )
        pao_basis = self.ions[atom_symbol][n][l][zeta]
        pao_unit = pao_basis.get('length_unit')
        if pao_unit is None:
            raise ValueError(
                'Missing PAO length unit metadata for orbital table at '
                f'io={io}, ia={ia}, symbol={atom_symbol}, (n,l,zeta)=({n},{l},{zeta}).'
            )
        if str(pao_unit).lower() != INTERNAL_LENGTH_UNIT:
            raise ValueError(
                'Length-unit mismatch between projector and PAO table: '
                f'projector={INTERNAL_LENGTH_UNIT}, pao={pao_unit}, '
                f'io={io}, ia={ia}, symbol={atom_symbol}, (n,l,zeta)=({n},{l},{zeta}), r={r}.'
            )
        cutoff_radius = pao_basis['cutoff']
        if r < cutoff_radius:
            f_phi = interp1d(pao_basis['r'], pao_basis['phi'])
            return f_phi(r)
        return 0

    def Yml(self, vector, m, l):
        x, y, z = vector
        r2 = vector ** 2
        r = np.sqrt(sum(r2))
        if r < 1.0e-20:
            # Match SIESTA's tiny-radius safeguard in all_phi/rlylm path.
            # At r≈0, angular coordinates are undefined; choosing (theta,phi)=0
            # avoids NaNs while radial factors control the physical limit.
            phi = 0.0
            theta = 0.0
            return scipy.special.sph_harm(m, l, theta, phi)

        phi = np.arccos(z / r)
        theta = np.arctan2(y, x) + pi
        return scipy.special.sph_harm(m, l, theta, phi)

    def unit_cell_grid(self, cell, mesh):
        cell = np.array(cell)
        na, nb, nc = int(mesh[0]), int(mesh[1]), int(mesh[2])

        # SIESTA real-space mesh is periodic on [0, a), [0, b), [0, c).
        # Therefore grid increments are cell-vector / mesh-size.
        ua = cell[0] / na
        ub = cell[1] / nb
        uc = cell[2] / nc

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
