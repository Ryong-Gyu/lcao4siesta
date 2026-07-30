# LCAO4SIESTA

LCAO4SIESTA reads SIESTA's localized-orbital files and reconstructs real-space
electron density from a sparse density matrix. It also contains routines for
orbital-projected bands, DOS, and LDOS.

## Install

Python 3.8 or newer is required.

```bash
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e ".[test]"
pytest
```

## Reconstruct electron density

The following files must come from the same SIESTA calculation:

- `SYSTEM.DM` — sparse density matrix
- `SYSTEM.ORB_INDX` — orbital and periodic-image mapping
- `STRUCT.fdf` — lattice vectors and atomic positions
- one `.ion` file for each chemical species

```python
from pathlib import Path

from lcao import LcaoProjector
from lcao.io import read_rho, write_rho

case = Path("my_siesta_run")
cell, mesh, _ = read_rho(case / "SYSTEM.RHO")

projector = LcaoProjector(
    system=str(case / "SYSTEM"),
    dm_file=case / "SYSTEM.DM",
    ion_files={
        "Si": case / "Si.ion",
        # Add every species used by the calculation.
    },
    struct_file=case / "STRUCT.fdf",
)

rho = projector.electron_density(cell, mesh)
write_rho(case / "RECONSTRUCTED.RHO", cell, mesh, rho)
```

`rho` has shape `(nspin, mesh_a, mesh_b, mesh_c)`. Coordinates and lattice
vectors are converted internally to Bohr, matching the radial grid in SIESTA
`.ion` files.

If no reference `.RHO` is available, pass a cell and mesh directly:

```python
import numpy as np

cell = np.array([
    [7.0, 0.0, 0.0],
    [0.0, 7.0, 0.0],
    [0.0, 0.0, 7.0],
])  # Bohr
mesh = np.array([48, 48, 48])
rho = projector.electron_density(cell, mesh)
```

## Validate with the included silicon case

```bash
python test/rho_Si/compare_rho_example.py
```

The example reconstructs `test/rho_Si/OUT.RHO`, compares it with SIESTA's
`Si.RHO`, and checks a maximum absolute error of `3e-6`. The density should
integrate to eight valence electrons for the two-atom silicon cell.

## Legacy API

Existing code using `from lcao4siesta import lcao` remains supported. New code
should use `from lcao import LcaoProjector`.

## Notes

- `DM`, `ORB_INDX`, structure, and ion files must describe the same basis and
  geometry. Mixing runs can produce incorrect densities or metadata errors.
- The reconstruction includes periodic-image remapping from `ORB_INDX`; using
  only the unit-cell DM rows is not sufficient near cell boundaries.
- Numba compiles the density kernel on first use. Later calls reuse its cache
  and are faster.
