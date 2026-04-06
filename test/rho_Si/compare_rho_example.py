import numpy as np

from lcao.io import read_rho, write_rho
from lcao4siesta import lcao

io_cell, io_mesh, io_rho = read_rho("Si.RHO")

model = lcao(
    system="Si",
    dm_file="Si.DM",
    ion_files={"Si": "Si.ion"},
)
gen_rho = model.electron_density(io_cell, io_mesh)
write_rho("OUT.RHO", io_cell, io_mesh, gen_rho)

print("mesh", tuple(io_mesh), "shape", io_rho.shape)
print("max|Δρ|", float(np.max(np.abs(gen_rho[0] - io_rho[0]))))
