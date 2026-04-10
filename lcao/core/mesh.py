from dataclasses import dataclass

import numpy as np

from lcao.core.mesh_util import dismin, modulo, reclat, volcel


@dataclass
class MeshModule:
    idop: np.ndarray = None
    ipa: np.ndarray = None
    dxa: np.ndarray = None
    xdop: np.ndarray = None
    xdsp: np.ndarray = None
    mop: int = 0
    ne: np.ndarray = None
    nem: np.ndarray = None
    nmsc: np.ndarray = None
    nmuc: np.ndarray = None
    nusc: np.ndarray = None
    meshLim: np.ndarray = None
    nmeshg: np.ndarray = None
    nsm: int = 1
    nsp: int = 1
    cmesh: np.ndarray = None
    rcmesh: np.ndarray = None
    indexp: np.ndarray = None
    iatfold: np.ndarray = None


def init_mesh(cell, mesh, nsc, rmax, nsm=1, meshLim=None):
    """InitMesh equivalent for Python (MPI-free path).

    Parameters follow SIESTA naming where possible:
    - cell: unit-cell vectors
    - mesh: ntm(3), mesh intervals incl. subpoints
    - nsc: number of unit cells in supercell direction
    - rmax: maximum orbital radius
    """
    mesh_m = MeshModule()
    mesh_m.nsm = int(nsm)
    mesh_m.nsp = int(mesh_m.nsm**3)

    ntm = np.asarray(mesh, dtype=int)
    nm = ntm // mesh_m.nsm

    mesh_m.nmeshg = ntm.astype(int)
    mesh_m.nmsc = (nm * np.asarray(nsc, dtype=int)).astype(int)
    mesh_m.nmuc = nm.astype(int)
    mesh_m.nusc = np.asarray(nsc, dtype=int)

    mesh_m.cmesh = np.zeros((3, 3), dtype=float)
    cell_arr = np.asarray(cell, dtype=float)
    for i in range(3):
        mesh_m.cmesh[:, i] = cell_arr[:, i] / float(nm[i])
    mesh_m.rcmesh = reclat(mesh_m.cmesh, with_2pi=False)

    mesh_m.ne = np.zeros((3,), dtype=int)
    for i in range(3):
        pldist = 1.0 / np.sqrt(np.dot(mesh_m.rcmesh[:, i], mesh_m.rcmesh[:, i]))
        mesh_m.ne[i] = int(rmax / pldist)
    mesh_m.ne[:] = mesh_m.ne[:] + 1

    mesh_m.xdsp = np.zeros((3, mesh_m.nsp), dtype=float)
    isp = 0
    for i3 in range(mesh_m.nsm):
        for i2 in range(mesh_m.nsm):
            for i1 in range(mesh_m.nsm):
                mesh_m.xdsp[:, isp] = (
                    mesh_m.cmesh[:, 0] * i1 + mesh_m.cmesh[:, 1] * i2 + mesh_m.cmesh[:, 2] * i3
                ) / float(mesh_m.nsm)
                isp += 1

    mop = 0
    for i3 in range(-mesh_m.ne[2], mesh_m.ne[2] + 1):
        for i2 in range(-mesh_m.ne[1], mesh_m.ne[1] + 1):
            for i1 in range(-mesh_m.ne[0], mesh_m.ne[0] + 1):
                dxp = mesh_m.cmesh[:, 0] * i1 + mesh_m.cmesh[:, 1] * i2 + mesh_m.cmesh[:, 2] * i3
                within = False
                for isp in range(mesh_m.nsp):
                    dx = dxp + mesh_m.xdsp[:, isp]
                    if dismin(mesh_m.cmesh, dx) < rmax:
                        within = True
                        break
                if within:
                    mop += 1
    mesh_m.mop = int(mop)

    if meshLim is None:
        mesh_m.meshLim = np.array([[1, 1, 1], [mesh_m.nmsc[0], mesh_m.nmsc[1], mesh_m.nmsc[2]]], dtype=int)
    else:
        mesh_m.meshLim = np.asarray(meshLim, dtype=int)

    setup_ext_mesh(mesh_m, rmax)
    dvol = float(volcel(cell_arr) / np.prod(ntm))

    return {'mesh_module': mesh_m, 'nm': nm, 'ntm': ntm, 'dvol': dvol}


def init_atom_mesh(mesh_m, xa):
    """InitAtomMesh equivalent: fills ipa, dxa, iatfold for given atoms."""
    xa_arr = np.asarray(xa, dtype=float)
    if xa_arr.shape[0] != 3:
        xa_arr = xa_arr.T
    na = xa_arr.shape[1]

    mesh_m.ipa = np.zeros((na,), dtype=int)
    mesh_m.dxa = np.zeros((3, na), dtype=float)
    mesh_m.iatfold = np.zeros((3, na), dtype=int)

    myBox = mesh_m.meshLim - 1
    myExtBox = np.zeros((2, 3), dtype=int)
    myExtBox[0, :] = myBox[0, :] - 2 * mesh_m.ne[:]
    myExtBox[1, :] = myBox[1, :] + 2 * mesh_m.ne[:]
    nem = myExtBox[1, :] - myExtBox[0, :] + 1

    for ia in range(na):
        cxa = xa_arr[:, ia] @ mesh_m.rcmesh
        ixabeffold = np.floor(cxa).astype(int)
        cxa = np.mod(cxa, mesh_m.nmsc.astype(float))
        ixa = np.floor(cxa).astype(int)
        mesh_m.iatfold[:, ia] = (ixa - ixabeffold) // mesh_m.nmsc

        cxa = cxa - ixa
        mesh_m.dxa[:, ia] = mesh_m.cmesh @ cxa

        assigned = False
        for j3 in (-1, 0, 1):
            for j2 in (-1, 0, 1):
                for j1 in (-1, 0, 1):
                    jsc = np.array([j1, j2, j3], dtype=int)
                    jxa = ixa + jsc * mesh_m.nmsc
                    if np.all(jxa >= (myBox[0, :] - mesh_m.ne)) and np.all(jxa <= (myBox[1, :] + mesh_m.ne)):
                        jxa = jxa - myExtBox[0, :]
                        mesh_m.ipa[ia] = int(1 + jxa[0] + nem[0] * jxa[1] + nem[0] * nem[1] * jxa[2])
                        mesh_m.iatfold[:, ia] = mesh_m.iatfold[:, ia] + jsc
                        assigned = True
                        break
                if assigned:
                    break
            if assigned:
                break

    return mesh_m


def setup_ext_mesh(mesh_m, rmax):
    """setupExtMesh equivalent: fills indexp, idop, xdop."""
    myBox = mesh_m.meshLim - 1
    myExtBox = np.zeros((2, 3), dtype=int)
    myExtBox[0, :] = myBox[0, :] - 2 * mesh_m.ne[:]
    myExtBox[1, :] = myBox[1, :] + 2 * mesh_m.ne[:]

    mesh_m.nem = myExtBox[1, :] - myExtBox[0, :] + 1
    nep = int(mesh_m.nem[0] * mesh_m.nem[1] * mesh_m.nem[2])

    mesh_m.indexp = np.zeros((nep,), dtype=int)
    mesh_m.idop = np.zeros((mesh_m.mop,), dtype=int)
    mesh_m.xdop = np.zeros((3, mesh_m.mop), dtype=float)

    boxWidth = myBox[1, :] - myBox[0, :] + 1
    extWidth = myExtBox[1, :] - myExtBox[0, :] + 1

    for i3 in range(myExtBox[0, 2], myExtBox[1, 2] + 1):
        for i2 in range(myExtBox[0, 1], myExtBox[1, 1] + 1):
            for i1 in range(myExtBox[0, 0], myExtBox[1, 0] + 1):
                j1 = modulo(i1, mesh_m.nmsc[0])
                j2 = modulo(i2, mesh_m.nmsc[1])
                j3 = modulo(i3, mesh_m.nmsc[2])

                j1r = j1 - myBox[0, 0]
                j2r = j2 - myBox[0, 1]
                j3r = j3 - myBox[0, 2]
                k1 = i1 - myExtBox[0, 0]
                k2 = i2 - myExtBox[0, 1]
                k3 = i3 - myExtBox[0, 2]

                k = 1 + k1 + extWidth[0] * k2 + extWidth[0] * extWidth[1] * k3
                if 0 <= j1r < boxWidth[0] and 0 <= j2r < boxWidth[1] and 0 <= j3r < boxWidth[2]:
                    j = 1 + j1r + boxWidth[0] * j2r + boxWidth[0] * boxWidth[1] * j3r
                    mesh_m.indexp[k - 1] = int(j)
                else:
                    mesh_m.indexp[k - 1] = 0

    mop = 0
    for i3 in range(-mesh_m.ne[2], mesh_m.ne[2] + 1):
        for i2 in range(-mesh_m.ne[1], mesh_m.ne[1] + 1):
            for i1 in range(-mesh_m.ne[0], mesh_m.ne[0] + 1):
                dxp = mesh_m.cmesh[:, 0] * i1 + mesh_m.cmesh[:, 1] * i2 + mesh_m.cmesh[:, 2] * i3
                within = False
                for isp in range(mesh_m.nsp):
                    dx = dxp + mesh_m.xdsp[:, isp]
                    if dismin(mesh_m.cmesh, dx) < rmax:
                        within = True
                        break
                if within:
                    mesh_m.idop[mop] = int(i1 + mesh_m.nem[0] * i2 + mesh_m.nem[0] * mesh_m.nem[1] * i3)
                    mesh_m.xdop[:, mop] = dxp
                    mop += 1

    mesh_m.mop = int(mop)
    if mop < mesh_m.idop.shape[0]:
        mesh_m.idop = mesh_m.idop[:mop]
        mesh_m.xdop = mesh_m.xdop[:, :mop]

    return mesh_m


# Fortran-style aliases
InitMesh = init_mesh
InitAtomMesh = init_atom_mesh
setupExtMesh = setup_ext_mesh

__all__ = ['MeshModule', 'init_mesh', 'init_atom_mesh', 'setup_ext_mesh', 'InitMesh', 'InitAtomMesh', 'setupExtMesh']
