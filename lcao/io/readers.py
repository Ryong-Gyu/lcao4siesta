import glob

import siesta_io as siesta


def read_dm(path):
    return siesta.readDM(path)


def read_wfsx(path):
    return siesta.readWFSX(path)


def read_hsx(path):
    return siesta.readHSX(path)


def read_struct():
    return siesta.readStruct()


def read_ion(path):
    return siesta.read_ion(path)


def discover_ion_files(pattern='*.ion'):
    return glob.glob(pattern)
