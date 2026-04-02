from lcao.compute.projected_band import orbital_projected_bandstructure as _orbital_projected_bandstructure
from lcao.compute.projected_dos import orbital_projected_denstiy_of_state as _orbital_projected_denstiy_of_state
from lcao.compute.projected_ldos import orbital_projected_local_density_of_state as _orbital_projected_local_density_of_state
from lcao.core.model import LcaoProjector
from lcao.selection.orbital_selector import mask_to_pointer as _mask_to_pointer
from lcao.selection.orbital_selector import orbital_mask as _orbital_mask


class lcao(LcaoProjector):
    """Backward-compatible public adapter class."""

    def orbital_mask(self, select):
        return _orbital_mask(self, select)

    def mask_to_pointer(self):
        return _mask_to_pointer(self)

    def orbital_projected_bandstructure(self, select):
        return _orbital_projected_bandstructure(self, select)

    def orbital_projected_denstiy_of_state(self, select, energys):
        return _orbital_projected_denstiy_of_state(self, select, energys)

    def orbital_projected_local_density_of_state(self, select, energys, cell, mesh):
        return _orbital_projected_local_density_of_state(self, select, energys, cell, mesh)
