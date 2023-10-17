from pydantic import BaseModel
from typing import List

class CNNPlasticRequest(BaseModel):
    angular_of_weaving: float = 90
    width_of_yarn: float = 0.9
    height_of_yarn: float = 0.3
    space: float = 1.8
    epoxy_E: float = 20000
    epoxy_v: float = 0.4
    epoxy_yield_strength_1: float = 3
    epoxy_plastic_strain_1: float = 0
    epoxy_yield_strength_2: float = 600
    epoxy_plastic_strain_2: float = 0.3
    fibre_density: float = 2550
    fibre_linear_density: float = 0.00056
    fibre_E1: float = 72000
    fibre_E2: float = 72000
    fibre_E3: float = 72000
    fibre_G12: float = 30000
    fibre_G23: float = 30000
    fibre_G13: float = 30000
    fibre_v1: float = 0.2
    fibre_v2: float = 0.2
    fibre_v3: float = 0.2
    selected_cells: List[int] = []