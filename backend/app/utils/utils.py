from pydantic import BaseModel
from typing import List

class CNNPlasticRequest(BaseModel):
    angular_of_weaving: float
    width_of_yarn: float
    height_of_yarn: float
    space: float 
    epoxy_E: float 
    epoxy_v: float 
    epoxy_yield_strength_1: float 
    epoxy_plastic_strain_1: float 
    epoxy_yield_strength_2: float 
    epoxy_plastic_strain_2: float 
    fibre_density: float 
    fibre_linear_density: float 
    fibre_E1: float 
    fibre_E2: float 
    fibre_E3: float 
    fibre_G12: float 
    fibre_G23: float 
    fibre_G13: float 
    fibre_v1: float 
    fibre_v2: float 
    fibre_v3: float
    selected_cells: List[int] 

class COMPRequest(BaseModel):
    gamma: float
    selected_cells: List[int]