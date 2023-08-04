from fastapi import FastAPI
from enum import Enum
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from PIL import Image
from io import BytesIO
import numpy as np
from typing import List, Optional

from cnn_plastic import predict_materials

app = FastAPI()

@app.get('/')
async def root():
    return {'message': 'hello'}

class Model1Request(BaseModel):
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
    selected_cells: List[str]

@app.post('/model1')
async def model1(request: Model1Request):
    material_var = [
        request.angular_of_weaving,
        request.width_of_yarn,
        request.height_of_yarn,
        request.space,
        request.epoxy_E,
        request.epoxy_v,
        request.epoxy_yield_strength_1,
        request.epoxy_plastic_strain_1,
        request.epoxy_yield_strength_2,
        request.epoxy_plastic_strain_2,
        request.fibre_density,
        request.fibre_linear_density,
        request.fibre_E1,
        request.fibre_E2,
        request.fibre_E3,
        request.fibre_G12,
        request.fibre_G23,
        request.fibre_G13,
        request.fibre_v1,
        request.fibre_v2,
        request.fibre_v3]
    # 獲取選定的格子資料
    selected_cells = request.selected_cells
    # 將選取的網格索引轉換為整數
    selected_cells = list(map(int, selected_cells))

    # 在這裡進行相應的處理和計算
    img_data = np.ones((1, 25))
    # 將 img_data 中對應索引的值改為 0
    img_data[0, selected_cells] = 0
    img_result = predict_materials(img_data, material_var)

    # read image
    image = Image.open(BytesIO(img_result))
    result = BytesIO()
    # save image
    image.save(result, "jpeg")
    result.seek(0)
    # response with image
    return StreamingResponse(result, media_type="image/jpeg")

async def math():
    return 