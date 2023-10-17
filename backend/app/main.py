from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from typing import Any, Annotated, List, Optional
from PIL import Image
from io import BytesIO
from enum import Enum
import numpy as np
import pandas as pd
import os

from app.models.cnn_plastic import predict_materials
from app.utils.utils import CNNPlasticRequest

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

# U-Net DDPG
@app.post('/model_ddpg_ice_crystal')
def model_ddpg_ice_crystal(target_ratio: float):
    result = target_ratio + 1
    file_path = "path_to_your_gif_file.gif"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/gif")
    else:
        return {"error": "File not found"}
    
# CNN 
# plastic
@app.post('/model_cnn_plastic')
def model_cnn_plastic(request: CNNPlasticRequest):
    # Extract values from the request model using dict() method
    material_var = list(request.dict().values())[:-1]  # Exclude 'selected_cells' from the list
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
    buffer = BytesIO()
    # save image
    image.save(buffer, "PNG")
    buffer.seek(0)
    # response with image
    return StreamingResponse(buffer, media_type="image/jpeg")


# This is a test of picture turn into grey
@app.post('/ai-art-portrait')
def ai_art_portrait(file: Annotated[bytes, File()]):
    image = Image.open(BytesIO(file))
    image = CycleGAN(image)
    buffer = BytesIO()
    image.save(buffer, 'jpeg')
    buffer.seek(0)
    return StreamingResponse(buffer, media_type='image/jpeg')

def CycleGAN(image: Image):
    image = image.convert('RGB')
    image = image.convert('L')
    return image