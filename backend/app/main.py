from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Annotated, List, Optional
from PIL import Image
from io import BytesIO
from enum import Enum
import numpy as np
import pandas as pd
import imageio
import os

from app.models.cnn_plastic import predict_materials, clear_plt
from app.utils.utils import CNNPlasticRequest
from app.models.ddpg_ice import generate_gif

app = FastAPI()

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Allow this origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

# U-Net 
# DDPG
@app.post('/model_ddpg_ice_crystal')
def model_ddpg_ice_crystal(request_ratio: float = np.random.uniform(0.3,0.7)): # 0.3~0.7
    # gif = generate_gif(request_ratio)
    # return {"test": f"this is a test of input {gif}"}
    gif_images = generate_gif(request_ratio)
    buffer = BytesIO()
    imageio.mimsave(buffer, gif_images, format='GIF', fps=10)
    buffer.seek(0)

    # 使用 StreamingResponse 回傳 GIF 檔案
    return StreamingResponse(buffer, media_type="image/gif")

# CNN 
# plastic
@app.post('/model_smart_rve')
def model_smart_rve(request: CNNPlasticRequest):
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

@app.post('/clear_plot')
def clear_plot():
    clear_plt()
    return {"message": "Plot cleared successfully"}


# COMP
# Composites design
@app.post('/model_composites_disgn')
def model_composites_disgn():
        return {"error": "File not found"}

# HRRL
# comp2field
@app.post('/model_composites_disgn')
def model_composites_disgn():
        return {"error": "File not found"}

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