from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Annotated, List, Optional
from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt
import numpy as np
import imageio

from app.utils.utils import COMPRequest
from app.utils.utils import CNNPlasticRequest
from app.models.cnn_plastic import predict_materials, clear_plt
from app.models.ddpg_ice import generate_gif
from app.models.code.main import comp_in

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
@app.post('/model_comp')
def model_comp(request: COMPRequest):
    cells = request.selected_cells
    gamma = request.gamma

    # 定義映射對照表
    index_mapping = {0: 0, 1: 1, 4: 2, 5: 3, 8: 4, 9: 5, 12: 6, 13: 7}

    # 創建一個8個元素的全零陣列
    state_0 = np.zeros(8, dtype=int)

    # 對於被選中的網格，使用映射對照表進行轉換，並將其值設為1
    for idx in cells:
        mapped_idx = index_mapping.get(idx)
        if mapped_idx is not None:
            state_0[mapped_idx] = 1

    img_result = comp_in(state_0, gamma)

    # img_result = get_test_image()
    # read image
    try:
        image = Image.open(BytesIO(img_result))
        buffer = BytesIO()
        # save image as PNG
        image.save(buffer, "PNG")
        buffer.seek(0)
        # response with image as PNG
        return StreamingResponse(buffer, media_type="image/png")
    except Exception as e:
        return {"error": str(e)}
    
def get_test_image():
    # 這只是一個示例，您可以根據需要生成任何圖片
    fig, ax = plt.subplots(figsize=(6, 6))  # 創建一個6x6英寸的正方形圖片
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y)
    
    # 設置軸的限制以確保它們是相等的，這樣圖片就會是正方形的
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 1)
    
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close(fig)  # 關閉圖片以釋放資源
    buffer.seek(0)
    return buffer.getvalue()

# HRRL
# comp2field
@app.post('/model_comp2field')
def model_comp2field(file: Annotated[bytes, File()]):
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