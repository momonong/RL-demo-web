from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.responses import JSONResponse
import io
from PIL import Image
import numpy as np

app = FastAPI()

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # input_data = np.array(image)

        prediction = testPredict(image)

        return JSONResponse(content={'prediction': prediction.tolist()})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def testPredict(image):
    # 轉換圖像為灰階
    grayscale_image = image.convert("L")
    # 將灰階圖像轉換為 numpy 陣列，並將其轉換為列表以便於 JSON 序列化
    grayscale_array = np.array(grayscale_image)
    return grayscale_array