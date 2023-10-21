from io import BytesIO
from typing import Annotated
from fastapi import FastAPI, File, HTTPException
from enum import Enum
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image


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

class OperatorEnum(str, Enum):
    add = 'add'
    sub = 'sub'
    mul = 'mul'
    div = 'div'

class CalculateRequest(BaseModel):
    operator: OperatorEnum
    a: int
    b: int

class CalculateResponse(BaseModel):
    result: int

@app.post('/math', response_model=CalculateResponse)
def calculate_post(request: CalculateRequest):
    operator = request.operator
    a = request.a
    b = request.b
    match operator:
        case OperatorEnum.add:
            return {'result': a+b}
        case OperatorEnum.sub:
            return {'result': a-b}
        case OperatorEnum.mul:
            return {'result': a*b}
        case OperatorEnum.div:
            return {'result': a/b}
        case _:
            raise HTTPException(status_code=400, detail='Invalid operator')
        
@app.get('/math', response_model=CalculateResponse)
def calculate_get(operator: OperatorEnum, a: str, b: str):
    operator = operator
    a = int(a)
    b = int(b)
    match operator:
        case OperatorEnum.add:
            return {'result': a+b}
        case OperatorEnum.sub:
            return {'result': a-b}
        case OperatorEnum.mul:
            return {'result': a*b}
        case OperatorEnum.div:
            return {'result': a/b}
        case _:
            raise HTTPException(status_code=400, detail='Invalid operator')
        
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