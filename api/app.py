from fastapi import FastAPI
from predict import predict_next_word
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def get_next_word(data: TextRequest):
    result = predict_next_word(data.text)
    return {"next_word": result}