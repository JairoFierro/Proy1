import traceback
from typing import Optional

from fastapi import FastAPI

from joblib import load
from fastapi import HTTPException
from message import Message

from pydantic import BaseModel
from typing import List

import pandas as pd

app = FastAPI()


@app.get("/")
def read_root():
   return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
   return {"item_id": item_id, "q": q}


@app.post("/predict")
def make_predictions(messages: List[Message]):
    textos_combinados = [m.Titulo + " " + m.Descripcion for m in messages]
    df = pd.DataFrame({"combined_text": textos_combinados})
    print(df)

    model = load("model.joblib")

    X = df["combined_text"]  # Solo la columna que espera el vectorizador
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    response = []
    for pred, prob in zip(predictions, probabilities):
        print(prob)
        response.append({
            "prediction": int(pred),
            "probability": float(max(prob))
        })

    return response
   
    


