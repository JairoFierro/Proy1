import traceback
from typing import Optional,List

from fastapi import FastAPI

from joblib import load,dump
from fastapi import HTTPException
from message import Message,TrainingInstance
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
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
   
@app.post("/retrain")
def reentrenamiento (data: List[TrainingInstance]):
    textos_combinados = [m.Titulo + " " + m.Descripcion for m in data]
    etiquetas = [e.Etiqueta for e in data]
    
    vectorizer = TfidfVectorizer()
    X= vectorizer.fit_transform(textos_combinados)
    Y= etiquetas
    
    model = LogisticRegression()
    model.fit(X,Y)
    
    
    y_pred=model.predict(X)
    precision = precision_score(Y, y_pred, average="weighted")
    recall = recall_score(Y, y_pred, average="weighted")
    f1 = f1_score(Y, y_pred, average="weighted")
    dump((vectorizer,model),"model1.joblib")
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "mensaje": "Modelo reentrenado y guardado exitosamente"
    }