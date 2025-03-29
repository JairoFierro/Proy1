from datetime import datetime  
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
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import accuracy_score, classification_report

from joblib import load
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from scipy.sparse import vstack
import numpy as np
from joblib import dump, load


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # O reemplaza "*" por ["http://127.0.0.1:5500"] si sirves el HTML desde ahí
    allow_credentials=True,
    allow_methods=["*"],  # Muy importante para permitir OPTIONS, POST, etc.
    allow_headers=["*"],
)


@app.get("/")
def read_root():
   return {"Hello": "World"}

@app.get("/web")
def get_web():
    return FileResponse("static/index.html")

@app.get('/reentrenar')
def reentrenar():
    return FileResponse("static/reentrenar.html")

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
   return {"item_id": item_id, "q": q}


@app.post("/predict")
def make_predictions(messages: List[Message]):
    vectorizer,model=load('model.joblib') 
    textos_combinados = [m.Titulo + " " + m.Descripcion for m in messages]
    X=vectorizer.transform(textos_combinados)

    # df = pd.DataFrame({"combined_text": textos_combinados})
    # print(df)

    # model = load("model.joblib")

    # X = df["combined_text"]  # Solo la columna que espera el vectorizador
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    # response = []
    # for pred, prob in zip(predictions, probabilities):
    #     print(prob)
    #     response.append({
    #         "prediction": int(pred),
    #         "probability": float(max(prob))
    #     })

    # return response

    return [{
        "prediction": int(pred),
        "probability": float(max(prob)),
        "probabilities": {  # Opcional: mostrar todas las probabilidades
            str(i): float(prob[i]) for i in range(len(prob))
        }
    } for pred, prob in zip(predictions, probabilities)]
    

def convert_numpy_types(obj):
    """Convierte tipos de NumPy a tipos nativos de Python para serialización JSON"""
    if isinstance(obj, (np.integer, np.floating)):
        return int(obj) if isinstance(obj, np.integer) else float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    return obj


@app.post("/retrain")
async def reentrenamiento(data: List[TrainingInstance]):
    try:
        try:
            vectorizer, old_model = load("model.joblib")
            X_old, y_old = load("historical_data.joblib")
            y_old = np.array(y_old)  # Asegurar que es numpy array
        except FileNotFoundError:
            vectorizer = TfidfVectorizer()
            X_old, y_old = None, None

        nuevos_textos = [f"{d.Titulo} {d.Descripcion}" for d in data]
        nuevas_etiquetas = np.array([d.Etiqueta for d in data])

        if X_old is None:
            X_new = vectorizer.fit_transform(nuevos_textos)
        else:
            X_new = vectorizer.transform(nuevos_textos)

        if X_old is not None:
            X_combined = vstack([X_old, X_new])
            y_combined = np.concatenate([y_old, nuevas_etiquetas])
        else:
            X_combined, y_combined = X_new, nuevas_etiquetas

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_combined, y_combined)

        y_pred = model.predict(X_combined)
        accuracy = float(accuracy_score(y_combined, y_pred)) 
        report = classification_report(y_combined, y_pred, output_dict=True)
        
        # Convertir valores NumPy en el reporte
        report = convert_numpy_types(report)

        unique_classes, counts = np.unique(y_combined, return_counts=True)
        class_distribution = {
            str(cls): int(count)  # Convertir a int
            for cls, count in zip(unique_classes, counts)
        }

        dump((vectorizer, model), "model.joblib")
        dump((X_combined, y_combined), "historical_data.joblib")

        return {
            "status": "success",
            "samples": {
                "total": int(X_combined.shape[0]), 
                "new_added": len(data),
                "class_distribution": class_distribution
            },
            "metrics": {
                "accuracy": accuracy,
                "precision": float(report['weighted avg']['precision']),
                "recall": float(report['weighted avg']['recall']),
                "f1_score": float(report['weighted avg']['f1-score'])
            },
            "details": {
                "model_type": "RandomForest",
                "features": int(X_combined.shape[1]),  
                "last_trained": datetime.now().isoformat()
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
