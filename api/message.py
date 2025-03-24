from pydantic import BaseModel
from typing import Optional

class Message(BaseModel):
    Titulo: str
    Descripcion: str 

class TrainingInstance(BaseModel):
    Titulo: str
    Descripcion: str 
    Etiqueta: int 
