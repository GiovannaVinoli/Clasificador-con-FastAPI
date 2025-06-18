from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import pickle

# Cargamos el clasificador 
with open('./data/claims_clf.pkl', 'rb') as archivo:
    clf = pickle.load(archivo)

# Inicializar app
app = FastAPI()

class ReclamosRequest(BaseModel):
    textos: List[str]


# Endpoint de clasificación
@app.post("/clasificar")
def clasificar_reclamos(request: ReclamosRequest):
    try:
        predicciones = clf.clasificar(request.textos)
        return {"resultados": [
            {"reclamo": texto, "clasificado_en": categoria}
            for texto, categoria in zip(request.textos, predicciones)
        ]}
    except Exception as e:
        return {"error": str(e)}

app.mount("/", StaticFiles(directory="static", html=True), name="static")


