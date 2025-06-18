# Importamos FastAPI para crear la API web
from fastapi import FastAPI

# Importamos StaticFiles para servir archivos HTML desde una carpeta
from fastapi.staticfiles import StaticFiles

# Importamos BaseModel de Pydantic para definir el esquema de los datos de entrada
from pydantic import BaseModel

# Importamos List para indicar que recibiremos una lista de textos
from typing import List

# Importamos pickle para cargar el clasificador entrenado previamente
import pickle

# -------------------------------------------------------------------
# Cargar el modelo ya entrenado desde el archivo .pkl
# -------------------------------------------------------------------
with open('./data/claims_clf.pkl', 'rb') as archivo:
    clf = pickle.load(archivo)

# -------------------------------------------------------------------
# Inicializamos la aplicación de FastAPI
# -------------------------------------------------------------------
app = FastAPI()

# -------------------------------------------------------------------
# Definimos el formato esperado del cuerpo del POST (una lista de textos)
# -------------------------------------------------------------------
class ReclamosRequest(BaseModel):
    """
    Modelo de entrada para el endpoint /clasificar.

    Atributos:
        textos (List[str]): Lista de textos de reclamos a clasificar.
    """
    
    textos: List[str]

# -------------------------------------------------------------------
# Endpoint POST: /clasificar
# Recibe una lista de reclamos y devuelve las categorías asignadas
# -------------------------------------------------------------------
@app.post("/clasificar")
def clasificar_reclamos(request: ReclamosRequest):
    """
    Endpoint que recibe una lista de reclamos y devuelve su clasificación.

    Args:
        request (ReclamosRequest): Objeto que contiene una lista de textos.

    Returns:
        dict: Lista de resultados donde cada texto se clasifica en una categoría.

    Ejemplo de respuesta:
        {
            "resultados": [
                {"reclamo": "falta papel", "clasificado_en": "maestranza"}
            ]
        }
    """
    
    try:
        # Clasifica los textos usando el modelo previamente cargado
        predicciones = clf.clasificar(request.textos)
        
        # Devuelve los resultados como lista de pares: texto + categoría
        return {"resultados": [
            {"reclamo": texto, "clasificado_en": categoria}
            for texto, categoria in zip(request.textos, predicciones)
        ]}
    except Exception as e:
        # Captura errores durante la clasificación y los devuelve
        return {"error": str(e)}

# -------------------------------------------------------------------
# Monta la carpeta 'static' para servir archivos HTML (como index.html)
# Esto permite que al visitar la raíz ("/") se cargue la interfaz visual
# -------------------------------------------------------------------
app.mount("/", StaticFiles(directory="static", html=True), name="static")
