import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Cargar el modelo
with open("modelo_svc.pkl", "rb") as f:
    modelo = joblib.load(f)

# Inicializar FastAPI
app = FastAPI()

# Definir la estructura de los datos de entrada
class InputData(BaseModel):
    features: list

@app.get("/")
def saludo():
    return {"mensaje": "Modelo de Machine Learning!",
            "saludo": "Que fue Harold"}

@app.post("/predict/")
def predict(data: InputData):
    # Convertir los datos de entrada a un array numpy
    X = np.array(data.features).reshape(1, -1)
    
    # Hacer la predicción
    prediccion = modelo.predict(X)
    
    return {"prediction": prediccion.tolist()}

# Ejecutar solo si se corre localmente
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
