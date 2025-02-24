import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

modelo = joblib.load("modelo_svc.pkl")
scaler = joblib.load("scaler.pkl")

columnas = ['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyRate',
            'YearsAtCompany', 'BusinessTravel_Travel_Frequently',
            'JobRole_Laboratory Technician', 'JobRole_Manager',
            'JobRole_Research Director', 'JobRole_Sales Representative',
            'MaritalStatus_Single', 'OverTime_Yes', 'JobLevel_2', 'JobLevel_4',
            'StockOptionLevel_1']

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    features: list

@app.get("/")
def home():
    return {
        "mensaje": "Bienvenido a la API de Machine Learning para predicción de deserción laboral.",
        "descripcion": "Esta API usa un modelo de Machine Learning para hacer predicciones de renuncias basadas en características de empleados.",
        "endpoints": {
            "GET /": "Información sobre la API.",
            "POST /predict/": "Enviar características para obtener una predicción.",
        },
        "ejemplo_prediccion": {
            "url": "/predict/",
            "metodo": "POST",
            "cuerpo": {
                "features": [42.0, 933, 19, 57, 20366.0, 2, True, False, False, False, False, False, True, False, False, False]
            },
            "respuesta_esperada": {
                "prediction": [1] 
            }
        }
    }


@app.post("/predict/")
def predict(data: InputData):
    X = pd.DataFrame([data.features], columns=columnas)

    X = X.applymap(lambda x: int(x) if isinstance(x, bool) else x)

    X_scaled = scaler.transform(X)

    prediccion = modelo.predict(X_scaled)

    return {"prediction": prediccion.tolist()}
