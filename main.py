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
        "mensaje": "Bienvenido a la API de Machine Learning para predicción de empleados.",
        "descripcion": "Esta API usa un modelo de Machine Learning para hacer predicciones basadas en características laborales y personales del empleado.",
        "endpoints": {
            "GET /": "Información sobre la API.",
            "POST /predict/": "Enviar características para obtener una predicción.",
        },
        "variables": {
            "Age": "Edad del empleado.",
            "DailyRate": "Salario diario del empleado.",
            "DistanceFromHome": "Distancia desde la casa al trabajo en kilómetros.",
            "HourlyRate": "Pago por hora del empleado.",
            "MonthlyRate": "Salario mensual del empleado.",
            "YearsAtCompany": "Años trabajando en la empresa.",
            "BusinessTravel_Travel_Frequently": "Si el empleado viaja frecuentemente por trabajo (True/False).",
            "MaritalStatus_Single": "Si el empleado es soltero (True/False).",
            "OverTime_Yes": "Si el empleado trabaja horas extras (True/False).",
            "JobRole_Technical": "Si el empleado tiene un rol técnico (True/False).",
            "JobRole_Management": "Si el empleado tiene un rol de gestión o liderazgo (True/False).",
            "JobRole_Research": "Si el empleado trabaja en investigación y desarrollo (True/False).",
            "JobRole_Sales": "Si el empleado trabaja en ventas (True/False).",
            "JobSeniority_Mid": "Si el empleado tiene un nivel de experiencia intermedio (True/False).",
            "JobSeniority_Senior": "Si el empleado tiene un nivel de experiencia alto (True/False).",
            "StockOption_Basic": "Si el empleado tiene beneficios básicos de acciones en la empresa (True/False)."
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
