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
def saludo():
    return {"mensaje": "API de Machine Learning en Render con preprocesamiento automático!",
           "prueba": "'features': [42.0, 933, 19, 57, 20366.0, 2, true, false, false, false, false, false, true, false, false, false]"}

@app.post("/predict/")
def predict(data: InputData):
    X = pd.DataFrame([data.features], columns=columnas)

    X = X.applymap(lambda x: int(x) if isinstance(x, bool) else x)

    X_scaled = scaler.transform(X)

    prediccion = modelo.predict(X_scaled)

    return {"prediction": prediccion.tolist()}


