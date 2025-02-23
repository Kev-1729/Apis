import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Cargar el modelo y el StandardScaler
modelo = joblib.load("modelo_svc.pkl")
scaler = joblib.load("scaler.pkl")

# Definir las columnas esperadas (ajústala según tu dataset)
columnas = ['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyRate',
            'YearsAtCompany', 'BusinessTravel_Travel_Frequently',
            'JobRole_Laboratory Technician', 'JobRole_Manager',
            'JobRole_Research Director', 'JobRole_Sales Representative',
            'MaritalStatus_Single', 'OverTime_Yes', 'JobLevel_2', 'JobLevel_4',
            'StockOptionLevel_1']

# Inicializar FastAPI
app = FastAPI()

# Definir la estructura de los datos de entrada
class InputData(BaseModel):
    features: list

@app.get("/")
def saludo():
    return {"mensaje": "API de Machine Learning en Render con preprocesamiento automático!"}

@app.post("/predict/")
def predict(data: InputData):
    # Convertir los datos de entrada en un DataFrame con nombres de columnas
    X = pd.DataFrame([data.features], columns=columnas)

    # Convertir booleanos a enteros (1 y 0)
    X = X.applymap(lambda x: int(x) if isinstance(x, bool) else x)

    # Aplicar la transformación con el StandardScaler
    X_scaled = scaler.transform(X)

    # Hacer la predicción con el modelo
    prediccion = modelo.predict(X_scaled)

    return {"prediction": prediccion.tolist()}


