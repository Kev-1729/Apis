import joblib
import numpy as np
import pandas as pd
import requests

# Cargar el StandardScaler previamente guardado
scaler = joblib.load("scaler.pkl")

# Lista de nombres de las columnas (ajústala según tu dataset)
columnas = ['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyRate',
       'YearsAtCompany', 'BusinessTravel_Travel_Frequently',
       'JobRole_Laboratory Technician', 'JobRole_Manager',
       'JobRole_Research Director', 'JobRole_Sales Representative',
       'MaritalStatus_Single', 'OverTime_Yes', 'JobLevel_2', 'JobLevel_4',
       'StockOptionLevel_1']

# Definir el nuevo dato
nuevo_dato = np.array([42.0, 933, 19, 57, 20366.0, 2, True, False, False, False, False,
       False, True, False, False, False], dtype=object)

# Convertir booleanos a enteros (1 y 0)
nuevo_dato = np.array([int(x) if isinstance(x, bool) else x for x in nuevo_dato], dtype=float).reshape(1, -1)

# Convertir a DataFrame con nombres de columnas
nuevo_dato_df = pd.DataFrame(nuevo_dato, columns=columnas)

# Escalar el dato
dato_escalado = scaler.transform(nuevo_dato_df)

# Convertir a JSON con la clave correcta ("features")
nuevo_dato_json = {"features": dato_escalado.tolist()[0]}

# URL de la API
url_api = "https://apis-s9ku.onrender.com/predict/"
response = requests.post(url_api, json=nuevo_dato_json)

# Mostrar la predicción
if response.status_code == 200:
    print("Predicción:", response.json())
else:
    print(f"Error {response.status_code}: {response.text}")
