# app.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load your trained model (pipeline) once
model = joblib.load("depression_model.pkl")  # your saved pipeline

app = FastAPI(title="Family Depression Predictor")

# Define input schema
class PatientData(BaseModel):
    Age: float
    Number_of_Children: float
    Income: float
    Chronic_Medical_Conditions: float
    # Categorical features encoded as numeric (after one-hot)
    Marital_Status_Married: float
    Marital_Status_Single: float
    Marital_Status_Widowed: float
    Education_Level_Bachelor: float
    Education_Level_High_School: float
    Education_Level_Master: float
    Physical_Activity_Level_Moderate: float
    Physical_Activity_Level_Sedentary: float
    Physical_Activity_Level_Active: float
    Employment_Status_Employed: float
    Employment_Status_Unemployed: float
    Alcohol_Consumption_High: float
    Alcohol_Consumption_Low: float
    Alcohol_Consumption_Moderate: float
    Dietary_Habits_Moderate: float
    Dietary_Habits_Unhealthy: float
    Sleep_Patterns_Fair: float
    Sleep_Patterns_Good: float
    Sleep_Patterns_Poor: float
    History_of_Mental_Illness_Yes: float

@app.post("/predict")
def predict(data: PatientData):
    import traceback
    try:
        # Convert input to numpy array
        x = np.array([[
            data.Age,
            data.Number_of_Children,
            data.Income,
            data.Chronic_Medical_Conditions,
            data.Marital_Status_Married,
            data.Marital_Status_Single,
            data.Marital_Status_Widowed,
            data.Education_Level_Bachelor,
            data.Education_Level_High_School,
            data.Education_Level_Master,
            data.Physical_Activity_Level_Moderate,
            data.Physical_Activity_Level_Sedentary,
            data.Physical_Activity_Level_Active,
            data.Employment_Status_Employed,
            data.Employment_Status_Unemployed,
            data.Alcohol_Consumption_High,
            data.Alcohol_Consumption_Low,
            data.Alcohol_Consumption_Moderate,
            data.Dietary_Habits_Moderate,
            data.Dietary_Habits_Unhealthy,
            data.Sleep_Patterns_Fair,
            data.Sleep_Patterns_Good,
            data.Sleep_Patterns_Poor,
            data.History_of_Mental_Illness_Yes
        ]])

        # Make prediction
        pred = model.predict(x)[0]

        # Return result
        return {"family_history_of_depression_prediction": float(pred)}

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}
