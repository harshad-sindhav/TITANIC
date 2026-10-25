
from pathlib import Path
import pickle

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Predict Titanic passenger survival using SVC.",
    version="1.0.0"
)

# Load the same model created by train.py
MODEL_PATH = Path(__file__).with_name("svc.pkl")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

FEATURES = [
    "pclass", "age", "sibsp", "parch", "fare",
    "sex_male", "embarked_Q", "embarked_S"
]


# Validate user input
class Passenger(BaseModel):
    pclass: int = Field(ge=1, le=3)
    sex: Literal["male", "female"]
    age: float = Field(ge=0, le=100)
    sibsp: int = Field(ge=0, le=10)
    parch: int = Field(ge=0, le=10)
    fare: float = Field(ge=0, le=1000)
    embarked: Literal["C", "Q", "S"]


@app.get("/")
def home():
    return {
        "message": "Titanic Survival Prediction API is running",
        "docs": "/docs"
    }


@app.post("/predict")
def predict(passenger: Passenger):
    # Convert passenger details into the model's 8 features
    input_data = pd.DataFrame([{
        "pclass": passenger.pclass,
        "age": passenger.age,
        "sibsp": passenger.sibsp,
        "parch": passenger.parch,
        "fare": passenger.fare,
        "sex_male": int(passenger.sex == "male"),
        "embarked_Q": int(passenger.embarked == "Q"),
        "embarked_S": int(passenger.embarked == "S")
    }], columns=FEATURES)

    prediction = int(model.predict(input_data)[0])

    probabilities = model.predict_proba(input_data)[0]
    classes = list(model.classes_)

    survival_probability = float(
        probabilities[classes.index(1)]
    )
    death_probability = float(
        probabilities[classes.index(0)]
    )

    return {
        "prediction": prediction,
        "result": (
            "Passenger predicted to survive"
            if prediction == 1
            else "Passenger predicted not to survive"
        ),
        "survival_probability": round(survival_probability, 4),
        "not_survival_probability": round(death_probability, 4)
    }
