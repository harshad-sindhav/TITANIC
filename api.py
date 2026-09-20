from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

# Load trained model
with open("svc.pkl", "rb") as file:
    svc = pickle.load(file)


@app.get("/")
def home():
    return {
        "message": "Titanic Survival Prediction API is running"
    }


@app.post("/predict")
def predict(
    pclass: int,
    age: float,
    sibsp: int,
    parch: int,
    fare: float,
    sex_male: int,
    embarked_Q: int,
    embarked_S: int
):

    input_data = pd.DataFrame([[
        pclass,
        age,
        sibsp,
        parch,
        fare,
        sex_male,
        embarked_Q,
        embarked_S
    ]], columns=[
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare",
        "sex_male",
        "embarked_Q",
        "embarked_S"
    ])

    prediction = svc.predict(input_data)[0]

    probability = svc.predict_proba(input_data)[0]

    if prediction == 1:
        result = "Passenger will survive"
    else:
        result = "Passenger will not survive"

    return {
        "prediction": int(prediction),
        "result": result,
        "survival_probability": float(probability[1]),
        "not_survival_probability": float(probability[0])
    }