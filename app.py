import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# PAGE
# ---------------------------------------------------------
st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢", layout="wide")

st.title("🚢 Titanic Survival Prediction")
st.write("Logistic Regression model based on the Titanic dataset from the uploaded notebook.")

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    return sns.load_dataset("titanic")

df = load_data()

st.subheader("1. Dataset")
st.write(f"Dataset shape: {df.shape}")
st.dataframe(df.head())

# ---------------------------------------------------------
# PREPROCESSING
# Same main preprocessing used in the notebook
# ---------------------------------------------------------
X = df[["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]].copy()
Y = df["survived"]

X["age"] = X["age"].fillna(X["age"].median())
X["embarked"] = X["embarked"].fillna(X["embarked"].mode()[0])

X = pd.get_dummies(X, columns=["sex", "embarked"], drop_first=True)

# Make sure the columns match the notebook's 8 features
feature_columns = [
    "pclass", "age", "sibsp", "parch", "fare",
    "sex_male", "embarked_Q", "embarked_S"
]
X = X[feature_columns]

# ---------------------------------------------------------
# TRAIN / TEST
# ---------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.20, random_state=42
)

# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------
lor = LogisticRegression()
lor.fit(x_train, y_train)

pred = lor.predict(x_test)

accuracy = accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred)
report = classification_report(y_test, pred, output_dict=True)

# ---------------------------------------------------------
# MODEL RESULTS
# ---------------------------------------------------------
st.subheader("2. Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", f"{accuracy:.2%}")

with col2:
    st.metric("Test Samples", len(y_test))

st.write("### Confusion Matrix")
st.dataframe(
    pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    )
)

st.write("### Classification Report")
st.dataframe(pd.DataFrame(report).transpose())

# ---------------------------------------------------------
# CONFUSION MATRIX PLOT
# ---------------------------------------------------------
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
st.pyplot(fig)
plt.close(fig)

# ---------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------
st.subheader("3. Predict a Passenger")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3], index=0)
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0)
    sibsp = st.number_input("Siblings / Spouses (sibsp)", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parents / Children (parch)", min_value=0, max_value=10, value=0)

with col2:
    fare = st.number_input("Fare", min_value=0.0, max_value=1000.0, value=80.0)
    sex = st.selectbox("Sex", ["female", "male"])
    embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Convert user input to the exact 8 model features
new_passenger = pd.DataFrame({
    "pclass": [pclass],
    "age": [age],
    "sibsp": [sibsp],
    "parch": [parch],
    "fare": [fare],
    "sex_male": [1 if sex == "male" else 0],
    "embarked_Q": [1 if embarked == "Q" else 0],
    "embarked_S": [1 if embarked == "S" else 0]
})

st.write("### Input sent to model")
st.dataframe(new_passenger)

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
if st.button("🔮 Predict Survival", type="primary"):
    prediction = lor.predict(new_passenger)[0]
    probability = lor.predict_proba(new_passenger)[0]

    st.write("### Prediction")

    if prediction == 1:
        st.success("Passenger predicted to survive.")
    else:
        st.error("Passenger predicted not to survive.")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Probability of Not Survived", f"{probability[0]:.2%}")

    with col2:
        st.metric("Probability of Survived", f"{probability[1]:.2%}")

# ---------------------------------------------------------
# NOTE ABOUT THE ORIGINAL NOTEBOOK
# ---------------------------------------------------------
st.info(
    "The original notebook's titanicpred() print labels are reversed: "
    "it prints 'Passenger will survive' for prediction 0 and "
    "'Passenger will not survive' for prediction 1. "
    "This Streamlit app uses the actual Titanic target meaning: "
    "0 = did not survive, 1 = survived."
)
