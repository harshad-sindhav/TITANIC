# 🚢 Titanic Survival Prediction

A Machine Learning project that predicts whether a passenger survived the Titanic disaster using **Logistic Regression**. The project uses Python, Pandas, Scikit-learn, Seaborn, Matplotlib, and Streamlit to build an interactive prediction application.

## 📌 Project Overview

The goal of this project is to predict passenger survival based on passenger information such as age, gender, ticket class, fare, family members aboard, and port of embarkation.

The project includes data preprocessing, model training, model evaluation, and an interactive Streamlit web application.

## ✨ Features

- Explore and preview the Titanic dataset.
- Handle missing values in the dataset.
- Convert categorical variables into numerical features.
- Train a Logistic Regression classification model.
- Evaluate model performance using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix
  - Classification Report
- Predict passenger survival using user-provided inputs.
- Display estimated probabilities for survival and non-survival.
- Interactive web interface built with Streamlit.

## 🛠️ Technologies Used

- **Python** — Programming language
- **Pandas** — Data manipulation
- **NumPy** — Numerical computing
- **Seaborn** — Dataset loading and visualization
- **Matplotlib** — Data visualization
- **Scikit-learn** — Machine Learning and model evaluation
- **Streamlit** — Interactive web application

## 📂 Project Structure

```text
TITANIC/
│
├── app.py              # Streamlit application
├── train.py            # Model training script (if included)
├── api.py              # FastAPI application (if included)
├── requirements.txt    # Project dependencies
├── README.md           # Project
