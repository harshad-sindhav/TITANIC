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
├── README.md           # Project documentation
└── svc.pkl             # Saved SVC model (if used)
```

*Note: Keep only the files that actually exist in your repository. The improved Logistic Regression version of `app.py` trains its model within the application and does not require `svc.pkl`.*

## 📊 Dataset

This project uses the Titanic dataset available through Seaborn.

The dataset contains passenger information, including:

- `pclass` — Passenger class
- `sex` — Passenger gender
- `age` — Passenger age
- `sibsp` — Siblings and spouses aboard
- `parch` — Parents and children aboard
- `fare` — Ticket fare
- `embarked` — Port of embarkation
- `survived` — Survival outcome (target variable)

### Target Variable

- `0` — Did not survive
- `1` — Survived

## ⚙️ Machine Learning Workflow

1. Load the Titanic dataset.
2. Select relevant input features and the target variable.
3. Handle missing age and embarkation values.
4. Encode categorical variables using one-hot encoding.
5. Split the data into training and testing sets.
6. Train a Logistic Regression model.
7. Evaluate the model using classification metrics.
8. Use the trained model to predict passenger survival.

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/harshad-sindhav/TITANIC.git
cd TITANIC
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

If `requirements.txt` is not available, install the required packages:

```bash
python -m pip install streamlit pandas numpy seaborn matplotlib scikit-learn
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL, usually:

`http://localhost:8501`

**Note:** The app needs internet access to download the Seaborn Titanic dataset on its first uncached run.

## 📈 Model Evaluation

The application displays the following evaluation metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Overall proportion of correct predictions |
| Precision | How many predicted survivors actually survived |
| Recall | How many actual survivors were correctly identified |
| F1-score | Balance between precision and recall |
| Confusion Matrix | Comparison of actual and predicted classes |

Actual evaluation scores depend on the dataset, preprocessing, and train-test split. Run the application to obtain the results; do not assume a score before testing.

## 🖥️ How to Use the Application

1. Start the Streamlit application.
2. Review the dataset overview and model performance.
3. Enter passenger details, including class, age, gender, fare, family members, and embarkation port.
4. Click **Predict Survival**.
5. Review the predicted outcome and estimated survival probabilities.

## 🎯 Project Objectives

- Understand the end-to-end Machine Learning workflow.
- Practice data cleaning and feature engineering.
- Apply a classification algorithm to a real-world dataset.
- Evaluate a model using multiple classification metrics.
- Build an interactive Machine Learning application.

## 🔮 Future Improvements

- Compare Logistic Regression with SVC, Random Forest, and other classifiers.
- Add cross-validation and hyperparameter tuning.
- Improve feature engineering and model explainability.
- Save the trained model separately from the Streamlit application.
- Add automated tests and deploy the application online.

## 👨‍💻 Author

**Harshad Sindhav**

GitHub: [harshad-sindhav](https://github.com/harshad-sindhav)

Project Repository: [TITANIC](https://github.com/harshad-sindhav/TITANIC)

---

*This project was developed for learning and demonstrating fundamental Machine Learning and data science concepts.*
