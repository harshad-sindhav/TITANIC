
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

st.title("🚢 Titanic Survival Prediction")
st.write(
    "A Machine Learning application that predicts "
    "passenger survival using Logistic Regression."
)

st.divider()

# --------------------------------------------------
# 2. CONSTANTS
# --------------------------------------------------
FEATURES = [
    "pclass", "age", "sibsp", "parch", "fare",
    "sex_male", "embarked_Q", "embarked_S"
]

# --------------------------------------------------
# 3. LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    return sns.load_dataset("titanic")


# --------------------------------------------------
# 4. PREPROCESS DATA
# --------------------------------------------------
@st.cache_data
def preprocess_data():
    df = load_data()

    X = df[
        ["pclass", "sex", "age", "sibsp",
         "parch", "fare", "embarked"]
    ].copy()

    y = df["survived"].copy()

    # Handle missing values
    X["age"] = X["age"].fillna(X["age"].median())
    X["embarked"] = X["embarked"].fillna(
        X["embarked"].mode()[0]
    )

    # Convert categorical features to numeric
    X = pd.get_dummies(
        X,
        columns=["sex", "embarked"],
        drop_first=True,
        dtype=int
    )

    # Ensure consistent feature order
    X = X.reindex(columns=FEATURES, fill_value=0)

    return df, X, y


# --------------------------------------------------
# 5. TRAIN MODEL
# --------------------------------------------------
@st.cache_resource
def train_model():
    df, X, y = preprocess_data()

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ),
        "report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Not Survived", "Survived"],
            output_dict=True,
            zero_division=0
        ),
        "test_samples": len(y_test)
    }

    return model, metrics


try:
    df, X, Y = preprocess_data()
    model, metrics = train_model()
except Exception as e:
    st.error(f"Could not load or train the model: {e}")
    st.stop()


# --------------------------------------------------
# 6. DATASET OVERVIEW
# --------------------------------------------------
st.header("1. Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Passengers", len(df))
col2.metric("Total Features", len(FEATURES))
col3.metric("Missing Values (Raw Dataset)", int(df.isna().sum().sum()))

with st.expander("View Dataset Preview"):
    st.dataframe(df.head(10), use_container_width=True)

with st.expander("View Dataset Information"):
    st.write("Dataset shape:", df.shape)
    st.write("Survival distribution:")
    st.dataframe(
        df["survived"]
        .value_counts()
        .rename(index={0: "Not Survived", 1: "Survived"})
        .rename_axis("Outcome")
        .to_frame("Passengers")
    )


# --------------------------------------------------
# 7. MODEL PERFORMANCE
# --------------------------------------------------
st.header("2. Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
col2.metric("Precision", f"{metrics['precision']:.2%}")
col3.metric("Recall", f"{metrics['recall']:.2%}")
col4.metric("F1-Score", f"{metrics['f1']:.2%}")

st.caption(
    f"Evaluation uses a held-out test set of "
    f"{metrics['test_samples']} passengers. "
    "Results may vary with different data or settings."
)

cm = metrics["confusion_matrix"]

col1, col2 = st.columns(2)

with col1:
    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Survived", "Survived"],
        yticklabels=["Not Survived", "Survived"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("Classification Report")

    report_df = pd.DataFrame(
        metrics["report"]
    ).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )


# --------------------------------------------------
# 8. PASSENGER INPUT
# --------------------------------------------------
st.divider()
st.header("3. Predict Passenger Survival")

st.write("Enter passenger details below.")

with st.form("passenger_form"):
    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox(
            "Passenger Class",
            [1, 2, 3],
            index=0
        )

        sex = st.selectbox(
            "Sex",
            ["female", "male"]
        )

        age = st.number_input(
            "Age",
            min_value=0,
            max_value=100,
            value=18,
            step=1
        )

        sibsp = st.number_input(
            "Siblings / Spouses Aboard",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )

    with col2:
        parch = st.number_input(
            "Parents / Children Aboard",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )

        fare = st.number_input(
            "Ticket Fare",
            min_value=0,
            max_value=1000,
            value=32,
            step=1
        )

        embarked = st.selectbox(
            "Port of Embarkation",
            ["S", "C", "Q"],
            format_func=lambda x: {
                "S": "Southampton (S)",
                "C": "Cherbourg (C)",
                "Q": "Queenstown (Q)"
            }[x]
        )

    submitted = st.form_submit_button(
        "🔮 Predict Survival",
        type="primary",
        use_container_width=True
    )


# --------------------------------------------------
# 9. PREDICTION
# --------------------------------------------------
if submitted:
    passenger = pd.DataFrame([{
        "pclass": pclass,
        "age": age,
        "sibsp": sibsp,
        "parch": parch,
        "fare": fare,
        "sex_male": int(sex == "male"),
        "embarked_Q": int(embarked == "Q"),
        "embarked_S": int(embarked == "S")
    }], columns=FEATURES)

    prediction = int(model.predict(passenger)[0])
    probabilities = model.predict_proba(passenger)[0]
    classes = list(model.classes_)

    survival_probability = float(
        probabilities[classes.index(1)]
    )
    death_probability = float(
        probabilities[classes.index(0)]
    )

    st.divider()
    st.header("4. Prediction Result")

    if prediction == 1:
        st.success("✅ Prediction: Passenger may survive.")
    else:
        st.error("❌ Prediction: Passenger may not survive.")

    col1, col2 = st.columns(2)

    col1.metric(
        "Probability of Survival",
        f"{survival_probability:.2%}"
    )

    col2.metric(
        "Probability of Not Surviving",
        f"{death_probability:.2%}"
    )

    st.subheader("Passenger Details Sent to Model")
    st.dataframe(passenger, use_container_width=True)

    st.caption(
        "Probabilities are model estimates, not guarantees "
        "of what would have happened."
    )


# --------------------------------------------------
# 10. ABOUT
# --------------------------------------------------
with st.expander("About This Project"):
    st.markdown("""
    **Algorithm:** Logistic Regression

    **Dataset:** Titanic dataset provided by Seaborn

    **Target variable:** `survived`

    - `0` = Did not survive
    - `1` = Survived

    **Preprocessing:**
    - Missing age and embarkation values are filled.
    - Categorical features are converted to numeric features.
    - Data is split into training and test sets.

    **Evaluation:** Accuracy, precision, recall, F1-score,
    confusion matrix and classification report.

    This is an educational Machine Learning project.
    """)
