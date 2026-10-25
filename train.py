
import pandas as pd
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load dataset
df = sns.load_dataset("titanic")

# 2. Select input features
X = df[
    ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
].copy()

y = df["survived"]

# 3. Handle missing values
X["age"] = X["age"].fillna(X["age"].median())
X["embarked"] = X["embarked"].fillna(
    X["embarked"].mode()[0]
)

# 4. Convert categorical columns into numeric columns
X = pd.get_dummies(
    X,
    columns=["sex", "embarked"],
    drop_first=True,
    dtype=int
)

# 5. Keep the exact same 8 features
features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
    "sex_male",
    "embarked_Q",
    "embarked_S"
]

X = X.reindex(columns=features, fill_value=0)

# 6. Split the data
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 7. Train SVC model
svc = SVC(probability=True, random_state=42)
svc.fit(x_train, y_train)

# 8. Evaluate model
y_pred = svc.predict(x_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 9. Save the trained model
with open("svc.pkl", "wb") as file:
    pickle.dump(svc, file)

print("\nModel saved successfully as svc.pkl")
