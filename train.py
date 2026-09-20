import pandas as pd
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Titanic dataset
df = sns.load_dataset("titanic")

# Input features
X = df[
    ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
].copy()

# Target
y = df["survived"]

# Missing values
X["age"] = X["age"].fillna(X["age"].median())
X["embarked"] = X["embarked"].fillna(X["embarked"].mode()[0])

# Convert categorical columns
X = pd.get_dummies(
    X,
    columns=["sex", "embarked"],
    drop_first=True
)

# Model ke features
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

X = X[features]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# SVC model
svc = SVC(probability=True)

# Model train
svc.fit(x_train, y_train)

# Model ko save karo
with open("svc.pkl", "wb") as file:
    pickle.dump(svc, file)

print("svc.pkl successfully created!")