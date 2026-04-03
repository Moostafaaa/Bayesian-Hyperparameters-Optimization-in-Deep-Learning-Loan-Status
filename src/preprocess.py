"""
preprocess.py
-------------
Data cleaning, feature engineering, and scaling pipeline
for the Loan Default Prediction dataset (Kaggle PS S4E8).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Full preprocessing pipeline. Returns scaled numpy arrays
    ready for PyTorch DataLoader consumption.

    Steps:
      1. Drop id column
      2. Binary-encode cb_person_default_on_file (Y→1, N→0)
      3. Ordinal-encode loan_grade (A→1, B→2, C→3, D→4)
      4. One-hot encode person_home_ownership and loan_intent
      5. Log-transform person_income (suppresses extreme outliers)
      6. Drop rows with NaN (primarily from unknown loan_grade values)
      7. Train/test split → StandardScaler fit on train, transform on test
    """
    df = pd.read_csv(csv_path)
    df.drop(columns=["id"], inplace=True)

    # Binary encoding
    df["cb_person_default_on_file"] = df["cb_person_default_on_file"].map({"Y": 1, "N": 0})

    # Ordinal encoding (A=best, D=worst)
    grade_map = {"A": 1, "B": 2, "C": 3, "D": 4}
    df["loan_grade"] = df["loan_grade"].map(grade_map)

    # One-hot encoding for nominal categorical features
    df = pd.get_dummies(df, columns=["person_home_ownership", "loan_intent"],
                        prefix=["home", "intent"])

    # Log-transform income: range 4,200 → 1,900,000 (450× span)
    # Log compresses the distribution before StandardScaler is applied
    df["person_income"] = np.log(df["person_income"])

    # Remove rows with NaN (1,191 rows from unrecognised loan_grade values)
    df.dropna(inplace=True)

    X = df.drop(columns=["loan_status"])
    y = df["loan_status"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
