import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def main():

    # ✅ Works in BOTH SageMaker and Local
    DATA_DIR = os.environ.get("SM_CHANNEL_TRAIN", "data")
    MODEL_DIR = os.environ.get("SM_MODEL_DIR", "model")

    print("DATA_DIR:", DATA_DIR)
    print("MODEL_DIR:", MODEL_DIR)

    # Debug: list files
    print("FILES IN DATA_DIR:", os.listdir(DATA_DIR))

    # ✅ Load dataset (MAKE SURE filename matches exactly)
    file_path = os.path.join(DATA_DIR, "Telco-Customer-Churn.csv")
    df = pd.read_csv(file_path)

    print("Data loaded successfully")

    # -----------------------------
    # Data preprocessing
    # -----------------------------
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df.drop('customerID', axis=1, inplace=True)

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(exclude=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', 'passthrough', numerical_cols)
        ]
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ))
    ])

    # -----------------------------
    # Train
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    print("Model training completed")

    # -----------------------------
    # Save model
    # -----------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "model.pkl")

    joblib.dump(model, model_path)

    print("Model saved at:", model_path)


if __name__ == "__main__":
    main()