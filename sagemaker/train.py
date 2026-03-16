import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

DATA_DIR = "/opt/ml/input/data/train"
MODEL_DIR = "/opt/ml/model"

def main():

    files = os.listdir(DATA_DIR)
    print("Training files:", files)

    df = pd.read_csv(os.path.join(DATA_DIR, files[0]))

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    df['Churn'] = df['Churn'].map({'Yes':1,'No':0})
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))

if __name__ == "__main__":
    main()