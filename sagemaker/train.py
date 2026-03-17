import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

def main():
    # SageMaker environment variables (fallback to local)
    data_dir = os.environ.get("SM_CHANNEL_TRAIN", "./data")
    model_dir = os.environ.get("SM_MODEL_DIR", "./model")
    os.makedirs(model_dir, exist_ok=True)

    file_path = os.path.join(data_dir, "Telco-Customer-Churn.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found at {file_path}")

    print("🔥 TRAIN.PY IS RUNNING 🔥")
    print("DATA_DIR:", data_dir)
    print("MODEL_DIR:", model_dir)

    # Load data
    df = pd.read_csv(file_path)

    # Drop customerID (not useful for prediction)
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    # Encode target
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    # Encode categorical features
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Split features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print("✅ Model saved at:", model_path)

if __name__ == "__main__":
    main()