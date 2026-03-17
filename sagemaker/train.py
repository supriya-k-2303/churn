import os
import pandas as pd
import joblib
import tarfile

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def main():
    print("🔥 TRAIN.PY IS RUNNING 🔥")
    
    # SageMaker channels
    DATA_DIR = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    
    print("DATA_DIR:", DATA_DIR)
    print("MODEL_DIR:", MODEL_DIR)
    
    # Load data
    file_path = os.path.join(DATA_DIR, "Telco-Customer-Churn.csv")
    df = pd.read_csv(file_path)
    print("✅ Data loaded")
    
    # Preprocessing
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df.drop('customerID', axis=1, inplace=True)
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(exclude=['object']).columns
    
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numerical_cols)
    ])
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)
    print("✅ Model trained")
    
    # Save the model locally first
    local_model_file = "model.joblib"
    joblib.dump(model, local_model_file)
    
    # Package into tar.gz for SageMaker
    tar_path = os.path.join(MODEL_DIR, "model.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(local_model_file, arcname=os.path.basename(local_model_file))
    
    print(f"✅ Model packaged and saved at {tar_path}")

if __name__ == "__main__":
    main()