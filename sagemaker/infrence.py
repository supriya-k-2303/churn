# sagemaker/infrence.py
import os
import joblib
import pandas as pd

class ChurnPredictor:
    def __init__(self, model_path="./model/model.joblib"):
        # Load trained model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = joblib.load(model_path)
        print("✅ Model loaded successfully")

    def preprocess(self, df):
        """
        Preprocess input data to match training features
        """
        # Drop target column if present
        if 'Churn' in df.columns:
            df = df.drop(columns=['Churn'])
        
        # Encode categorical variables (one-hot encoding)
        cat_cols = df.select_dtypes(include='object').columns
        df_processed = pd.get_dummies(df, columns=cat_cols)
        
        # Align columns with training model (fill missing columns with 0)
        if hasattr(self.model, 'feature_names_in_'):
            for col in self.model.feature_names_in_:
                if col not in df_processed.columns:
                    df_processed[col] = 0
            # Ensure same column order
            df_processed = df_processed[self.model.feature_names_in_]
        
        return df_processed

    def predict(self, df):
        df_processed = self.preprocess(df)
        predictions = self.model.predict(df_processed)
        return predictions

if __name__ == "__main__":
    # Example usage
    data_path = "./data/Telco-Customer-Churn.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    df = pd.read_csv(data_path)
    predictor = ChurnPredictor()
    preds = predictor.predict(df)
    
    print("Predictions:")
    print(preds[:10])  # Show first 10 predictions