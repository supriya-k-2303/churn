import os
import joblib
import pandas as pd
from sagemaker_inference import content_types, decoder, default_inference_handler

class ChurnPredictor(default_inference_handler.DefaultInferenceHandler):
    
    def default_model_fn(self, model_dir):
        # Load the model from the tar.gz
        model_path = os.path.join(model_dir, "model.joblib")
        model = joblib.load(model_path)
        return model

    def default_input_fn(self, input_data, content_type):
        # Expecting CSV input
        if content_type == content_types.CSV:
            return pd.read_csv(pd.compat.StringIO(input_data))
        else:
            raise ValueError(f"Unsupported content type: {content_type}")

    def default_predict_fn(self, data, model):
        # Return predictions
        return model.predict(data)

    def default_output_fn(self, prediction, accept):
        # Return CSV or JSON
        if accept == content_types.CSV:
            return ",".join(map(str, prediction)), content_types.CSV
        else:
            return prediction.tolist(), "application/json"