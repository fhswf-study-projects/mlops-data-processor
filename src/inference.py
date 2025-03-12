import joblib
import pandas as pd
from data_preprocessing import clean_data, feature_engineering, preprocess_data, full_preprocessing_pipeline

def make_inference(best_model_path):
    """
        Loads a trained model and makes a test prediction.

        Parameters:
            best_model_path (str): Path to the trained model file.

        Returns:
            str: The predicted income class (">50K" or "<=50K").
    """
    if best_model_path is None:
        print("No optimized model available for inference.")
        return None

    # Load the trained model (Pipeline including the preprocessor)
    model = joblib.load(best_model_path)

    # Define a new sample test input (one person's data)
    sample_data = pd.DataFrame([{
        "age": 39, "workclass": "Private", "education": "Bachelors",
        "education_num": 13, "marital_status": "Never-married", "occupation": "Adm-clerical",
        "relationship": "Not-in-family", "race": "White", "sex": "Male",
        "capital_gain": 2174, "capital_loss": 0, "hours_per_week": 40, "native_country": "United-States",
        "fnlwgt": 226802, "income": ">50K"
    }])

    # Apply cleaning and feature engineering
    sample_data = clean_data(sample_data)
    sample_data = feature_engineering(sample_data)

    # Drop the target column if it exists (not needed for inference)
    if "income" in sample_data.columns:
        sample_data = sample_data.drop(columns=["income"])

    # Direkt die Vorhersage aus der Pipeline holen, ohne extra Transformation:
    prediction = model.predict(sample_data)[0]
    predicted_class = ">50K" if prediction == 1 else "<=50K"

    print(f"Test Prediction: {predicted_class}")
    return predicted_class

