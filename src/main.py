import logging
import os
import joblib
from data_preprocessing import full_preprocessing_pipeline
from train import train_model, model_exists
from evaluate import evaluate_model
from hyperparameter_optimization import optimize_hyperparameters, optimized_model_exists as optimized_model_exists
from inference import make_inference
from model_selection import get_best_optimized_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    """
    Runs the ML pipeline but avoids retraining models if they already exist.
    Uses the best available optimized model for inference.
    """
    logging.info("Starting the ML pipeline...")

    # Load and preprocess data
    logging.info("Loading and preprocessing data...")
    preprocessor, X, y = full_preprocessing_pipeline("data/raw/adult.csv")

    # Save preprocessor for later inference
    joblib.dump(preprocessor, "models/preprocessor.joblib")
    logging.info("Preprocessing pipeline saved.")

    # Define models
    models = ["random_forest", "logistic_regression", "svm", "xgboost"]

    # Baseline Training (Train only if necessary)
    logging.info("Checking for existing baseline models...")

    for model in models:
        if not model_exists(model):
            logging.info(f"Training baseline model: {model}...")
            train_model(X, y, preprocessor, model_type=model, optimized=False)
            evaluate_model(X, y, model_name=model)
        else:
            logging.info(f"Skipping training for {model}, already exists.")

    # Hyperparameter Optimization & Training (Only if necessary)
    logging.info("Checking for optimized models...")
    best_params_dict = {}

    for model in models:
        optimized_model_name = f"{model}_optimized_model.joblib"

        if not optimized_model_exists(model):
            logging.info(f"Optimizing hyperparameters for {model}...")
            best_params = optimize_hyperparameters(X, y, preprocessor, model_type=model)
            best_params_dict[model] = best_params

            logging.info(f"Training optimized {model}...")
            train_model(X, y, preprocessor, model_type=model, optimized=True, best_params=best_params_dict[model])
            evaluate_model(X, y, model_name=optimized_model_name)
        else:
            logging.info(f"Skipping optimization for {model}, already exists.")

    # Load the best optimized model for inference
    best_model_path = get_best_optimized_model()
    if best_model_path:
        logging.info(f"Using {best_model_path} for inference...")
        test_prediction = make_inference(best_model_path)  # Preprocessing loaded in inference.py
        logging.info(f"Test Prediction: {test_prediction}")
    else:
        logging.warning("No optimized model found for inference.")

    logging.info("ML pipeline completed.")


if __name__ == "__main__":
    main()
