import joblib
import os
import pandas as pd
from sklearn.metrics import classification_report


def evaluate_model(X, y, model_name="random_forest"):
    """
    Loads a trained model and evaluates its performance.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        model_name (str): Name of the model to evaluate.

    Returns:
        None
    """
    # Define model filename
    model_path = f"models/{model_name}_model.joblib"

    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model file '{model_path}' not found! Please ensure training was successful.")
        return

    # Load the trained model
    model = joblib.load(model_path)
    y_pred = model.predict(X)

    # Generate classification report
    report = classification_report(y, y_pred, output_dict=True)
    df_results = pd.DataFrame(report).T

    # Save results
    results_path = f"results/classification_report_{model_name}.csv"
    os.makedirs("results", exist_ok=True)
    df_results.to_csv(results_path)

    print(f"Evaluation completed. Results saved to '{results_path}'.")
    print(df_results)
