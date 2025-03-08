import os


def get_best_optimized_model():
    """
    Finds the best optimized model from the saved models directory.

    Returns:
        str: Path to the best optimized model file.
    """
    model_dir = "models"
    if not os.path.exists(model_dir):
        print("No saved models found.")
        return None

    # Check for optimized models in the directory
    optimized_models = [f for f in os.listdir(model_dir) if "_optimized_model.joblib" in f]

    if not optimized_models:
        print("No optimized models found.")
        return None

    # Select the first optimized model found (you could implement a ranking mechanism)
    best_model_path = os.path.join(model_dir, optimized_models[0])
    print(f"Using optimized model: {best_model_path}")

    return best_model_path
