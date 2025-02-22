import logging
from data_preprocessing import load_data, preprocess_data
from train import train_model
from evaluate import evaluate_model
from hyperparameter_optimization import optimize_hyperparameters

# Logging einrichten
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    logging.info("Starte die Machine Learning Pipeline...")

    # Daten laden
    logging.info("Lade die Daten...")
    df = load_data("data/raw/adult.csv")

    # Daten vorbereiten (Feature Engineering)
    logging.info("Starte Feature Engineering...")
    preprocessor, X, y = preprocess_data(df)

    print(X.head())  # Zeigt die transformierten Features
    print(y.head())  # Zeigt die Zielvariable


    # Basismodell trainieren
    logging.info("Trainiere das erste Modell ohne Hyperparameter-Optimierung...")
    train_model(X, y, preprocessor, optimized=False)


    # Hyperparameter-Optimierung mit Optuna
    logging.info("🔹 Starte Hyperparameter-Optimierung...")
    best_params = optimize_hyperparameters(X, y, preprocessor)

    # Trainiere das beste Modell mit den optimierten Hyperparametern
    logging.info("🔹 Trainiere das Modell mit den besten Hyperparametern...")
    train_model(X, y, preprocessor, optimized=True, best_params=best_params)

    """
    # Evaluierung des Modells
    logging.info("🔹 Evaluierung des Modells...")
    evaluate_model(X, y)

    logging.info("Machine Learning Pipeline abgeschlossen!")
    """

if __name__ == "__main__":
    main()
