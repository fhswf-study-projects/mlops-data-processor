import joblib
import pandas as pd
from sklearn.metrics import classification_report

def evaluate_model(X, y):
    """Lädt das Modell und führt eine Evaluierung durch."""

    # Modell laden
    model = joblib.load("models/final_model.joblib")
    y_pred = model.predict(X)

    # Evaluierungsmetriken berechnen
    report = classification_report(y, y_pred, output_dict=True)
    df_results = pd.DataFrame(report).T
    print(df_results)

    # Ergebnisse speichern
    df_results.to_csv("results/classification_report.csv")
    print("✅ Evaluierung abgeschlossen! Ergebnisse gespeichert in results/classification_report.csv")
