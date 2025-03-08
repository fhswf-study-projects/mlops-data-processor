import joblib
import os
from data_preprocessing import full_preprocessing_pipeline

# Lade und verarbeite die Daten
preprocessor, X, y = full_preprocessing_pipeline("data/raw/adult.csv")

# Trainiere den Preprocessor
preprocessor.fit(X)

# Speicherpfad
preprocessor_path = "models/preprocessor.joblib"

# Falls bereits eine `preprocessor.joblib` existiert, lösche sie
if os.path.exists(preprocessor_path):
    os.remove(preprocessor_path)
    print(f"Alte `preprocessor.joblib` wurde gelöscht.")

# Speichere den neuen trainierten Preprocessor
joblib.dump(preprocessor, preprocessor_path)

print("Preprocessing pipeline trained and saved successfully.")
