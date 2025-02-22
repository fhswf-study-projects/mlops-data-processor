import joblib
import pandas as pd

# Neues Beispiel-Datenpunkt (eine Person)
sample_data = pd.DataFrame([{
    "age": 39, "workclass": "Private", "fnlwgt": 77516, "education": "Bachelors",
    "educational-num": 13, "marital-status": "Never-married", "occupation": "Adm-clerical",
    "relationship": "Not-in-family", "race": "White", "gender": "Male",
    "capital-gain": 2174, "capital-loss": 0, "hours-per-week": 40, "native-country": "United-States"
}])

# Modell laden
model = joblib.load("models/RandomForest.joblib")

# Vorhersage machen
prediction = model.predict(sample_data)
print("Predicted Income Class:", ">50K" if prediction[0] == 1 else "<=50K")
