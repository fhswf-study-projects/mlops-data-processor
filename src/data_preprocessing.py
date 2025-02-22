import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_data(filepath: str):
    """Lädt die CSV-Datei und gibt einen Pandas DataFrame zurück."""
    df = pd.read_csv(filepath)
    return df


def preprocess_data(df):
    """Bereinigt und transformiert die Daten."""

    # Definiere numerische und kategorische Spalten
    numeric_features = ["age", "fnlwgt", "educational-num", "capital-gain", "capital-loss", "hours-per-week"]
    categorical_features = ["workclass", "education", "marital-status", "occupation",
                            "relationship", "race", "gender", "native-country"]

    # Preprocessing Pipeline
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])

    X = df.drop("income", axis=1)
    y = df["income"].apply(lambda x: 1 if x == ">50K" else 0)  # Zielvariable binär codieren

    return preprocessor, X, y
