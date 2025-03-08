import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads the dataset from a CSV file.

    Parameters:
        filepath (str): The path to the dataset.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the dataset.
    """
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
    ]
    df = pd.read_csv(filepath, names=column_names, na_values="?", skiprows=1)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataset by handling missing values and removing irrelevant columns.

    Parameters:
        df (pd.DataFrame): The input dataframe.

    Returns:
        pd.DataFrame: The cleaned dataframe.
    """
    # Remove 'fnlwgt' only if it exists
    if "fnlwgt" in df.columns:
        df.drop(columns=["fnlwgt"], inplace=True)

    # Replace missing values in categorical columns with "Unknown"
    df.fillna({"workclass": "Unknown", "occupation": "Unknown", "native_country": "Unknown"}, inplace=True)

    return df



def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhances the dataset by adding new features.

    Parameters:
        df (pd.DataFrame): The input dataframe.

    Returns:
        pd.DataFrame: The dataframe with additional features.
    """
    # Create age bins
    bins = [0, 25, 40, 60, 100]
    labels = ["Young", "Middle-aged", "Experienced", "Senior"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)

    # Encode income as binary (0 = <=50K, 1 = >50K)
    df["income"] = df["income"].apply(lambda x: 1 if ">50K" in x else 0)

    return df


def preprocess_data(df: pd.DataFrame):
    """
    Prepares the dataset for model training by encoding categorical variables
    and scaling numerical features.

    Parameters:
        df (pd.DataFrame): The input dataframe.

    Returns:
        tuple: A tuple containing the preprocessing pipeline, feature matrix X, and target variable y.
    """
    # Define categorical and numerical columns
    categorical_cols = ["workclass", "education", "marital_status", "occupation",
                        "relationship", "race", "sex", "native_country", "age_group"]
    numerical_cols = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])

    # Define feature matrix X and target variable y
    X = df.drop(columns=["income"])
    y = df["income"]

    return preprocessor, X, y


def analyze_dataset(X, y):
    """
    Performs a full dataset check after preprocessing.

    Parameters:
        X (pd.DataFrame): The feature matrix after preprocessing.
        y (pd.Series): The target variable.

    Returns:
        None (Prints dataset summary to console).
    """
    print("\nDATASET ANALYSIS AFTER PREPROCESSING\n")

    # Overview of dataset shape
    print(f"Dataset contains {X.shape[0]:,} rows and {X.shape[1]} features.")

    # Checking for missing values
    missing_values = X.isnull().sum()
    missing_values = missing_values[missing_values > 0]

    if missing_values.empty:
        print("No missing values found in the dataset!")
    else:
        print("Missing values detected:")
        print(missing_values)

    # Display feature types
    print("\nFeature Types:")
    print(X.dtypes.value_counts())

    # Show example rows
    print("\nSample Data (First 5 rows):")
    print(X.head())

    # Show target variable distribution
    print("\nTarget Variable Distribution:")
    print(y.value_counts(normalize=True) * 100)


def full_preprocessing_pipeline(filepath: str):
    """
    Loads, cleans, processes, and transforms the dataset for machine learning.

    Parameters:
        filepath (str): The path to the dataset.

    Returns:
        tuple: A tuple containing the preprocessing pipeline, feature matrix X, and target variable y.
    """
    df = load_data(filepath)
    df = clean_data(df)
    df = feature_engineering(df)

    # Preprocess and get transformed data
    preprocessor, X, y = preprocess_data(df)

    # Perform dataset analysis after preprocessing
    analyze_dataset(X, y)

    return preprocessor, X, y
