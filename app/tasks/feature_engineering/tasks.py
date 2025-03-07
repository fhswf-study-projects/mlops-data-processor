import logging

from pandas import DataFrame
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from app.core.celery.app import current_app
from app.core.dvc_client import DVCClient

logger = logging.getLogger(__name__)


@current_app.task(name="feature_engineering.clean_and_tranform")
def clean_and_tranform(*args, **kwargs):
    """Bereinigt und transformiert die Daten."""
    logger.info("Starte Feature Engineering...")
    # pick first non empty
    data_paths = list(filter(None, args))[0]
    mode = kwargs["body"]["mode"]
    dvc_client = DVCClient()

    # Read data
    df = dvc_client.read_data_from(data_paths["df"])
    ###

    # Type check on returned data
    if not isinstance(df, DataFrame):
        raise TypeError("df meant to be a DataFrame, but it is not")

    # Definiere numerische und kategorische Spalten
    numeric_features = [
        "age",
        "fnlwgt",
        "educational-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]
    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "gender",
        "native-country",
    ]

    # Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    if mode == "predict":
        assert len(df) == 1
        return {
            "features": dvc_client.save_data_to(
                df, destination="clean_and_tranform/features.joblib"
            )
        }

    X = df.drop("income", axis=1)
    y = df["income"].apply(
        lambda x: 1 if x == ">50K" else 0
    )  # Zielvariable binär codieren

    return {
        "pipeline": dvc_client.save_data_to(
            preprocessor, destination="clean_and_tranform/column_transformer.joblib"
        ),
        "features": dvc_client.save_data_to(
            X, destination="clean_and_tranform/features.joblib"
        ),
        "dependent": dvc_client.save_data_to(
            y, destination="clean_and_tranform/dependent.joblib"
        ),
    }
