import logging

from pandas import DataFrame, cut
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from app.core.celery.app import current_app
from app.core.dvc_client import DVCClient

logger = logging.getLogger(__name__)


@current_app.task(name="data_transformation.engineer_features")
def engineer_features(*args, **kwargs):
    """
    Enhances the dataset by adding new features.

    Parameters:
        df (pd.DataFrame): The input dataframe.

    Returns:
        dict: A dict with the s3 path to the dataframe with additional features.
    """
    logger.info("Start Feature Engineering...")
    # pick first non empty
    data_paths = list(filter(None, args))[0]
    mode = kwargs["body"]["mode"]
    dvc_client = DVCClient()

    # Read data
    df = dvc_client.read_data_from(data_paths["df"])
    ###

    # Type check on returned data
    if not isinstance(df, DataFrame):
        logger.error("Provided path can't be read as dataframe.")
        raise TypeError("df meant to be a DataFrame, but it is not")

    # Create age bins
    bins = [0, 25, 40, 60, 100]
    labels = ["Young", "Middle-aged", "Experienced", "Senior"]
    df["age-group"] = cut(df["age"], bins=bins, labels=labels)

    if mode != "predict":
        # Encode income as binary (0 = <=50K, 1 = >50K)
        df["income"] = df["income"].apply(lambda x: 1 if ">50K" in x else 0)

    return {
        "df": dvc_client.save_data_to(df, "engineer_features/df.jobib"),
    }


@current_app.task(name="data_transformation.encode_data")
def encode_data(*args, **kwargs):
    """
    Prepares the dataset for model training by encoding categorical variables
    and scaling numerical features.

    Returns:
        dict: A dict containing the s3 paths to preprocessing pipeline, feature matrix X, and target variable y.
    """
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
        logger.error("Provided path can't be read as dataframe.")
        raise TypeError("df meant to be a DataFrame, but it is not")

    if mode == "predict":
        preprocessor = dvc_client.read_data_from(
            "encode_data/column_transformer.joblib"
        )
        assert len(df) == 1
        return {
            "features": dvc_client.save_data_to(
                df, destination="encode_data/features.joblib"
            )
        }

    # Definiere numerische und kategorische Spalten
    numeric_features = [
        "age",
        "educational-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]
    categorical_features = [
        "age-group",
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

    X = df.drop("income", axis=1)
    y = df["income"]

    return {
        "pipeline": dvc_client.save_data_to(
            preprocessor, destination="encode_data/column_transformer.joblib"
        ),
        "features": dvc_client.save_data_to(
            X, destination="encode_data/features.joblib"
        ),
        "dependent": dvc_client.save_data_to(
            y, destination="encode_data/dependent.joblib"
        ),
    }
