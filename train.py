import logging
import pandas as pd

from explore_data import prepare_target
from features import get_feature_groups, build_preprocessing_pipeline

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_dataset(logger):

    logger.info("Loading dataset")

    df = pd.read_csv(
        "data/loan_2019_20.csv",
        low_memory=False
    )

    logger.info(f"Dataset shape: {df.shape}")

    return df


def main():

    logger = logging.getLogger(__name__)

    # load dataset
    df = load_dataset(logger)

    # prepare target
    df = prepare_target(df, logger)

    # split features and target
    X = df.drop(columns=["default"])
    y = df["default"]

    logger.info(f"Feature matrix shape: {X.shape}")

    # identify feature groups
    numeric_features, categorical_features, text_features = get_feature_groups(
        X, logger
    )

    # build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(
        numeric_features,
        categorical_features,
        text_features,
        logger
    )

    logger.info("Feature pipeline ready")

    logger.info("Training pipeline initialization complete")

    logger.info("Creating ML pipeline")

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    logger.info("ML pipeline created")


if __name__ == "__main__":
    setup_logging()
    main()