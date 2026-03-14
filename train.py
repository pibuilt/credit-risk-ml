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


def clean_dataset(df, logger):

    logger.info("Removing leakage columns")

    leakage_cols = [
        "out_prncp",
        "out_prncp_inv",
        "total_pymnt",
        "total_pymnt_inv",
        "total_rec_prncp",
        "total_rec_int",
        "total_rec_late_fee",
        "recoveries",
        "collection_recovery_fee",
        "last_pymnt_amnt",
        "last_pymnt_d",
        "next_pymnt_d",
        "last_credit_pull_d",
        "last_fico_range_high",
        "last_fico_range_low",
    ]

    df = df.drop(columns=[c for c in leakage_cols if c in df.columns])

    logger.info("Dropping columns with >50% missing values")

    missing_pct = df.isnull().mean()

    cols_to_drop = missing_pct[missing_pct > 0.5].index.tolist()

    logger.info(f"Columns dropped: {len(cols_to_drop)}")

    df = df.drop(columns=cols_to_drop)

    logger.info(f"Dataset shape after cleaning: {df.shape}")

    return df


def main():

    logger = logging.getLogger(__name__)

    # load dataset
    df = load_dataset(logger)

    # prepare target
    df = prepare_target(df, logger)

    # clean dataset (Day 4 Step 1)
    df = clean_dataset(df, logger)

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

    logger.info("Training baseline model")

    pipeline.fit(X, y)

    logger.info("Model training complete")

    logger.info("Generating sample predictions")

    predictions = pipeline.predict(X[:5])
    probabilities = pipeline.predict_proba(X[:5])[:, 1]

    logger.info(f"Sample predictions: {predictions}")
    logger.info(f"Sample probabilities: {probabilities}")


if __name__ == "__main__":
    setup_logging()
    main()