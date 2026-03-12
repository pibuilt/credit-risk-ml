import logging


def get_feature_groups(df, logger):

    logger.info("Identifying feature groups")

    numeric_features = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = df.select_dtypes(
        include=["object"]
    ).columns.tolist()

    text_features = [
        "emp_title",
        "title"
    ]

    # remove text columns from categorical
    categorical_features = [
        c for c in categorical_features if c not in text_features
    ]

    # remove target if present
    if "default" in numeric_features:
        numeric_features.remove("default")

    logger.info(f"Numeric features: {len(numeric_features)}")
    logger.info(f"Categorical features: {len(categorical_features)}")
    logger.info(f"Text features: {len(text_features)}")

    return numeric_features, categorical_features, text_features