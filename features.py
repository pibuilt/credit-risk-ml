import logging
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer

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

def build_preprocessing_pipeline(
    numeric_features,
    categorical_features,
    text_features,
    logger
):

    logger.info("Building preprocessing pipeline")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = [
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ]

    # add text pipelines
    for text_col in text_features:

        text_pipeline = Pipeline(
            steps=[
                (
                    "fillna",
                    FunctionTransformer(
                        lambda x: x.fillna("").astype(str),
                        validate=False
                    ),
                ),
                ("tfidf", TfidfVectorizer(max_features=500)),
            ]
        )

        transformers.append(
            (f"text_{text_col}", text_pipeline, text_col)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    logger.info("Preprocessing pipeline created")

    return preprocessor