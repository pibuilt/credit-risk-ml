import logging
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.cluster import KMeans


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

    # ensure risk_cluster is treated as categorical
    if "risk_cluster" in numeric_features:
        numeric_features.remove("risk_cluster")
        categorical_features.append("risk_cluster")

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


def generate_risk_clusters(X_train, X_val, X_test, numeric_features, n_clusters=5):
    """
    Train a KMeans clustering model on numeric borrower features
    and create a new feature called 'risk_cluster'.
    """

    logger = logging.getLogger(__name__)

    logger.info("Training KMeans risk clustering model")

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    # Fit clustering on training data only
    kmeans.fit(X_train[numeric_features])

    logger.info("Assigning risk clusters")

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    X_train["risk_cluster"] = kmeans.predict(X_train[numeric_features])
    X_val["risk_cluster"] = kmeans.predict(X_val[numeric_features])
    X_test["risk_cluster"] = kmeans.predict(X_test[numeric_features])

    logger.info("Risk clustering completed")

    return X_train, X_val, X_test