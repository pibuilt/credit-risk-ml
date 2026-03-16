import logging
import pandas as pd
import json
import os
import matplotlib.pyplot as plt

from explore_data import prepare_target
from features import (
    get_feature_groups,
    build_preprocessing_pipeline,
    generate_risk_clusters
)

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve, brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import optuna

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
        "grade",
        "sub_grade",
        "int_rate",
        "issue_d",
        "earliest_cr_line"
    ]

    df = df.drop(columns=[c for c in leakage_cols if c in df.columns])

    logger.info("Dropping columns with >50% missing values")

    missing_pct = df.isnull().mean()

    cols_to_drop = missing_pct[missing_pct > 0.5].index.tolist()

    logger.info(f"Columns dropped: {len(cols_to_drop)}")

    df = df.drop(columns=cols_to_drop)

    logger.info(f"Dataset shape after cleaning: {df.shape}")

    return df


def split_dataset(X, y, logger):

    logger.info("Creating train/validation/test split")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42
    )

    logger.info(f"Train set shape: {X_train.shape}")
    logger.info(f"Validation set shape: {X_val.shape}")
    logger.info(f"Test set shape: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_model(y_true, y_pred, y_prob, logger):

    logger.info("Evaluating model")

    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob)

    logger.info(f"ROC-AUC: {roc_auc:.4f}")
    logger.info(f"PR-AUC: {pr_auc:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Brier Score: {brier:.4f}")

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "brier_score": float(brier)
    }


def save_metrics(metrics, logger):

    if not os.path.exists("reports"):
        os.makedirs("reports")

    path = "reports/metrics.json"

    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Metrics saved to {path}")


def plot_confusion_matrix(cm, logger):

    if not os.path.exists("reports"):
        os.makedirs("reports")

    plt.figure()

    plt.imshow(cm, interpolation="nearest")

    plt.title("Confusion Matrix")

    plt.colorbar()

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    for i in range(len(cm)):
        for j in range(len(cm[0])):
            plt.text(j, i, cm[i][j], ha="center", va="center")

    path = "reports/confusion_matrix.png"

    plt.savefig(path)

    plt.close()

    logger.info(f"Confusion matrix saved to {path}")


def plot_roc_curve(y_true, y_prob, logger):

    if not os.path.exists("reports"):
        os.makedirs("reports")

    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure()

    plt.plot(fpr, tpr, label="Model")

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    path = "reports/roc_curve.png"

    plt.savefig(path)

    plt.close()

    logger.info(f"ROC curve saved to {path}")


def plot_pr_curve(y_true, y_prob, logger):

    if not os.path.exists("reports"):
        os.makedirs("reports")

    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    plt.figure()

    plt.plot(recall, precision, label="Model")

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title("Precision-Recall Curve")

    plt.legend()

    path = "reports/pr_curve.png"

    plt.savefig(path)

    plt.close()

    logger.info(f"PR curve saved to {path}")

def objective(trial, X_train, y_train, preprocessor):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "num_leaves": trial.suggest_int("num_leaves", 20, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": 42
    }

    model = LGBMClassifier(**params)

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = []

    for train_idx, val_idx in cv.split(X_train, y_train):

        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]

        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]

        pipeline.fit(X_tr, y_tr)

        preds = pipeline.predict_proba(X_val)[:, 1]

        score = roc_auc_score(y_val, preds)

        scores.append(score)

    return sum(scores) / len(scores)


def main():

    logger = logging.getLogger(__name__)

    df = load_dataset(logger)

    df = prepare_target(df, logger)

    df = clean_dataset(df, logger)

    X = df.drop(columns=["default", "loan_status"])
    y = df["default"]

    logger.info(f"Sample feature columns: {list(X.columns)[:10]}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y, logger)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    logger.info(f"Feature matrix shape: {X.shape}")

    # identify feature groups
    numeric_features, categorical_features, text_features = get_feature_groups(
        X_train, logger
    )

    # generate risk clusters
    logger.info("Generating risk clusters using KMeans")

    X_train, X_val, X_test = generate_risk_clusters(
        X_train,
        X_val,
        X_test,
        numeric_features
    )

    # re-detect feature groups (now includes risk_cluster)
    numeric_features, categorical_features, text_features = get_feature_groups(
        X_train, logger
    )

    # build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(
        numeric_features,
        categorical_features,
        text_features,
        logger
    )

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            class_weight="balanced"
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            verbose=-1
        )
    }

    logger.info("Feature pipeline ready")

    logger.info("Starting model comparison with cross-validation")

    best_model_name = None
    best_score = 0

    for model_name, model in models.items():

        logger.info(f"Training model: {model_name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessing", preprocessor),
                ("model", model)
            ]
        )

        roc_scores = []
        pr_scores = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):

            logger.info(f"{model_name} | Fold {fold + 1}")

            X_tr = X_train.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]

            X_val_fold = X_train.iloc[val_idx]
            y_val_fold = y_train.iloc[val_idx]

            pipeline.fit(X_tr, y_tr)

            preds = pipeline.predict_proba(X_val_fold)[:, 1]

            roc = roc_auc_score(y_val_fold, preds)
            pr = average_precision_score(y_val_fold, preds)

            roc_scores.append(roc)
            pr_scores.append(pr)

        mean_roc = sum(roc_scores) / len(roc_scores)
        mean_pr = sum(pr_scores) / len(pr_scores)

        logger.info(f"{model_name} | Mean ROC-AUC: {mean_roc:.4f}")
        logger.info(f"{model_name} | Mean PR-AUC: {mean_pr:.4f}")

        if mean_pr > best_score:
            best_score = mean_pr
            best_model_name = model_name

    logger.info(f"Best model selected: {best_model_name}")

    if best_model_name is None:
        raise RuntimeError("No model was selected during cross-validation")


    # ------------------------------------------------
    # Hyperparameter tuning with Optuna (LightGBM only)
    # ------------------------------------------------

    if best_model_name == "lightgbm":

        logger.info("Starting Optuna hyperparameter tuning for LightGBM")

        study = optuna.create_study(direction="maximize")

        study.optimize(
            lambda trial: objective(trial, X_train, y_train, preprocessor),
            n_trials=5
        )

        logger.info(f"Best Optuna score: {study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")

        best_model = LGBMClassifier(
            **study.best_params,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
            verbose=-1
        )

    else:

        best_model = models[best_model_name]

    logger.info("Training best model on full training data")

    final_pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", best_model)
        ]
    )

    final_pipeline.fit(X_train, y_train)

    logger.info("Model training complete")

    logger.info("Running validation evaluation")

    val_predictions = final_pipeline.predict(X_val)
    val_probabilities = final_pipeline.predict_proba(X_val)[:, 1]

    metrics = evaluate_model(
        y_val,
        val_predictions,
        val_probabilities,
        logger
    )

    save_metrics(metrics, logger)

    plot_confusion_matrix(metrics["confusion_matrix"], logger)
    plot_roc_curve(y_val, val_probabilities, logger)
    plot_pr_curve(y_val, val_probabilities, logger)

    logger.info("Generating sample predictions")

    predictions = final_pipeline.predict(X_test[:5])
    probabilities = final_pipeline.predict_proba(X_test[:5])[:, 1]

    logger.info(f"Sample predictions: {predictions}")
    logger.info(f"Sample probabilities: {probabilities}")


if __name__ == "__main__":
    setup_logging()
    main()