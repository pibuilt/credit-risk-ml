import logging
import joblib
import pandas as pd

from ml.data import prepare_target


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# -----------------------------------
# DATA CLEANING (MATCH TRAINING)
# -----------------------------------

def clean_dataset(df):

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

    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > 0.5].index.tolist()

    df = df.drop(columns=cols_to_drop)

    return df


# -----------------------------------
# MODEL SERVICE
# -----------------------------------

class ModelService:

    def __init__(self, model_path: str):
        self.logger = logging.getLogger(__name__)

        self.logger.info("Loading model")

        self.pipeline = joblib.load(model_path)

        self.logger.info("Model loaded successfully")

    def predict(self, input_df):

        self.logger.info("Running inference")

        probs = self.pipeline.predict_proba(input_df)[:, 1]

        return probs


    def predict_with_risk(self, input_df):
        probs = self.predict(input_df)
        results = []

        # Try to get risk_cluster from input_df if it exists, else fill with None
        risk_clusters = input_df["risk_cluster"].tolist() if "risk_cluster" in input_df.columns else [None] * len(input_df)

        for p, rc in zip(probs, risk_clusters):
            score = 850 - (p * 550)
            if score >= 750:
                level = "Low Risk"
            elif score >= 650:
                level = "Medium Risk"
            elif score >= 550:
                level = "High Risk"
            else:
                level = "Very High Risk"

            results.append({
                "default_probability": float(p),
                "risk_score": int(score),
                "risk_level": level,
                "risk_cluster": int(rc) if rc is not None else None
            })

        return results


# -----------------------------------
# LOCAL TEST
# -----------------------------------

if __name__ == "__main__":
    setup_logging()

    logger = logging.getLogger(__name__)

    service = ModelService("models/credit_model_v1.pkl")

    logger.info("Loading raw dataset for test")

    df = pd.read_csv("data/loan_2019_20.csv", low_memory=False)

    df = prepare_target(df, logger)
    df = clean_dataset(df)

    df = df.drop(columns=["default", "loan_status"])

    df = df.head(5)

    logger.info("Running test inference")

    results = service.predict_with_risk(df)

    print(results)