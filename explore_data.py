import logging
import pandas as pd


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


def inspect_dataset(df, logger):

    logger.info("Listing columns")
    logger.info(df.columns.tolist())

    logger.info("Preview rows")
    print(df.head())

    logger.info("Column data types")
    print(df.dtypes)

    logger.info("Top 20 columns with missing values")
    print(
        df.isnull()
        .sum()
        .sort_values(ascending=False)
        .head(20)
    )


def inspect_target(df, logger):

    logger.info("Loan status distribution")

    print(df["loan_status"].value_counts())


def prepare_target(df, logger):

    logger.info("Filtering final loan outcomes")

    df = df[
        df["loan_status"].isin(
            ["Fully Paid", "Charged Off", "Default"]
        )
    ].copy()

    logger.info(f"Dataset size after filtering: {df.shape}")

    df["default"] = df["loan_status"].apply(
        lambda x: 1 if x in ["Charged Off", "Default"] else 0
    )

    logger.info("Target distribution")
    print(df["default"].value_counts())

    return df

def remove_leakage_columns(df, logger):

    logger.info("Removing leakage columns")

    leakage_cols = [
        "Unnamed: 0",
        "id",
        "url",

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
        "next_pymnt_d"
    ]

    existing_cols = [c for c in leakage_cols if c in df.columns]

    df = df.drop(columns=existing_cols)

    logger.info(f"Columns removed: {len(existing_cols)}")
    logger.info(f"Remaining columns: {len(df.columns)}")

    return df


def main():

    logger = logging.getLogger(__name__)

    df = load_dataset(logger)

    inspect_dataset(df, logger)

    inspect_target(df, logger)

    df = prepare_target(df, logger)

    df = remove_leakage_columns(df, logger)


if __name__ == "__main__":
    setup_logging()
    main()