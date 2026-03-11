import logging
import pandas as pd
import numpy as np
import sklearn


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def main():
    logger = logging.getLogger(__name__)

    logger.info("Environment check successful")

    logger.info(f"Pandas version: {pd.__version__}")
    logger.info(f"Numpy version: {np.__version__}")
    logger.info(f"Scikit-learn version: {sklearn.__version__}")


if __name__ == "__main__":
    setup_logging()
    main()