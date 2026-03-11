import pandas as pd
import numpy as np
import sklearn


def main():
    print("Environment check successful.")
    print(f"Pandas version: {pd.__version__}")
    print(f"Numpy version: {np.__version__}")
    print(f"Scikit-learn version: {sklearn.__version__}")


if __name__ == "__main__":
    main()