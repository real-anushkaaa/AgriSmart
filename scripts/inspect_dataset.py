
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw")


def main():
    csv_files = list(DATA_DIR.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {DATA_DIR.resolve()}")
        print("Download the dataset and place its CSV file in data/raw/.")
        return

    print("CSV files found:")
    for index, file_path in enumerate(csv_files, start=1):
        print(f"{index}. {file_path.name}")

    if len(csv_files) > 1:
        print("\nMultiple CSV files found.")
        print("Select the correct training dataset before proceeding.")
        return

    file_path = csv_files[0]
    df = pd.read_csv(file_path)

    print("\nDataset:", file_path.name)
    print("Shape:", df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst five rows:")
    print(df.head().to_string())

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nMissing values per column:")
    print(df.isnull().sum().to_string())

    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nNumeric column summary:")
    print(df.describe().to_string())


if __name__ == "__main__":
    main()
