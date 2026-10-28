from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
TARGET = "Yield"

FEATURES = [
    "Crop",
    "Crop_Year",
    "Season",
    "State",
    "Area",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide",
]
CATEGORICAL_FEATURES = ["Crop", "Season", "State"]


def main():
    csv_files = list(DATA_DIR.glob("*.csv"))

    if len(csv_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one CSV file in {DATA_DIR.resolve()}, "
            f"but found {len(csv_files)}."
        )

    df = pd.read_csv(csv_files[0])

    required_columns = FEATURES + [TARGET, "Production"]
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("Original dataset shape:", df.shape)

    # Convert numeric columns and identify invalid values.
    numeric_columns = [
        "Crop_Year",
        "Area",
        "Production",
        "Annual_Rainfall",
        "Fertilizer",
        "Pesticide",
        TARGET,
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in CATEGORICAL_FEATURES:
        df[column] = df[column].astype("string").str.strip()
        df[column] = df[column].replace("", pd.NA)

    df = df.dropna(subset=FEATURES + [TARGET]).copy()

    # Reject infinite numeric values.
    df = df[
        df[numeric_columns].apply(
            lambda column: column.map(
                lambda value: pd.notna(value)
                and abs(value) != float("inf")
            )
        ).all(axis=1)
    ].copy()

    df = df[
        (df["Area"] > 0)
        & (df["Production"] >= 0)
        & (df["Annual_Rainfall"] >= 0)
        & (df["Fertilizer"] >= 0)
        & (df["Pesticide"] >= 0)
    ].copy()

    # Keep high values for review because crop yield units can vary.
    df = df[df[TARGET] >= 0].copy()
    df = df.drop_duplicates(subset=FEATURES + [TARGET])

    print("Rows after basic validation:", len(df))
    print("\nYield summary:")
    print(df[TARGET].describe())

    print("\nYield maximum by crop:")
    print(df.groupby("Crop")[TARGET].max().sort_values(ascending=False).head(10))

    print("\nLargest 10 yield values:")
    print(df.nlargest(10, TARGET)[
        ["Crop", "State", "Crop_Year", "Area", "Production", TARGET]
    ].to_string(index=False))

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_data = X_train.copy()
    train_data[TARGET] = y_train

    test_data = X_test.copy()
    test_data[TARGET] = y_test

    train_path = OUTPUT_DIR / "yield_train.csv"
    test_path = OUTPUT_DIR / "yield_test.csv"

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    print("\nData preparation completed.")
    print("Training rows:", len(train_data))
    print("Testing rows:", len(test_data))
    print("Features:", FEATURES)
    print("Target:", TARGET)
    print("Saved training data to:", train_path)
    print("Saved testing data to:", test_path)


if __name__ == "__main__":
    main()
