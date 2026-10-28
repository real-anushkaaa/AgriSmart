
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_DIR = Path("data/processed")
ARTIFACT_DIR = Path("artifacts")
TARGET = "Yield"

CATEGORICAL_FEATURES = [
    "Crop",
    "Season",
    "State",
]

NUMERICAL_FEATURES = [
    "Crop_Year",
    "Area",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide",
]

FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


def main():
    train_path = DATA_DIR / "yield_train.csv"
    test_path = DATA_DIR / "yield_test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Training or testing data is missing. "
            "Run scripts/prepare_yield_data.py first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    for frame in (train_df, test_df):
        for column in CATEGORICAL_FEATURES:
            frame[column] = frame[column].astype("string").str.strip()

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("numerical", numerical_pipeline, NUMERICAL_FEATURES),
        ]
    )

    regressor = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )
    # Log scaling limits the influence of very large crop-specific values.
    model = TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
    )

    print("Training Random Forest model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    median_ae = median_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print("\nModel evaluation results")
    print("------------------------")
    print(f"MAE:  {mae:.4f}")
    print(f"Median AE: {median_ae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACT_DIR / "crop_yield_model.joblib"
    joblib.dump(model, model_path)

    results = test_df[FEATURES + [TARGET]].copy()
    results["Predicted_Yield"] = predictions
    results.to_csv(
        ARTIFACT_DIR / "yield_test_predictions.csv",
        index=False,
    )

    per_crop_metrics = {}
    for crop_name, group in results.groupby("Crop"):
        per_crop_metrics[str(crop_name)] = {
            "support": int(len(group)),
            "mae": float(
                mean_absolute_error(group[TARGET], group["Predicted_Yield"])
            ),
            "median_absolute_error": float(
                median_absolute_error(group[TARGET], group["Predicted_Yield"])
            ),
        }
    metrics_path = ARTIFACT_DIR / "yield_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "mae": float(mae),
                "median_absolute_error": float(median_ae),
                "rmse": float(rmse),
                "r2": float(r2),
                "per_crop": per_crop_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nSaved trained model to: {model_path}")
    print(
        "Saved test predictions to: "
        "artifacts/yield_test_predictions.csv"
    )
    print(f"Saved per-crop evaluation metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
