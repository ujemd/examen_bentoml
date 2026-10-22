"""Train a linear regression on the admission data, evaluate it on the test
set and save it in the BentoML model store as `admission_lr`."""
from pathlib import Path

import bentoml
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_NAME = "admission_lr"


def load_data():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze("columns")
    return X_train, X_test, y_train, y_test


def main() -> None:
    X_train, X_test, y_train, y_test = load_data()

    # Fitted on a DataFrame, so the model remembers the column names; the
    # service has to send a DataFrame with the same columns.
    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "r2": float(r2_score(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "mae": float(mean_absolute_error(y_test, predictions)),
    }
    print("Test performance:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    saved = bentoml.sklearn.save_model(
        MODEL_NAME,
        model,
        signatures={"predict": {"batchable": True}},
        labels={"project": "admission_prediction"},
        metadata={"features": list(X_train.columns), **metrics},
    )
    print("Saved model:", saved)


if __name__ == "__main__":
    main()