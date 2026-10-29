
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = ROOT / "data" / "scenarios"
OUTPUT_DIR = ROOT / "evaluation"

FEATURES = [
    "cpu",
    "memory",
    "latency",
    "error_rate",
    "request_rate",
]

THRESHOLDS = {
    "cpu": 80,
    "memory": 85,
    "latency": 500,
    "error_rate": 5,
}

TRAINING_SIZES = [20, 50, 100, 150]


def load_normal_data():
    """Load normal-operation payment-service metrics."""
    path = (
        SCENARIO_DIR
        / "normal_operation"
        / "simulated_observability.csv"
    )

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[df["service"] == "payment-service"].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    if len(df) < max(TRAINING_SIZES):
        raise ValueError(
            f"Need at least {max(TRAINING_SIZES)} normal rows; "
            f"found {len(df)}."
        )

    if df[FEATURES].isnull().any().any():
        raise ValueError("Normal dataset contains missing features.")

    if not np.isfinite(df[FEATURES].to_numpy()).all():
        raise ValueError("Normal dataset contains non-finite features.")

    return df


def build_fixed_test_set(normal_data):
    """
    Reproduce the original sub-threshold test set exactly:
    100 normal observations and 20 controlled incident observations.
    """
    normal_test = normal_data.iloc[20:120].copy()
    normal_test["true_label"] = 0
    normal_test["test_pattern"] = "normal"

    incident_test = normal_data.iloc[120:140].copy()
    incident_test = incident_test.reset_index(drop=True)
    incident_test["true_label"] = 1
    incident_test["test_pattern"] = "subthreshold_incident"

    # Same injected pattern as the original experiment.
    incident_test["latency"] = np.linspace(
        220, 420, len(incident_test)
    )
    incident_test["error_rate"] = np.linspace(
        1.0, 4.5, len(incident_test)
    )

    test_data = pd.concat(
        [normal_test, incident_test],
        ignore_index=True,
    )

    return test_data


def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "observations": len(y_true),
        "actual_incidents": int(np.sum(y_true)),
        "predicted_incidents": int(np.sum(y_pred)),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "false_positive_rate": (
            fp / (fp + tn) if fp + tn else 0.0
        ),
    }


def main():
    normal_data = load_normal_data()
    test_data = build_fixed_test_set(normal_data)

    # Fit one model per training size, keeping test rows fixed.
    summaries = []
    predictions = []

    for training_size in TRAINING_SIZES:
        train_data = normal_data.iloc[:training_size]

        model = IsolationForest(
            n_estimators=200,
            contamination=0.05,
            random_state=42,
        )
        model.fit(train_data[FEATURES])

        result = test_data.copy()
        result["training_size"] = training_size
        result["iforest_prediction"] = (
            model.predict(result[FEATURES]) == -1
        ).astype(int)
        result["anomaly_score"] = (
            -model.decision_function(result[FEATURES])
        )

        metrics = calculate_metrics(
            result["true_label"].to_numpy(),
            result["iforest_prediction"].to_numpy(),
        )

        summaries.append({
            "training_size": training_size,
            **metrics,
        })
        predictions.append(result)

    summary = pd.DataFrame(summaries)
    all_predictions = pd.concat(
        predictions, ignore_index=True
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = (
        OUTPUT_DIR / "training_size_summary.csv"
    )
    predictions_path = (
        OUTPUT_DIR / "training_size_predictions.csv"
    )

    summary.to_csv(summary_path, index=False)
    all_predictions.to_csv(predictions_path, index=False)

    display_columns = [
        "training_size",
        "observations",
        "actual_incidents",
        "predicted_incidents",
        "precision",
        "recall",
        "f1_score",
        "false_positive_rate",
    ]

    print("\n=== TRAINING-SIZE EXPERIMENT ===")
    print("Fixed test set: 100 normal + 20 sub-threshold incidents")
    print(summary[display_columns].round(3).to_string(index=False))

    print("\nSummary saved to:", summary_path)
    print("Predictions saved to:", predictions_path)


if __name__ == "__main__":
    main()
