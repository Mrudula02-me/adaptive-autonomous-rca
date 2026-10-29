
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

INCIDENT_START = pd.Timestamp("2026-10-01 10:07:00")
INCIDENT_END = pd.Timestamp("2026-10-01 10:12:00")


def load_payment_metrics(scenario):
    """Load payment-service metrics from one scenario."""
    path = (
        SCENARIO_DIR
        / scenario
        / "simulated_observability.csv"
    )

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df[df["service"] == "payment-service"].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No payment-service data in {path}")

    if df[FEATURES].isnull().any().any():
        raise ValueError(f"Missing metric values in {path}")

    if not np.isfinite(df[FEATURES].to_numpy()).all():
        raise ValueError(f"Non-finite metric values in {path}")

    return df


def make_subthreshold_test_data():
    """
    Construct a controlled test set from normal observations.

    The injected patterns have elevated latency and error rates,
    but remain below the existing fixed thresholds.
    """
    source = load_payment_metrics("normal_operation")

    # Use a reproducible selection of normal rows as a baseline.
    normal = source.iloc[20:120].copy()
    normal["true_label"] = 0
    normal["test_pattern"] = "normal"

    # Use separate normal rows as the basis for controlled incidents.
    incident = source.iloc[120:140].copy().reset_index(drop=True)
    incident["true_label"] = 1
    incident["test_pattern"] = "subthreshold_incident"

    # Three levels of elevated latency and error rate.
    # All injected values remain below the fixed thresholds.
    incident["latency"] = np.linspace(220, 420, len(incident))
    incident["error_rate"] = np.linspace(1.0, 4.5, len(incident))

    # Keep CPU, memory, and request rate close to the source values.
    # This experiment isolates whether the detector recognizes the
    # joint latency/error pattern without threshold crossings.
    test_data = pd.concat(
        [normal, incident],
        ignore_index=True,
    )

    # Confirm the injected incident rows do not cross any threshold.
    for column, threshold in THRESHOLDS.items():
        if (incident[column] > threshold).any():
            raise ValueError(
                f"Injected incident crosses threshold: {column}"
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
    # Train only on normal observations from the normal-operation
    # scenario. These rows are not used in the test set.
    source = load_payment_metrics("normal_operation")
    train_data = source.iloc[:20].copy()

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
    )
    model.fit(train_data[FEATURES])

    test_data = make_subthreshold_test_data()

    # Isolation Forest predictions: -1 means anomaly, 1 means normal.
    test_data["iforest_prediction"] = (
        model.predict(test_data[FEATURES]) == -1
    ).astype(int)

    test_data["anomaly_score"] = (
        -model.decision_function(test_data[FEATURES])
    )

    # Existing fixed-threshold baseline.
    test_data["threshold_prediction"] = (
        test_data[list(THRESHOLDS)]
        .gt(pd.Series(THRESHOLDS))
        .any(axis=1)
    ).astype(int)

    rows = []

    for method, column in [
        ("Isolation Forest", "iforest_prediction"),
        ("Threshold baseline", "threshold_prediction"),
    ]:
        rows.append({
            "method": method,
            **calculate_metrics(
                test_data["true_label"].to_numpy(),
                test_data[column].to_numpy(),
            ),
        })

    summary = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions_path = (
        OUTPUT_DIR / "subthreshold_anomaly_predictions.csv"
    )
    summary_path = (
        OUTPUT_DIR / "subthreshold_anomaly_summary.csv"
    )

    test_data.to_csv(predictions_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\n=== SUB-THRESHOLD ANOMALY EXPERIMENT ===")
    print("Training observations:", len(train_data))
    print("Test observations:", len(test_data))
    print("Known incident observations:", int(test_data["true_label"].sum()))
    print(
        "Incident rows crossing any fixed threshold:",
        int(
            test_data.loc[
                test_data["true_label"] == 1,
                list(THRESHOLDS),
            ].gt(pd.Series(THRESHOLDS)).any(axis=1).sum()
        ),
    )
    print("\nResults:")
    print(summary.round(3).to_string(index=False))
    print("\nPredictions saved to:", predictions_path)
    print("Summary saved to:", summary_path)


if __name__ == "__main__":
    main()
