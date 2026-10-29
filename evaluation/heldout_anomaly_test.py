
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

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

FAILURE_SCENARIOS = [
    "database_failure",
    "network_failure",
    "application_failure",
]

ALL_SCENARIOS = FAILURE_SCENARIOS + ["normal_operation"]


# ---------------------------------------------------------
# 2. Load and label the observations
# ---------------------------------------------------------

def load_scenario(scenario_name):
    """Load payment-service metrics and assign labels from
    the known simulated incident window.
    """
    path = (
        SCENARIO_DIR
        / scenario_name
        / "simulated_observability.csv"
    )

    df = pd.read_csv(path)

    required = ["timestamp", "service", *FEATURES]
    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"{scenario_name}: missing columns {sorted(missing)}"
        )

    df = df[df["service"] == "payment-service"].copy()

    if df.empty:
        raise ValueError(
            f"{scenario_name}: no payment-service observations"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="raise"
    )

    for feature in FEATURES:
        df[feature] = pd.to_numeric(
            df[feature], errors="raise"
        )

    if not np.isfinite(df[FEATURES].to_numpy()).all():
        raise ValueError(
            f"{scenario_name}: metrics contain NaN or infinity"
        )

    # The ground-truth incident window is used only for evaluation
    # labels, not as an input feature to the detector.
    if scenario_name in FAILURE_SCENARIOS:
        df["true_label"] = (
            (df["timestamp"] >= INCIDENT_START)
            & (df["timestamp"] <= INCIDENT_END)
        ).astype(int)
    else:
        df["true_label"] = 0

    df["scenario"] = scenario_name

    return df.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------
# 3. Fit the detector on normal-operation data only
# ---------------------------------------------------------

def train_detector(normal_data):
    """Fit one Isolation Forest using normal-operation data."""
    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
    )

    model.fit(normal_data[FEATURES])

    return model


# ---------------------------------------------------------
# 4. Evaluate both methods on held-out rows
# ---------------------------------------------------------

def calculate_metrics(y_true, y_pred):
    """Return standard binary-classification metrics."""
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "observations": len(y_true),
        "actual_incidents": int(np.sum(y_true)),
        "predicted_incidents": int(np.sum(y_pred)),
        "accuracy": accuracy_score(y_true, y_pred),
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
            fp / (fp + tn) if (fp + tn) else 0.0
        ),
    }


def main():
    # Normal data is reserved exclusively for fitting the model.
    normal_data = load_scenario("normal_operation")
    model = train_detector(normal_data)

    results = []

    for scenario_name in ALL_SCENARIOS:
        df = load_scenario(scenario_name)

        # Evaluate only rows outside the training dataset.
        # For failure scenarios, retain both normal and incident
        # rows. The normal-operation scenario is also evaluated.
        predictions = model.predict(df[FEATURES])
        scores = -model.decision_function(df[FEATURES])

        df["iforest_prediction"] = (
            predictions == -1
        ).astype(int)
        df["anomaly_score"] = scores

        # Simple threshold baseline, independent of Isolation Forest.
        df["threshold_prediction"] = (
            df[list(THRESHOLDS)]
            .gt(pd.Series(THRESHOLDS))
            .any(axis=1)
        ).astype(int)

        results.append(df)

    combined = pd.concat(results, ignore_index=True)

    # Save every observation and both methods' predictions.
    row_output = OUTPUT_DIR / "heldout_anomaly_predictions.csv"
    combined.to_csv(row_output, index=False)

    # Calculate overall and per-scenario metrics.
    summary_rows = []

    for scenario_name in ALL_SCENARIOS + ["ALL_SCENARIOS"]:
        subset = (
            combined
            if scenario_name == "ALL_SCENARIOS"
            else combined[combined["scenario"] == scenario_name]
        )

        for method, prediction_column in [
            ("Isolation Forest", "iforest_prediction"),
            ("Threshold baseline", "threshold_prediction"),
        ]:
            metrics = calculate_metrics(
                subset["true_label"].to_numpy(),
                subset[prediction_column].to_numpy(),
            )

            summary_rows.append({
                "scenario": scenario_name,
                "method": method,
                **metrics,
            })

    summary = pd.DataFrame(summary_rows)

    summary_output = OUTPUT_DIR / "heldout_anomaly_summary.csv"
    summary.to_csv(summary_output, index=False)

    display_columns = [
        "scenario",
        "method",
        "observations",
        "actual_incidents",
        "predicted_incidents",
        "precision",
        "recall",
        "f1_score",
        "false_positive_rate",
    ]

    print("\n=== HELD-OUT ANOMALY EVALUATION ===")
    print(summary[display_columns].to_string(
        index=False,
        formatters={
            "precision": "{:.3f}".format,
            "recall": "{:.3f}".format,
            "f1_score": "{:.3f}".format,
            "false_positive_rate": "{:.3f}".format,
        },
    ))

    print("\nTraining observations:", len(normal_data))
    print("Training source: normal_operation/payment-service")
    print("Training incident labels:", int(normal_data["true_label"].sum()))
    print("Predictions saved to:", row_output)
    print("Summary saved to:", summary_output)


if __name__ == "__main__":
    main()
