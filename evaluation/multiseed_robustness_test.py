
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

TRAINING_SIZES = [20, 50, 100, 150]
MODEL_SEEDS = [11, 22, 33, 44, 55]


def load_normal_data():
    path = (
        SCENARIO_DIR
        / "normal_operation"
        / "simulated_observability.csv"
    )

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[df["service"] == "payment-service"].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    if len(df) < 170:
        raise ValueError(
            f"Expected at least 170 normal rows; found {len(df)}"
        )

    if df[FEATURES].isnull().any().any():
        raise ValueError("Normal metrics contain missing values.")

    if not np.isfinite(df[FEATURES].to_numpy()).all():
        raise ValueError("Normal metrics contain non-finite values.")

    return df


def build_fixed_test_set(normal_data):
    """Use the same 120 test observations for every experiment."""
    normal_test = normal_data.iloc[20:120].copy()
    normal_test["true_label"] = 0
    normal_test["test_pattern"] = "normal"

    incident_test = normal_data.iloc[120:140].copy()
    incident_test = incident_test.reset_index(drop=True)
    incident_test["true_label"] = 1
    incident_test["test_pattern"] = "subthreshold_incident"

    # Reproduce the original controlled incident pattern.
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


def evaluate_predictions(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "false_positive_rate": (
            fp / (fp + tn) if fp + tn else 0.0
        ),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


def main():
    normal_data = load_normal_data()
    test_data = build_fixed_test_set(normal_data)

    run_rows = []
    prediction_rows = []

    total_runs = len(TRAINING_SIZES) * len(MODEL_SEEDS)
    run_number = 0

    for training_size in TRAINING_SIZES:
        train_data = normal_data.iloc[:training_size]

        for seed in MODEL_SEEDS:
            run_number += 1

            model = IsolationForest(
                n_estimators=200,
                contamination=0.05,
                random_state=seed,
            )
            model.fit(train_data[FEATURES])

            result = test_data.copy()
            result["training_size"] = training_size
            result["model_seed"] = seed
            result["iforest_prediction"] = (
                model.predict(result[FEATURES]) == -1
            ).astype(int)
            result["anomaly_score"] = (
                -model.decision_function(result[FEATURES])
            )

            metrics = evaluate_predictions(
                result["true_label"].to_numpy(),
                result["iforest_prediction"].to_numpy(),
            )

            run_rows.append({
                "training_size": training_size,
                "model_seed": seed,
                **metrics,
            })

            prediction_rows.append(result)

            print(
                f"Completed run {run_number}/{total_runs}: "
                f"training_size={training_size}, seed={seed}"
            )

    runs = pd.DataFrame(run_rows)
    predictions = pd.concat(
        prediction_rows, ignore_index=True
    )

    # Summarize variation across model seeds for each training size.
    summary_rows = []

    metric_columns = [
        "precision",
        "recall",
        "f1_score",
        "false_positive_rate",
    ]

    for training_size, group in runs.groupby("training_size"):
        row = {
            "training_size": training_size,
            "runs": len(group),
        }

        for metric in metric_columns:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)

        row["min_f1"] = group["f1_score"].min()
        row["max_f1"] = group["f1_score"].max()

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    runs_path = OUTPUT_DIR / "multiseed_robustness_runs.csv"
    summary_path = OUTPUT_DIR / "multiseed_robustness_summary.csv"
    predictions_path = (
        OUTPUT_DIR / "multiseed_robustness_predictions.csv"
    )

    runs.to_csv(runs_path, index=False)
    summary.to_csv(summary_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    print("\n=== MULTI-SEED ROBUSTNESS SUMMARY ===")
    print("Training sizes:", TRAINING_SIZES)
    print("Model seeds:", MODEL_SEEDS)
    print("Fixed test observations:", len(test_data))
    print("\nMean and standard deviation across seeds:")
    print(summary.round(3).to_string(index=False))

    print("\nIndividual runs saved to:", runs_path)
    print("Summary saved to:", summary_path)
    print("Predictions saved to:", predictions_path)


if __name__ == "__main__":
    main()
