
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import main

SCENARIOS_DIR = PROJECT_ROOT / "data" / "scenarios"
GROUND_TRUTH_FILE = PROJECT_ROOT / "evaluation" / "ground_truth.json"
RESULTS_FILE = PROJECT_ROOT / "evaluation" / "robustness_results.csv"

SEEDS = [101, 202, 303]
NO_INCIDENT = "No incident"


def normalize_label(label):
    if label is None:
        return NO_INCIDENT

    text = str(label).strip()

    if text.lower() in {
        "no incident", "no_incident",
        "normal operation", "normal_operation", "none"
    }:
        return NO_INCIDENT

    return text


def make_metric_variation(original, seed):
    """Create a reproducible noisy copy without changing service labels."""
    varied = original.copy()
    rng = np.random.default_rng(seed)

    # Vary only the affected service's observations.
    mask = varied["service"] == "payment-service"

    noise_scales = {
        "cpu": 3.0,
        "memory": 3.0,
        "latency": 15.0,
        "error_rate": 0.5,
        "request_rate": 10.0,
    }

    for column, scale in noise_scales.items():
        if column in varied.columns:
            noise = rng.normal(0, scale, int(mask.sum()))
            varied.loc[mask, column] = (
                varied.loc[mask, column].to_numpy() + noise
            ).clip(min=0)

    return varied


def run_one_test(scenario, seed, reference):
    metrics_path = (
        SCENARIOS_DIR / scenario / "simulated_observability.csv"
    )

    # Keep the exact original file bytes for reliable restoration.
    original_bytes = metrics_path.read_bytes()

    try:
        original_metrics = pd.read_csv(metrics_path)
        varied_metrics = make_metric_variation(original_metrics, seed)

        # Temporarily expose the varied metrics to the existing pipeline.
        varied_metrics.to_csv(metrics_path, index=False)

        captured_output = io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            result = main({
                "scenario": scenario,
                "incident_id": reference["incident_id"],
                "affected_service": reference.get(
                    "affected_service", "payment-service"
                ),
            })

        expected = normalize_label(reference["ground_truth"])

        if not result.get("anomaly_detected", False):
            predicted = NO_INCIDENT
        else:
            predicted = normalize_label(result.get("root_cause"))

        return {
            "scenario": scenario,
            "seed": seed,
            "expected": expected,
            "predicted": predicted,
            "correct": predicted == expected,
            "anomaly_detected": bool(
                result.get("anomaly_detected", False)
            ),
            "report_status": result.get("report_status", "UNKNOWN"),
            "confidence": float(result.get("confidence") or 0.0),
            "investigation_operations": int(
                result.get("tool_calls", 0)
            ),
        }

    finally:
        # Restore the original scenario file, even if main() fails.
        metrics_path.write_bytes(original_bytes)


def main_test():
    with GROUND_TRUTH_FILE.open("r", encoding="utf-8") as file:
        ground_truth = json.load(file)

    rows = []

    print("\n========== RCA ROBUSTNESS EVALUATION ==========")

    for scenario, reference in ground_truth.items():
        for seed in SEEDS:
            print(f"Testing {scenario}, seed={seed} ...")
            row = run_one_test(scenario, seed, reference)
            rows.append(row)

            status = "CORRECT" if row["correct"] else "INCORRECT"
            print(
                f"  Expected: {row['expected']} | "
                f"Predicted: {row['predicted']} | {status}"
            )

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_FILE, index=False)

    print("\n========== SUMMARY ==========")
    print(f"Test runs: {len(results)}")
    print(f"Accuracy: {results['correct'].mean() * 100:.1f}%")

    incident_rows = results[results["expected"] != NO_INCIDENT]
    if not incident_rows.empty:
        print(
            "Incident-only accuracy: "
            f"{incident_rows['correct'].mean() * 100:.1f}%"
        )

    normal_rows = results[results["expected"] == NO_INCIDENT]
    if not normal_rows.empty:
        false_positive_rate = (
            normal_rows["anomaly_detected"].mean() * 100
        )
        print(
            "Normal-operation false-positive rate: "
            f"{false_positive_rate:.1f}%"
        )

    print(f"Detailed results: {RESULTS_FILE}")
    print("Original scenario metrics have been restored.")


if __name__ == "__main__":
    main_test()
