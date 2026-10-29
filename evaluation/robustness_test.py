
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
from agents.anomaly_agent import AnomalyAgent

SCENARIOS_DIR = PROJECT_ROOT / "data" / "scenarios"
GROUND_TRUTH_FILE = PROJECT_ROOT / "evaluation" / "ground_truth.json"
RESULTS_FILE = PROJECT_ROOT / "evaluation" / "robustness_results.csv"

SEEDS = [101, 202, 303]
NOISE_LEVELS = [0.0, 0.5, 1.0, 2.0]

NO_INCIDENT = "No incident"

NOISE_SCALES = {
    "cpu": 3.0,
    "memory": 3.0,
    "latency": 15.0,
    "error_rate": 0.5,
    "request_rate": 10.0,
}

REQUIRED_COLUMNS = {
    "service",
    *NOISE_SCALES.keys(),
}


def normalize_label(label):
    """Normalize labels so equivalent no-incident values match."""
    if label is None:
        return NO_INCIDENT

    text = str(label).strip()

    if text.lower() in {
        "no incident",
        "no_incident",
        "normal operation",
        "normal_operation",
        "none",
    }:
        return NO_INCIDENT

    return text


def make_metric_variation(original, seed, noise_level):
    """Create a reproducible noisy copy of payment-service metrics."""
    varied = original.copy()

    if noise_level == 0:
        return varied

    rng = np.random.default_rng(seed)
    mask = varied["service"] == "payment-service"

    for column, base_scale in NOISE_SCALES.items():
        values = varied.loc[mask, column].to_numpy(dtype=float)

        noise = rng.normal(
            loc=0.0,
            scale=base_scale * noise_level,
            size=len(values),
        )

        varied.loc[mask, column] = np.maximum(values + noise, 0)

    return varied


def measure_anomaly_detector(metrics, target_service):
    """
    Measure the anomaly detector independently of the RCA investigation.

    This uses the same detector and actionability thresholds as AnomalyAgent.
    It does not modify the production agent or main.py.
    """
    agent = AnomalyAgent()

    service_data = metrics.loc[
        metrics["service"] == target_service
    ].copy()

    if service_data.empty:
        return {
            "statistical_outlier_count": 0,
            "actionable_anomaly_count": 0,
            "measured_anomaly_score": 0.0,
            "measured_severity": "NONE",
            "measured_abnormal_metrics": "",
        }

    for column in agent.FEATURE_COLUMNS:
        service_data[column] = pd.to_numeric(
            service_data[column],
            errors="coerce",
        )

    features = service_data[agent.FEATURE_COLUMNS].to_numpy(dtype=float)

    if not np.isfinite(features).all():
        raise ValueError(
            f"Non-finite metric values found for {target_service}."
        )

    # Fit and score the same observations, matching the current agent.
    agent.detector.train(service_data)
    detected = agent.detector.detect(service_data)

    outliers = detected.loc[
        detected["anomaly_label"] == -1
    ].copy()

    actionable = []

    for _, row in outliers.iterrows():
        abnormal_metrics = agent._identify_abnormal_metrics(row)

        if abnormal_metrics:
            actionable.append((row, abnormal_metrics))

    if not actionable:
        return {
            "statistical_outlier_count": len(outliers),
            "actionable_anomaly_count": 0,
            "measured_anomaly_score": 0.0,
            "measured_severity": "NONE",
            "measured_abnormal_metrics": "",
        }

    selected_row, abnormal_metrics = max(
        actionable,
        key=lambda item: float(item[0]["anomaly_score"]),
    )

    return {
        "statistical_outlier_count": len(outliers),
        "actionable_anomaly_count": len(actionable),
        "measured_anomaly_score": float(
            selected_row["anomaly_score"]
        ),
        "measured_severity": agent._calculate_severity(
            abnormal_metrics
        ),
        "measured_abnormal_metrics": ",".join(abnormal_metrics),
    }


def run_one_test(scenario, seed, noise_level, reference):
    metrics_path = (
        SCENARIOS_DIR / scenario / "simulated_observability.csv"
    )

    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"Metrics file not found: {metrics_path}"
        )

    # Preserve the exact original bytes for reliable restoration.
    original_bytes = metrics_path.read_bytes()

    try:
        original_metrics = pd.read_csv(metrics_path)

        missing_columns = REQUIRED_COLUMNS - set(original_metrics.columns)

        if missing_columns:
            raise ValueError(
                f"{scenario}: missing required columns "
                f"{sorted(missing_columns)}"
            )

        varied_metrics = make_metric_variation(
            original_metrics,
            seed,
            noise_level,
        )

        # Measure the detector before running the complete RCA pipeline.
        detector_metrics = measure_anomaly_detector(
            varied_metrics,
            reference.get("affected_service", "payment-service"),
        )

        # The existing main() workflow reads the metrics CSV from disk.
        varied_metrics.to_csv(metrics_path, index=False)

        captured_output = io.StringIO()

        with contextlib.redirect_stdout(captured_output):
            result = main({
                "scenario": scenario,
                "incident_id": reference["incident_id"],
                "affected_service": reference.get(
                    "affected_service",
                    "payment-service",
                ),
            })

        expected = normalize_label(reference["ground_truth"])

        anomaly_detected = bool(
            result.get("anomaly_detected", False)
        )

        if not anomaly_detected:
            predicted = NO_INCIDENT
        else:
            predicted = normalize_label(result.get("root_cause"))

        return {
            "scenario": scenario,
            "seed": seed,
            "noise_level": noise_level,
            "expected": expected,
            "predicted": predicted,
            "correct": predicted == expected,
            "anomaly_detected": anomaly_detected,
            "report_status": result.get(
                "report_status", "UNKNOWN"
            ),
            "confidence": float(result.get("confidence") or 0.0),
            "investigation_operations": int(
                result.get("tool_calls", 0)
            ),
            **detector_metrics,
        }

    finally:
        # Restore original data even if evaluation raises an exception.
        metrics_path.write_bytes(original_bytes)


def print_accuracy(label, rows):
    """Print accuracy with its numerator and denominator."""
    if rows.empty:
        print(f"{label}: No cases")
        return

    correct_count = int(rows["correct"].sum())
    total_count = len(rows)
    accuracy = rows["correct"].mean() * 100

    print(
        f"{label}: {accuracy:.1f}% "
        f"({correct_count}/{total_count})"
    )


def main_test():
    with GROUND_TRUTH_FILE.open("r", encoding="utf-8") as file:
        ground_truth = json.load(file)

    rows = []

    print("\n========== RCA ROBUSTNESS EVALUATION ==========")
    print(f"Scenarios: {len(ground_truth)}")
    print(f"Seeds: {SEEDS}")
    print(f"Noise levels: {NOISE_LEVELS}")

    planned_runs = (
        len(ground_truth) * len(SEEDS) * len(NOISE_LEVELS)
    )
    print(f"Planned test runs: {planned_runs}")

    for scenario, reference in ground_truth.items():
        for noise_level in NOISE_LEVELS:
            for seed in SEEDS:
                print(
                    f"\nTesting {scenario}, noise={noise_level}, "
                    f"seed={seed} ..."
                )

                row = run_one_test(
                    scenario,
                    seed,
                    noise_level,
                    reference,
                )
                rows.append(row)

                status = "CORRECT" if row["correct"] else "INCORRECT"

                print(
                    f"  Prediction: {row['predicted']} | {status}"
                )
                print(
                    f"  Statistical outliers: "
                    f"{row['statistical_outlier_count']} | "
                    f"Actionable anomalies: "
                    f"{row['actionable_anomaly_count']}"
                )
                print(
                    f"  Detector score: "
                    f"{row['measured_anomaly_score']:.6f} | "
                    f"Severity: {row['measured_severity']} | "
                    f"RCA confidence: {row['confidence']:.3f}"
                )

    results = pd.DataFrame(rows)

    # Compare each noisy result with the same scenario and seed baseline.
    baseline = results.loc[
        results["noise_level"] == 0.0,
        [
            "scenario",
            "seed",
            "predicted",
            "anomaly_detected",
            "statistical_outlier_count",
            "actionable_anomaly_count",
            "measured_anomaly_score",
            "measured_severity",
            "confidence",
        ],
    ].copy()

    baseline = baseline.rename(columns={
        "predicted": "baseline_prediction",
        "anomaly_detected": "baseline_anomaly_detected",
        "statistical_outlier_count": "baseline_statistical_outliers",
        "actionable_anomaly_count": "baseline_actionable_anomalies",
        "measured_anomaly_score": "baseline_anomaly_score",
        "measured_severity": "baseline_severity",
        "confidence": "baseline_confidence",
    })

    results = results.merge(
        baseline,
        on=["scenario", "seed"],
        how="left",
        validate="many_to_one",
    )

    results["prediction_changed_from_baseline"] = (
        results["predicted"] != results["baseline_prediction"]
    )

    results["anomaly_decision_changed_from_baseline"] = (
        results["anomaly_detected"]
        != results["baseline_anomaly_detected"]
    )

    results["outlier_count_change"] = (
        results["statistical_outlier_count"]
        - results["baseline_statistical_outliers"]
    )

    results["actionable_count_change"] = (
        results["actionable_anomaly_count"]
        - results["baseline_actionable_anomalies"]
    )

    results["anomaly_score_change"] = (
        results["measured_anomaly_score"]
        - results["baseline_anomaly_score"]
    )

    results["confidence_change"] = (
        results["confidence"] - results["baseline_confidence"]
    )

    results.to_csv(RESULTS_FILE, index=False)

    print("\n========== OVERALL SUMMARY ==========")
    print(f"Completed runs: {len(results)}")
    print_accuracy("Overall accuracy", results)

    incident_rows = results[results["expected"] != NO_INCIDENT]
    print_accuracy("Incident-only accuracy", incident_rows)

    normal_rows = results[results["expected"] == NO_INCIDENT]

    if not normal_rows.empty:
        false_positive_rate = (
            normal_rows["anomaly_detected"].mean() * 100
        )
        print(
            "Normal-operation false-positive rate: "
            f"{false_positive_rate:.1f}%"
        )

    print("\n========== DETECTOR BEHAVIOUR BY NOISE LEVEL ==========")

    grouped = results.groupby("noise_level").agg(
        runs=("correct", "size"),
        RCA_accuracy=("correct", "mean"),
        mean_statistical_outliers=("statistical_outlier_count", "mean"),
        mean_actionable_anomalies=("actionable_anomaly_count", "mean"),
        mean_anomaly_score=("measured_anomaly_score", "mean"),
        mean_confidence=("confidence", "mean"),
        changed_predictions=("prediction_changed_from_baseline", "sum"),
        changed_anomaly_decisions=(
            "anomaly_decision_changed_from_baseline",
            "sum",
        ),
    )

    print(grouped.round(3).to_string())

    print("\n========== RESULTS BY SCENARIO ==========")

    for scenario in ground_truth:
        subset = results[results["scenario"] == scenario]
        print_accuracy(scenario, subset)

    print(f"\nDetailed results saved to: {RESULTS_FILE}")
    print("Original scenario metrics have been restored.")


if __name__ == "__main__":
    main_test()
