
import contextlib
import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_DIR = ROOT / "data" / "scenarios"
OUTPUT_DIR = ROOT / "evaluation"
GROUND_TRUTH_FILE = OUTPUT_DIR / "ground_truth.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import main as run_rca


FAILURE_SCENARIOS = {
    "database_failure": "Database-side failure",
    "network_failure": "Network or dependency communication failure",
    "application_failure": "Payment-service internal failure",
}

FEATURES = [
    "cpu",
    "memory",
    "latency",
    "error_rate",
    "request_rate",
]

# Test severity is defined by scaling the original incident's
# deviation from the normal baseline, not by changing labels.
SEVERITY_LEVELS = {
    "moderate": 0.40,
    "strong": 0.70,
    "original": 1.00,
}

# Keep the original incident start, but vary its duration.
DURATIONS = [2, 4, 6]

INCIDENT_START = pd.Timestamp("2026-10-01 10:07:00")


def load_ground_truth():
    with GROUND_TRUTH_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def create_variant(original_df, scenario, severity_factor, duration):
    """Create a temporary metrics variant without changing labels."""
    df = original_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    baseline = original_df.copy()
    baseline["timestamp"] = pd.to_datetime(baseline["timestamp"])

    service_mask = df["service"] == "payment-service"
    incident_end = INCIDENT_START + pd.Timedelta(minutes=duration - 1)

    incident_mask = (
        service_mask
        & (df["timestamp"] >= INCIDENT_START)
        & (df["timestamp"] <= incident_end)
    )

    # Obtain the original generated incident pattern as a reference.
    original_incident_mask = (
        service_mask
        & (baseline["timestamp"] >= INCIDENT_START)
        & (baseline["timestamp"] <= pd.Timestamp("2026-10-01 10:12:00"))
    )

    # Use observations immediately before the incident as the baseline.
    normal_mask = (
        service_mask
        & (baseline["timestamp"] >= INCIDENT_START - pd.Timedelta(minutes=5))
        & (baseline["timestamp"] < INCIDENT_START)
    )

    normal_rows = baseline.loc[normal_mask, FEATURES]
    if normal_rows.empty:
        raise ValueError("No pre-incident baseline observations found.")

    normal_values = normal_rows.median()

    reference_rows = baseline.loc[original_incident_mask, FEATURES]
    if reference_rows.empty:
        raise ValueError(f"No original incident rows for {scenario}.")

    incident_values = reference_rows.mean()

    # Scale each metric's deviation from the normal baseline.
    # This preserves direction while creating weaker variants.
    varied_values = (
        normal_values
        + severity_factor * (incident_values - normal_values)
    )

    for feature in FEATURES:
        df.loc[incident_mask, feature] = varied_values[feature]

    return df


def evaluate_one(scenario, severity_name, severity_factor, duration, truth):
    """Run one RCA case and restore the source file afterward."""
    metrics_path = (
        SCENARIOS_DIR / scenario / "simulated_observability.csv"
    )

    backup_path = metrics_path.with_suffix(".csv.rca-test-backup")

    if backup_path.exists():
        raise RuntimeError(
            f"Backup already exists: {backup_path}. "
            "Check whether a previous run was interrupted."
        )

    shutil.copy2(metrics_path, backup_path)

    try:
        original_df = pd.read_csv(backup_path)

        variant_df = create_variant(
            original_df,
            scenario,
            severity_factor,
            duration,
        )

        variant_df.to_csv(metrics_path, index=False)

        incident_id = (
            f"ROBUST-{scenario}-{severity_name}-{duration}m"
        )

        with contextlib.redirect_stdout(io.StringIO()):
            result = run_rca({
                "scenario": scenario,
                "incident_id": incident_id,
                "affected_service": "payment-service",
                "timestamp": INCIDENT_START
                + pd.Timedelta(minutes=max(duration - 1, 0) // 2),
            })

        predicted_cause = result.get("root_cause")
        anomaly_detected = bool(result.get("anomaly_detected", False))

        return {
            "scenario": scenario,
            "severity": severity_name,
            "severity_factor": severity_factor,
            "duration_minutes": duration,
            "expected_root_cause": truth[scenario]["ground_truth"],
            "predicted_root_cause": (
                predicted_cause if anomaly_detected else "No incident"
            ),
            "anomaly_detected": anomaly_detected,
            "root_cause_correct": (
                anomaly_detected
                and predicted_cause == truth[scenario]["ground_truth"]
            ),
            "report_status": result.get("report_status", "UNKNOWN"),
            "confidence": float(result.get("confidence") or 0.0),
            "investigation_operations": int(result.get("tool_calls", 0)),
            "agents_called": ", ".join(result.get("agents_called", [])),
        }

    finally:
        # Always restore the original metrics, including when the
        # RCA workflow raises an exception.
        shutil.copy2(backup_path, metrics_path)
        backup_path.unlink(missing_ok=True)


def main():
    truth = load_ground_truth()
    results = []

    total = (
        len(FAILURE_SCENARIOS)
        * len(SEVERITY_LEVELS)
        * len(DURATIONS)
    )
    completed = 0

    for scenario in FAILURE_SCENARIOS:
        for severity_name, severity_factor in SEVERITY_LEVELS.items():
            for duration in DURATIONS:
                completed += 1
                print(
                    f"Running {completed}/{total}: "
                    f"{scenario}, {severity_name}, {duration} minutes"
                )

                try:
                    result = evaluate_one(
                        scenario,
                        severity_name,
                        severity_factor,
                        duration,
                        truth,
                    )
                    results.append(result)

                except Exception as error:
                    results.append({
                        "scenario": scenario,
                        "severity": severity_name,
                        "severity_factor": severity_factor,
                        "duration_minutes": duration,
                        "expected_root_cause": truth[scenario]["ground_truth"],
                        "predicted_root_cause": None,
                        "anomaly_detected": False,
                        "root_cause_correct": False,
                        "report_status": "ERROR",
                        "confidence": 0.0,
                        "investigation_operations": 0,
                        "agents_called": "",
                        "error": f"{type(error).__name__}: {error}",
                    })
                    print(f"  Failed: {type(error).__name__}: {error}")

    results_df = pd.DataFrame(results)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_path = OUTPUT_DIR / "rca_variation_results.csv"
    results_df.to_csv(results_path, index=False)

    print("\n=== RCA SEVERITY AND DURATION RESULTS ===")
    print(results_df.to_string(index=False))
    print("\nResults saved to:", results_path)

    print("\n=== SUMMARY BY SEVERITY ===")
    print(
        results_df.groupby("severity").agg(
            cases=("scenario", "count"),
            detection_rate=("anomaly_detected", "mean"),
            root_cause_accuracy=("root_cause_correct", "mean"),
            mean_confidence=("confidence", "mean"),
        ).round(3).to_string()
    )

    print("\n=== SUMMARY BY DURATION ===")
    print(
        results_df.groupby("duration_minutes").agg(
            cases=("scenario", "count"),
            detection_rate=("anomaly_detected", "mean"),
            root_cause_accuracy=("root_cause_correct", "mean"),
            mean_confidence=("confidence", "mean"),
        ).round(3).to_string()
    )


if __name__ == "__main__":
    main()
