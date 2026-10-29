
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.anomaly_agent import AnomalyAgent


SCENARIO_FILE = (
    ROOT
    / "data"
    / "scenarios"
    / "network_failure"
    / "simulated_observability.csv"
)

FEATURES = [
    "cpu",
    "memory",
    "latency",
    "error_rate",
    "request_rate",
]

INCIDENT_START = pd.Timestamp("2026-10-01 10:07:00")
INCIDENT_END = pd.Timestamp("2026-10-01 10:12:00")


def main():
    data = pd.read_csv(SCENARIO_FILE)
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    service_mask = data["service"] == "payment-service"
    service_data = data.loc[service_mask].copy()

    incident_mask = service_data["timestamp"].between(
        INCIDENT_START,
        INCIDENT_END,
    )
    baseline_mask = (
        (service_data["timestamp"] < INCIDENT_START)
        & (service_data["timestamp"] >= INCIDENT_START - pd.Timedelta(minutes=5))
    )

    baseline_values = service_data.loc[
        baseline_mask, FEATURES
    ].median()

    original_incident_values = service_data.loc[
        incident_mask, FEATURES
    ].mean()

    # Reproduce the variation experiment's moderate severity factor.
    severity_factor = 0.4

    modified_values = (
        baseline_values
        + severity_factor
        * (original_incident_values - baseline_values)
    )

    print("=== MODERATE NETWORK INCIDENT ===")
    print("\nBaseline values:")
    print(baseline_values.round(3).to_string())

    print("\nOriginal incident means:")
    print(original_incident_values.round(3).to_string())

    print("\nModified values at severity factor 0.4:")
    print(modified_values.round(3).to_string())

    print("\nExisting fixed thresholds:")
    print(AnomalyAgent.METRIC_THRESHOLDS)

    print("\nThreshold comparison:")
    for metric, threshold in AnomalyAgent.METRIC_THRESHOLDS.items():
        value = modified_values[metric]
        print(
            f"{metric}: value={value:.3f}, "
            f"threshold={threshold}, "
            f"exceeded={value > threshold}"
        )

    # Replicate the six-minute modification used by the variation test.
    modified_data = data.copy()
    modified_window = (
        (modified_data["service"] == "payment-service")
        & modified_data["timestamp"].between(
            INCIDENT_START,
            INCIDENT_END,
        )
    )

    for metric in FEATURES:
        modified_data.loc[modified_window, metric] = modified_values[metric]

    print("\n=== ANOMALY AGENT RESULT ===")
    result = AnomalyAgent().analyze(modified_data)

    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
