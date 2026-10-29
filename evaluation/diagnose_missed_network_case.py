from pathlib import Path
import pandas as pd
import sys
from pathlib import Path

# Locate the project root directory.
ROOT = Path(__file__).resolve().parents[1]

# Add the project root so Python can find models and agents.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from models.anomaly_detector import AnomalyDetector
from agents.anomaly_agent import AnomalyAgent


ROOT = Path(__file__).resolve().parents[1]

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


def main():
    data = pd.read_csv(SCENARIO_FILE)
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    service_data = data.loc[
        data["service"] == "payment-service"
    ].copy()

    incident_start = pd.Timestamp("2026-10-01 10:07:00")
    incident_end = pd.Timestamp("2026-10-01 10:12:00")

    incident_mask = service_data["timestamp"].between(
        incident_start,
        incident_end,
    )

    normal_mask = service_data["timestamp"] < incident_start

    print("\n=== NETWORK SCENARIO METRICS ===")

    print("\nBaseline summary (before 10:07):")
    print(service_data.loc[normal_mask, FEATURES].describe().round(2))

    print("\nIncident-window summary (10:07–10:12):")
    print(service_data.loc[incident_mask, FEATURES].describe().round(2))

    print("\nIncident-window rows:")
    print(
        service_data.loc[
            incident_mask,
            ["timestamp", *FEATURES],
        ].to_string(index=False)
    )

    detector = AnomalyDetector()
    detector.train(service_data)
    detected = detector.detect(service_data)

    detected = detected.loc[incident_mask].copy()

    print("\n=== ISOLATION FOREST OUTPUT DURING INCIDENT ===")
    print(
        detected[
            ["timestamp", *FEATURES, "anomaly_label", "anomaly_score"]
        ].to_string(index=False)
    )

    print("\n=== CURRENT ANOMALY AGENT OUTPUT ===")
    result = AnomalyAgent().analyze(
        data,
        target_service="payment-service",
    )

    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
