
import numpy as np
import pandas as pd
import pytest

from agents.anomaly_agent import AnomalyAgent
from models.anomaly_detector import AnomalyDetector


@pytest.fixture
def normal_data():
    """Create reproducible, normal-looking observations."""
    rng = np.random.default_rng(42)
    rows = 100

    return pd.DataFrame({
        "timestamp": pd.date_range(
            "2026-10-01", periods=rows, freq="min"
        ),
        "service": ["payment-service"] * rows,
        "cpu": rng.uniform(20, 60, rows),
        "memory": rng.uniform(30, 70, rows),
        "latency": rng.uniform(50, 200, rows),
        "error_rate": rng.uniform(0, 2, rows),
        "request_rate": rng.uniform(100, 500, rows),
    })


def test_detector_rejects_empty_data():
    detector = AnomalyDetector()

    with pytest.raises(ValueError, match="empty dataset"):
        detector.train(pd.DataFrame())


def test_detector_rejects_missing_features():
    detector = AnomalyDetector()
    data = pd.DataFrame({
        "cpu": [30],
        "memory": [50],
    })

    with pytest.raises(ValueError, match="Missing required metric columns"):
        detector.train(data)


def test_agent_rejects_missing_service_column(normal_data):
    agent = AnomalyAgent()
    data = normal_data.drop(columns=["service"])

    with pytest.raises(ValueError, match="service"):
        agent.analyze(data)


def test_agent_reports_missing_target_service(normal_data):
    agent = AnomalyAgent()

    result = agent.analyze(
        normal_data,
        target_service="unknown-service",
    )

    assert result["anomaly_detected"] is False
    assert result["affected_service"] == "unknown-service"
    assert result["anomaly_count"] == 0


def test_agent_returns_expected_fields_for_actionable_incident(
    normal_data,
):
    agent = AnomalyAgent()

    # Add an extreme observation that should be a statistical
    # outlier and cross the actionable latency/error thresholds.
    incident = normal_data.iloc[[0]].copy()
    incident.loc[:, "timestamp"] = pd.Timestamp("2026-10-02")
    incident.loc[:, "cpu"] = 95
    incident.loc[:, "memory"] = 95
    incident.loc[:, "latency"] = 900
    incident.loc[:, "error_rate"] = 20
    incident.loc[:, "request_rate"] = 900

    data = pd.concat([normal_data, incident], ignore_index=True)
    result = agent.analyze(data)

    assert result["anomaly_detected"] is True
    assert result["affected_service"] == "payment-service"
    assert "latency" in result["abnormal_metrics"]
    assert "error_rate" in result["abnormal_metrics"]
    assert result["severity"] == "CRITICAL"
    assert isinstance(result["anomaly_score"], float)
    assert result["anomaly_count"] >= 1


def test_agent_ignores_other_services(normal_data):
    agent = AnomalyAgent()

    other_service = normal_data.copy()
    other_service["service"] = "auth-service"
    other_service["latency"] = 2000
    other_service["error_rate"] = 30

    data = pd.concat(
        [normal_data, other_service],
        ignore_index=True,
    )

    result = agent.analyze(
        data,
        target_service="payment-service",
    )

    assert result["affected_service"] == "payment-service"
    assert result["anomaly_detected"] is False
