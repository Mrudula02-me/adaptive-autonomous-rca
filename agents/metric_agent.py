
from typing import Any
import pandas as pd

from state.rca_state import RCAState
from tools.observability_tools import ObservabilityTools


class MetricAgent:
    """Analyze service metrics and produce RCA evidence."""

    def __init__(self, scenario: str = "database_failure"):
        self.tools = ObservabilityTools(scenario=scenario)

    def analyze(self, state: RCAState) -> dict[str, Any]:
        service = state["affected_service"]
        metrics = self.tools.query_metrics(service)

        if metrics.empty:
            return {
                "evidence": [],
                "finding": f"No metrics found for {service}.",
            }

        metrics = metrics.copy()
        metrics["timestamp"] = pd.to_datetime(metrics["timestamp"])

        incident_timestamp = state.get("timestamp")

        if incident_timestamp:
            timestamp = pd.to_datetime(incident_timestamp)
            start = timestamp.floor("min") - pd.Timedelta(minutes=2)
            end = timestamp.floor("min") + pd.Timedelta(minutes=3)

            metrics = metrics.loc[
                metrics["timestamp"].between(start, end)
            ]

        if metrics.empty:
            return {
                "evidence": [],
                "finding": (
                    f"No metrics found in the incident window "
                    f"for {service}."
                ),
            }

        # Check the maximum latency observation within the window.
        highest_latency = metrics.loc[metrics["latency"].idxmax()]
        evidence = []

        if highest_latency["latency"] > 500:
            evidence.append({
                "type": "metrics",
                "service": service,
                "finding": "High latency detected",
                "value": float(highest_latency["latency"]),
            })

        # Check the maximum error rate independently. The row with
        # maximum latency need not also have the maximum error rate.
        highest_error_rate = metrics.loc[metrics["error_rate"].idxmax()]

        if highest_error_rate["error_rate"] > 5:
            evidence.append({
                "type": "metrics",
                "service": service,
                "finding": "High error rate detected",
                "value": float(highest_error_rate["error_rate"]),
            })

        return {
            "evidence": evidence,
            "finding": f"Analyzed metrics for {service}.",
        }
