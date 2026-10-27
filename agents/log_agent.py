
from typing import Any
import pandas as pd

from state.rca_state import RCAState
from tools.observability_tools import ObservabilityTools


class LogAgent:
    """Analyze application logs and produce RCA evidence."""

    def __init__(self, scenario: str = "database_failure"):
        self.tools = ObservabilityTools(scenario=scenario)

    def analyze(self, state: RCAState) -> dict[str, Any]:
        service = state["affected_service"]
        logs = self.tools.query_logs(service)

        if logs.empty:
            return {
                "evidence": [],
                "finding": f"No logs found for {service}.",
            }

        # Restrict analysis to the incident window when a timestamp
        # is available. The simulated scenarios use a six-minute window.
        incident_timestamp = state.get("timestamp")

        if incident_timestamp and "timestamp" in logs.columns:
            timestamp = pd.to_datetime(incident_timestamp)
            logs = logs.copy()
            logs["timestamp"] = pd.to_datetime(logs["timestamp"])

            start = timestamp.floor("min") - pd.Timedelta(minutes=2)
            end = timestamp.floor("min") + pd.Timedelta(minutes=3)

            logs = logs.loc[
                logs["timestamp"].between(start, end)
            ]

        error_logs = logs.loc[logs["level"] == "ERROR"]
        evidence = []

        if not error_logs.empty:
            counts = error_logs["message"].value_counts()

            for message, count in counts.items():
                evidence.append({
                    "type": "logs",
                    "service": service,
                    "finding": str(message),
                    "occurrences": int(count),
                })

        return {
            "evidence": evidence,
            "finding": (
                f"Found {len(error_logs)} error log entries "
                f"for {service} in the selected window."
            ),
        }
