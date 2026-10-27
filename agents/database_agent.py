
from typing import Any
import pandas as pd

from state.rca_state import RCAState
from tools.observability_tools import ObservabilityTools


class DatabaseAgent:
    """Analyze database signals and produce RCA evidence."""

    def __init__(self, scenario: str = "database_failure"):
        self.tools = ObservabilityTools(scenario=scenario)

    def analyze(self, state: RCAState) -> dict[str, Any]:
        database = "payment-db"
        database_data = self.tools.query_database(database)

        if database_data.empty:
            return {
                "evidence": [],
                "finding": "No database data found.",
            }

        database_data = database_data.copy()
        database_data["timestamp"] = pd.to_datetime(
            database_data["timestamp"]
        )

        incident_timestamp = state.get("timestamp")

        if incident_timestamp:
            timestamp = pd.to_datetime(incident_timestamp)
            start = timestamp.floor("min") - pd.Timedelta(minutes=2)
            end = timestamp.floor("min") + pd.Timedelta(minutes=3)

            database_data = database_data.loc[
                database_data["timestamp"].between(start, end)
            ]

        if database_data.empty:
            return {
                "evidence": [],
                "finding": (
                    "No database observations found in the incident window."
                ),
            }

        evidence = []

        highest_latency = database_data.loc[
            database_data["query_latency"].idxmax()
        ]

        if highest_latency["query_latency"] > 500:
            evidence.append({
                "type": "database",
                "database": database,
                "finding": "High database query latency detected",
                "value": float(highest_latency["query_latency"]),
            })

        highest_errors = database_data.loc[
            database_data["connection_errors"].idxmax()
        ]

        if highest_errors["connection_errors"] > 0:
            evidence.append({
                "type": "database",
                "database": database,
                "finding": "Database connection errors detected",
                "value": int(highest_errors["connection_errors"]),
            })

        highest_connections = database_data.loc[
            database_data["active_connections"].idxmax()
        ]

        if highest_connections["active_connections"] > 80:
            evidence.append({
                "type": "database",
                "database": database,
                "finding": "High number of active database connections",
                "value": int(highest_connections["active_connections"]),
            })

        return {
            "evidence": evidence,
            "finding": (
                f"Analyzed database observability data for {database}."
            ),
        }
