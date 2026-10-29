
from typing import Any

import pandas as pd

from state.rca_state import RCAState
from tools.observability_tools import ObservabilityTools


class TraceAgent:
    """
    Analyze distributed traces to identify likely failure locations.

    The agent produces structured evidence. It does not independently
    declare the final root cause.
    """

    DATABASE_ERROR_TYPES = {
        "DATABASE_TIMEOUT",
        "DATABASE_CONNECTION_ERROR",
        "DATABASE_ERROR",
    }

    NETWORK_ERROR_TYPES = {
        "DEPENDENCY_TIMEOUT",
        "NETWORK_TIMEOUT",
        "CONNECTION_REFUSED",
        "COMMUNICATION_FAILURE",
    }

    APPLICATION_ERROR_TYPES = {
        "APPLICATION_EXCEPTION",
        "UNHANDLED_EXCEPTION",
        "INTERNAL_APPLICATION_ERROR",
    }

    def __init__(self) -> None:
        self.tools = ObservabilityTools()

    def analyze(self, state: RCAState) -> dict[str, Any]:
        """Collect and interpret trace evidence for the incident."""

        service = state.get("affected_service", "payment-service")

        # 1. Retrieve traces involving the affected service.
        try:
            traces = self.tools.query_traces(service)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            return {
                "evidence": [],
                "finding": f"Trace data could not be retrieved: {exc}",
            }

        if traces.empty:
            return {
                "evidence": [],
                "finding": f"No traces found for service {service}.",
            }

        traces = traces.copy()

        # 2. Validate the columns required by this agent.
        required_columns = {
            "payment_latency",
            "database_latency",
        }
        missing_columns = required_columns - set(traces.columns)

        if missing_columns:
            return {
                "evidence": [],
                "finding": (
                    "Trace analysis could not proceed. Missing columns: "
                    f"{sorted(missing_columns)}"
                ),
            }

        # 3. Parse numeric latency fields safely.
        numeric_columns = [
            "payment_latency",
            "database_latency",
        ]

        if "dependency_latency" in traces.columns:
            numeric_columns.append("dependency_latency")

        for column in numeric_columns:
            traces[column] = pd.to_numeric(
                traces[column],
                errors="coerce",
            )

        traces = traces.dropna(
            subset=["payment_latency", "database_latency"]
        )

        if traces.empty:
            return {
                "evidence": [],
                "finding": "No valid trace latency observations were found.",
            }

        # 4. If an incident timestamp is available, focus on its
        # surrounding time window.
        incident_timestamp = state.get("timestamp")

        if incident_timestamp is not None and "timestamp" in traces.columns:
            traces["timestamp"] = pd.to_datetime(
                traces["timestamp"],
                errors="coerce",
            )

            incident_time = pd.to_datetime(
                incident_timestamp,
                errors="coerce",
            )

            if pd.notna(incident_time):
                valid_timestamps = traces["timestamp"].notna()

                window = traces.loc[
                    valid_timestamps
                    & (
                        traces["timestamp"]
                        >= incident_time.floor("min")
                        - pd.Timedelta(minutes=2)
                    )
                    & (
                        traces["timestamp"]
                        <= incident_time.floor("min")
                        + pd.Timedelta(minutes=3)
                    )
                ]

                # Use the incident window when it contains data.
                # Otherwise retain the available traces instead of
                # silently losing all evidence.
                if not window.empty:
                    traces = window.copy()

        # 5. Select the trace with the highest payment latency.
        # Other trace rows remain available for future aggregation.
        selected = traces.loc[
            traces["payment_latency"].idxmax()
        ]

        payment_latency = float(selected["payment_latency"])
        database_latency = float(selected["database_latency"])

        dependency_latency = self._optional_number(
            selected,
            "dependency_latency",
        )

        span_status = self._optional_text(
            selected,
            "span_status",
        ).upper()

        error_type = self._optional_text(
            selected,
            "error_type",
        ).upper()

        downstream_service = self._optional_text(
            selected,
            "downstream_service",
        )

        timestamp = self._optional_text(
            selected,
            "timestamp",
        )

        # 6. Build structured evidence for the critic.
        evidence: list[dict[str, Any]] = []

        common_fields = {
            "type": "traces",
            "service": service,
            "timestamp": timestamp,
            "payment_latency": payment_latency,
            "database_latency": database_latency,
            "dependency_latency": dependency_latency,
            "span_status": span_status,
            "error_type": error_type,
            "downstream_service": downstream_service,
        }

        # General latency is a symptom, not proof of a root cause.
        if payment_latency > 500:
            evidence.append(
                {
                    **common_fields,
                    "finding": "High payment-service latency detected",
                    "message": (
                        f"Payment latency is {payment_latency:.2f}; "
                        f"database latency is {database_latency:.2f}."
                    ),
                    "metric": "payment_latency",
                    "value": payment_latency,
                }
            )

        # Database-specific evidence.
        if (
            error_type in self.DATABASE_ERROR_TYPES
            or database_latency > 500
        ):
            evidence.append(
                {
                    **common_fields,
                    "finding": "Database timeout or high database latency",
                    "message": (
                        "Trace evidence indicates a possible database-side "
                        "problem."
                    ),
                    "metric": "database_latency",
                    "value": database_latency,
                }
            )

        # Network/dependency-specific evidence.
        if (
            error_type in self.NETWORK_ERROR_TYPES
            or (
                dependency_latency is not None
                and dependency_latency > 500
                and database_latency <= 500
            )
        ):
            evidence.append(
                {
                    **common_fields,
                    "finding": "Dependency communication failure detected",
                    "message": (
                        "Trace evidence indicates a possible dependency "
                        "communication problem."
                    ),
                    "metric": "dependency_latency",
                    "value": dependency_latency,
                }
            )

        # Application-specific evidence.
        if error_type in self.APPLICATION_ERROR_TYPES:
            evidence.append(
                {
                    **common_fields,
                    "finding": "Application exception detected in trace",
                    "message": (
                        "The trace reports an application-specific error."
                    ),
                    "metric": "span_status",
                    "value": span_status,
                }
            )

        # An error span with normal dependency and database latency
        # can help narrow the investigation toward the application.
        if (
            span_status in {"ERROR", "FAILED"}
            and database_latency <= 500
            and dependency_latency is not None
            and dependency_latency <= 500
            and error_type not in self.DATABASE_ERROR_TYPES
            and error_type not in self.NETWORK_ERROR_TYPES
        ):
            evidence.append(
                {
                    **common_fields,
                    "finding": (
                        "Error span with normal database and dependency "
                        "latency"
                    ),
                    "message": (
                        "The span failed while the observed database and "
                        "dependency latencies remained within thresholds."
                    ),
                    "metric": "span_status",
                    "value": span_status,
                }
            )

        if not evidence:
            evidence.append(
                {
                    **common_fields,
                    "finding": "No cause-specific trace evidence identified",
                    "message": (
                        "Trace data was available, but the observed fields "
                        "did not meet a cause-specific rule."
                    ),
                }
            )

        return {
            "evidence": evidence,
            "finding": (
                f"Analyzed {len(traces)} trace observations for {service}; "
                f"generated {len(evidence)} evidence item(s)."
            ),
        }

    @staticmethod
    def _optional_text(row: pd.Series, column: str) -> str:
        """Read an optional field without converting NaN to the text 'nan'."""

        if column not in row.index:
            return ""

        value = row[column]

        if pd.isna(value):
            return ""

        return str(value).strip()

    @staticmethod
    def _optional_number(
        row: pd.Series,
        column: str,
    ) -> float | None:
        """Read an optional numeric field safely."""

        if column not in row.index:
            return None

        try:
            value = float(row[column])
        except (TypeError, ValueError):
            return None

        if pd.isna(value) or value in (float("inf"), float("-inf")):
            return None

        return value
