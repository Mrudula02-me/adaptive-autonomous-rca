
from typing import Any

from state.rca_state import RCAState
from tools.observability_tools import ObservabilityTools


class TraceAgent:
    """
    Analyze distributed traces to distinguish service latency
    from database/dependency latency.
    """

    def __init__(self, scenario="database_failure"):
        self.tools = ObservabilityTools(scenario=scenario)

    def analyze(self, state: RCAState) -> dict[str, Any]:
        service = state["affected_service"]
        traces = self.tools.query_traces(service)

        if traces.empty:
            return {
                "evidence": [],
                "finding": "No traces found for the affected service.",
            }

        required_columns = {
            "payment_latency",
            "database_latency",
        }

        missing_columns = required_columns - set(traces.columns)

        if missing_columns:
            return {
                "evidence": [],
                "finding": (
                    "Trace analysis could not run because these "
                    f"columns are missing: {sorted(missing_columns)}"
                ),
            }

        evidence = []

        # Select the row with the highest payment latency.
        # Its database latency provides a comparison from the
        # same trace observation.
        payment_row = traces.loc[
            traces["payment_latency"].idxmax()
        ]

        payment_latency = float(payment_row["payment_latency"])
        database_latency = float(payment_row["database_latency"])

        if payment_latency > 500:
            evidence.append({
                "type": "traces",
                "service": service,
                "finding": "High payment-service latency detected",
                "value": payment_latency,
            })

        # Check the database latency from the same observation.
        if database_latency > 500:
            evidence.append({
                "type": "traces",
                "service": service,
                "finding": "High database latency detected",
                "value": database_latency,
            })

        elif payment_latency > 500:
            # This is a comparison signal, not proof of a network fault.
            evidence.append({
                "type": "traces",
                "service": service,
                "finding": (
                    "Payment-service latency elevated while "
                    "database latency remains normal"
                ),
                "payment_latency": payment_latency,
                "database_latency": database_latency,
            })

        return {
            "evidence": evidence,
            "finding": (
                f"Analyzed distributed traces for {service}; "
                f"payment latency={payment_latency:.1f}, "
                f"database latency={database_latency:.1f}."
            ),
        }
