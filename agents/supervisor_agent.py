
from typing import Any

from state.rca_state import RCAState


class SupervisorAgent:
    """
    Adaptive Supervisor Agent.

    Responsibilities:
    1. Select the next investigation agent.
    2. Track previously attempted actions.
    3. Investigate relevant missing evidence.
    4. Avoid repeated actions that produce no progress.
    5. Respect the investigation budget.
    6. Stop when the evidence supports a sufficiently
       confident root-cause hypothesis.
    """

    def __init__(self):
        self.max_tool_calls = 10
        self.minimum_confidence = 0.70

    def decide_next_action(
        self,
        state: RCAState,
    ) -> dict[str, Any]:

        # --------------------------------------------------
        # 1. Check whether an anomaly was detected
        # --------------------------------------------------

        if not state.get("anomaly_detected", False):
            return {
                "next_action": "STOP",
                "reason": "No anomaly detected.",
            }

        # --------------------------------------------------
        # 2. Read the current investigation state
        # --------------------------------------------------

        evidence = state.get("evidence", [])
        hypotheses = state.get("hypotheses", [])
        history = state.get("investigation_history", [])

        evidence_types = {
            item.get("type")
            for item in evidence
            if isinstance(item, dict)
        }

        # The history records actions even when an agent
        # returns no evidence. This prevents endless retries.
        attempted_actions = {
            entry.get("action")
            for entry in history
            if isinstance(entry, dict)
        }

        root_causes = {
            hypothesis.get("cause")
            for hypothesis in hypotheses
            if isinstance(hypothesis, dict)
        }

        tool_calls = state.get("tool_calls", 0)

        # --------------------------------------------------
        # 3. Check whether the current result is sufficient
        # --------------------------------------------------

        root_cause = state.get("root_cause")
        confidence = state.get("confidence", 0.0)

        if (
            root_cause
            and confidence >= self.minimum_confidence
        ):
            return {
                "next_action": "STOP",
                "reason": (
                    f"Evidence is sufficient. Root cause "
                    f"'{root_cause}' has confidence "
                    f"{confidence:.3f}, meeting the threshold "
                    f"of {self.minimum_confidence:.2f}."
                ),
            }

        # --------------------------------------------------
        # 4. Enforce the investigation budget
        # --------------------------------------------------

        if tool_calls >= self.max_tool_calls:
            return {
                "next_action": "STOP",
                "reason": "Investigation budget exhausted.",
            }

        # --------------------------------------------------
        # 5. Collect initial observability evidence
        # --------------------------------------------------

        if "logs" not in evidence_types:
            if "LOG_AGENT" not in attempted_actions:
                return {
                    "next_action": "LOG_AGENT",
                    "reason": (
                        "Application log evidence is missing."
                    ),
                }

        if "metrics" not in evidence_types:
            if "METRIC_AGENT" not in attempted_actions:
                return {
                    "next_action": "METRIC_AGENT",
                    "reason": (
                        "Metric evidence is missing."
                    ),
                }

        # --------------------------------------------------
        # 6. Generate initial hypotheses
        # --------------------------------------------------

        if not hypotheses:
            if "HYPOTHESIS_AGENT" not in attempted_actions:
                return {
                    "next_action": "HYPOTHESIS_AGENT",
                    "reason": (
                        "Initial observability evidence is "
                        "available. Generate competing hypotheses."
                    ),
                }

        # --------------------------------------------------
        # 7. Detect network or dependency-related evidence
        # --------------------------------------------------

        network_log_indicators = (
            "network",
            "dependency",
            "connection refused",
            "communication timeout",
            "communication failure",
        )

        network_hint_in_logs = any(
            any(
                indicator in str(item.get("finding", "")).lower()
                for indicator in network_log_indicators
            )
            for item in evidence
            if isinstance(item, dict)
            and item.get("type") == "logs"
        )

        network_hypothesis = (
            "Network or dependency communication failure"
            in root_causes
        )

        # A network hypothesis cannot be investigated through
        # traces until trace evidence has been collected.
        # Use log indicators to avoid a circular dependency:
        # the hypothesis may require traces to be generated.
        if (
            "traces" not in evidence_types
            and "TRACE_AGENT" not in attempted_actions
            and (network_hint_in_logs or network_hypothesis)
        ):
            return {
                "next_action": "TRACE_AGENT",
                "reason": (
                    "Network or dependency-related evidence "
                    "was observed. Inspect traces to determine "
                    "whether dependency latency supports this cause."
                ),
            }

        # --------------------------------------------------
        # 8. Refresh hypotheses after new trace evidence
        # --------------------------------------------------

        # The initial hypothesis agent may have run before
        # trace evidence existed. Run it again when traces
        # are available but the network hypothesis is missing.
        if (
            "traces" in evidence_types
            and not network_hypothesis
            and "HYPOTHESIS_AGENT" in attempted_actions
        ):
            last_hypothesis_position = max(
                (
                    index
                    for index, entry in enumerate(history)
                    if isinstance(entry, dict)
                    and entry.get("action") == "HYPOTHESIS_AGENT"
                ),
                default=-1,
            )

            trace_positions = [
                index
                for index, entry in enumerate(history)
                if isinstance(entry, dict)
                and entry.get("action") == "TRACE_AGENT"
            ]

            if (
                trace_positions
                and trace_positions[-1] > last_hypothesis_position
            ):
                return {
                    "next_action": "HYPOTHESIS_AGENT",
                    "reason": (
                        "New trace evidence is available. "
                        "Reconsider the competing hypotheses."
                    ),
                }

        # --------------------------------------------------
        # 9. Investigate database candidates once
        # --------------------------------------------------

        database_hypothesis = (
            "Database-side failure" in root_causes
        )

        if (
            database_hypothesis
            and "database" not in evidence_types
            and "DATABASE_AGENT" not in attempted_actions
        ):
            return {
                "next_action": "DATABASE_AGENT",
                "reason": (
                    "A database-side failure is a candidate "
                    "root cause. Collect database evidence."
                ),
            }

        # --------------------------------------------------
        # 10. Evaluate the current hypotheses
        # --------------------------------------------------

        # If a critic has already run, only run it again if
        # subsequent investigation added evidence or generated
        # a refreshed set of hypotheses.
        last_critic_position = max(
            (
                index
                for index, entry in enumerate(history)
                if isinstance(entry, dict)
                and entry.get("action") == "CRITIC_AGENT"
            ),
            default=-1,
        )

        investigation_after_critic = (
            history[last_critic_position + 1:]
            if last_critic_position >= 0
            else history
        )

        new_evidence_or_hypotheses = any(
            isinstance(entry, dict)
            and (
                entry.get("evidence_added", 0) > 0
                or entry.get("action") == "HYPOTHESIS_AGENT"
            )
            for entry in investigation_after_critic
        )

        if (
            "CRITIC_AGENT" not in attempted_actions
            or new_evidence_or_hypotheses
        ):
            return {
                "next_action": "CRITIC_AGENT",
                "reason": (
                    "Evaluate the current hypotheses against "
                    "the collected observability evidence."
                ),
            }

        # --------------------------------------------------
        # 11. No more useful actions are available
        # --------------------------------------------------

        return {
            "next_action": "STOP",
            "reason": (
                "No further eligible investigation action is "
                "available. The evidence or confidence may be "
                "insufficient for a definitive root cause."
            ),
        }
