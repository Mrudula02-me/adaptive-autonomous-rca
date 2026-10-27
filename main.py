
from pathlib import Path
from typing import Any

import pandas as pd

from agents.anomaly_agent import AnomalyAgent
from agents.supervisor_agent import SupervisorAgent
from agents.log_agent import LogAgent
from agents.metric_agent import MetricAgent
from agents.trace_agent import TraceAgent
from agents.database_agent import DatabaseAgent
from agents.hypothesis_agent import HypothesisAgent
from agents.critic_agent import CriticAgent

from tools.observability_tools import ObservabilityTools


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SCENARIOS_DIR = PROJECT_ROOT / "data" / "scenarios"

DEFAULT_SCENARIO = "database_failure"

VALID_SCENARIOS = {
    "database_failure",
    "network_failure",
    "application_failure",
    "normal_operation",
}

MAX_ITERATIONS = 15
MINIMUM_CONFIDENCE = 0.70


# ============================================================
# 2. CONFIDENCE-AWARE REPORTING
# ============================================================

def apply_confidence_aware_reporting(
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    Classify the final RCA result according to its confidence.

    SUPPORTED:
        A leading root-cause hypothesis meets the threshold.

    PROVISIONAL:
        A leading hypothesis exists, but its confidence is
        below the configured threshold.

    UNDETERMINED:
        An anomaly exists, but no evaluated root cause is available.

    NO_INCIDENT:
        No actionable anomaly was detected.

    The original root_cause and confidence fields are preserved.
    """

    confidence = float(state.get("confidence") or 0.0)
    root_cause = state.get("root_cause")

    # Case 1: No actionable anomaly.
    if not state.get("anomaly_detected", False):
        state["report_status"] = "NO_INCIDENT"
        state["reported_root_cause"] = None
        state["provisional_hypothesis"] = None

        state["reporting_reason"] = (
            "No actionable anomaly was detected. "
            "Root-cause investigation was not required."
        )

        return state

    # Case 2: An anomaly exists, but no root cause was evaluated.
    if not root_cause:
        state["report_status"] = "UNDETERMINED"
        state["reported_root_cause"] = None
        state["provisional_hypothesis"] = None

        state["reporting_reason"] = (
            "An anomaly was detected, but the investigation "
            "did not establish a leading root-cause hypothesis."
        )

        return state

    # Case 3: The leading hypothesis meets the threshold.
    if confidence >= MINIMUM_CONFIDENCE:
        state["report_status"] = "SUPPORTED"
        state["reported_root_cause"] = root_cause
        state["provisional_hypothesis"] = None

        state["reporting_reason"] = (
            f"The leading hypothesis has a heuristic confidence "
            f"score of {confidence:.3f}, meeting the configured "
            f"threshold of {MINIMUM_CONFIDENCE:.2f}."
        )

        return state

    # Case 4: A leading hypothesis exists, but confidence is low.
    state["report_status"] = "PROVISIONAL"
    state["reported_root_cause"] = None
    state["provisional_hypothesis"] = root_cause

    state["reporting_reason"] = (
        f"The leading hypothesis has a heuristic confidence "
        f"score of {confidence:.3f}, below the configured "
        f"threshold of {MINIMUM_CONFIDENCE:.2f}. "
        "Further investigation or human review is recommended."
    )

    return state


# ============================================================
# 3. PRINT THE INVESTIGATION HEADER
# ============================================================

def print_investigation_header(state: dict[str, Any]) -> None:
    print("\n========== RCA INVESTIGATION ==========")
    print(f"Scenario: {state.get('scenario')}")
    print(f"Incident ID: {state.get('incident_id')}")
    print(f"Affected service: {state.get('affected_service')}")
    print(f"Timestamp: {state.get('timestamp')}")
    print()


# ============================================================
# 4. PRINT THE FINAL RCA REPORT
# ============================================================

def print_final_report(state: dict[str, Any]) -> None:
    print("\n========== RANKED HYPOTHESES ==========")

    ranked_hypotheses = state.get(
        "evaluated_hypotheses", []
    )

    if ranked_hypotheses:
        ranked_hypotheses = sorted(
            ranked_hypotheses,
            key=lambda item: float(item.get("confidence", 0.0)),
            reverse=True,
        )

        for index, hypothesis in enumerate(
            ranked_hypotheses, start=1
        ):
            print(
                f"{index}. {hypothesis.get('cause', 'Unknown')} "
                f"| Confidence: "
                f"{float(hypothesis.get('confidence', 0.0)):.3f}"
            )
    else:
        print("No evaluated hypotheses available.")

    print("\n========== FINAL RCA RESULT ==========")
    print(f"Scenario: {state.get('scenario')}")
    print(f"Incident: {state.get('incident_id')}")
    print(f"Affected service: {state.get('affected_service')}")
    print(f"Anomaly detected: {state.get('anomaly_detected')}")
    print(f"Severity: {state.get('severity', 'UNKNOWN')}")

    print(f"Report status: {state.get('report_status')}")

    print(
        f"Supported root cause: "
        f"{state.get('reported_root_cause')}"
    )

    print(
        f"Provisional hypothesis: "
        f"{state.get('provisional_hypothesis')}"
    )

    print(
        f"Leading hypothesis confidence: "
        f"{float(state.get('confidence') or 0.0):.3f}"
    )

    print(f"Impact: {state.get('impact')}")
    print(f"Reporting reason: {state.get('reporting_reason')}")

    print("\nSupporting evidence:")

    evidence = state.get("evidence", [])

    if evidence:
        for index, item in enumerate(evidence, start=1):
            print(f"{index}. {item}")
    else:
        print("No evidence collected.")

    rejected = state.get("rejected_hypotheses", [])

    print("\nRejected or weaker hypotheses:")

    if rejected:
        for item in rejected:
            if isinstance(item, dict):
                print(
                    f"- {item.get('cause', 'Unknown')}: "
                    f"{item.get('reasoning', item.get('reason', ''))}"
                )
            else:
                print(f"- {item}")
    else:
        print("None recorded.")

    print(f"\nAgents called: {state.get('agents_called', [])}")
    print(
        f"Investigation operations: "
        f"{state.get('tool_calls', 0)}"
    )

    print("======================================")

    # Make the reporting decision visible in plain language.
    status = state.get("report_status")

    if status == "SUPPORTED":
        print(
            "\nRCA outcome: A leading hypothesis meets the "
            "configured confidence threshold."
        )
    elif status == "PROVISIONAL":
        print(
            "\nRCA outcome: A provisional hypothesis is available. "
            "Do not treat it as a confirmed root cause."
        )
    elif status == "UNDETERMINED":
        print(
            "\nRCA outcome: The cause remains undetermined."
        )
    else:
        print(
            "\nRCA outcome: No actionable incident was detected."
        )


# ============================================================
# 5. MAIN RCA WORKFLOW
# ============================================================

def main(
    incident_config: dict[str, Any] | None = None,
) -> dict[str, Any]:

    config = incident_config or {}

    scenario = config.get(
        "scenario",
        DEFAULT_SCENARIO,
    )

    # --------------------------------------------------------
    # Validate the requested scenario.
    # --------------------------------------------------------

    if scenario not in VALID_SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{scenario}'. "
            f"Available scenarios: {sorted(VALID_SCENARIOS)}"
        )

    scenario_dir = SCENARIOS_DIR / scenario

    if not scenario_dir.is_dir():
        raise FileNotFoundError(
            f"Scenario directory not found: {scenario_dir}"
        )

    metrics_file = (
        scenario_dir / "simulated_observability.csv"
    )

    if not metrics_file.is_file():
        raise FileNotFoundError(
            f"Metrics file not found: {metrics_file}"
        )

    # --------------------------------------------------------
    # Set up the initial incident state.
    # --------------------------------------------------------

    default_config = {
        "incident_id": "INC-001",
        "timestamp": "2026-10-01 10:09:00",
        "affected_service": "payment-service",
        "severity": "UNKNOWN",
        "anomaly_score": 0.0,
        "abnormal_metrics": [],
        "anomaly_values": {},
    }

    incident_details = {
        **default_config,
        **config,
    }

    state: dict[str, Any] = {
        "scenario": scenario,
        "incident_id": incident_details["incident_id"],
        "timestamp": incident_details["timestamp"],
        "affected_service": incident_details["affected_service"],
        "anomaly_detected": False,
        "severity": incident_details["severity"],
        "anomaly_score": incident_details["anomaly_score"],
        "abnormal_metrics": incident_details["abnormal_metrics"],
        "anomaly_values": incident_details["anomaly_values"],
        "evidence": [],
        "hypotheses": [],
        "evaluated_hypotheses": [],
        "rejected_hypotheses": [],
        "investigation_history": [],
        "agents_called": [],
        "tool_calls": 0,
        "estimated_tokens": 0,
        "root_cause": None,
        "confidence": 0.0,
        "impact": None,
        "remediation_action": None,
        "remediation_result": None,
        "verification_result": None,
    }

    print_investigation_header(state)

    # --------------------------------------------------------
    # Initialize the agents.
    # --------------------------------------------------------

    anomaly_agent = AnomalyAgent()
    supervisor = SupervisorAgent()

    log_agent = LogAgent()
    metric_agent = MetricAgent()
    trace_agent = TraceAgent()
    database_agent = DatabaseAgent()

    hypothesis_agent = HypothesisAgent()
    critic_agent = CriticAgent()

    # Give all observability agents the selected scenario's data.
    scenario_tools = ObservabilityTools(scenario=scenario)

    for agent in (
        log_agent,
        metric_agent,
        trace_agent,
        database_agent,
    ):
        if hasattr(agent, "tools"):
            agent.tools = scenario_tools

    # --------------------------------------------------------
    # Load the selected scenario's metric data.
    # --------------------------------------------------------

    metrics_data = pd.read_csv(metrics_file)

    # --------------------------------------------------------
    # Run anomaly detection before starting RCA.
    # --------------------------------------------------------

    anomaly_result = anomaly_agent.analyze(
        metrics_data,
        target_service=state["affected_service"],
    )

    state.update(anomaly_result)

    # Restore the selected scenario in case an agent returned
    # additional fields in its result.
    state["scenario"] = scenario

    # A non-incident must not retain a misleading severity
    # from an earlier default configuration.
    if not state.get("anomaly_detected", False):
        state["severity"] = "NONE"
        state["root_cause"] = None
        state["confidence"] = 0.0
        state["impact"] = None

    state["agents_called"].append("anomaly_agent")

    # --------------------------------------------------------
    # Skip all RCA agents when no actionable anomaly exists.
    # --------------------------------------------------------

    if not state.get("anomaly_detected", False):
        print(
            f"\nAnomaly detected: False\n"
            f"Reason: {state.get('message', 'No actionable anomaly.')}"
        )

        state = apply_confidence_aware_reporting(state)
        print("\nSupervisor decision: STOP")
        print("Reason: No anomaly detected.")
        print("Investigation stopped.")

        print_final_report(state)
        return state

    # --------------------------------------------------------
    # Display anomaly details.
    # --------------------------------------------------------

    print(
        f"\nAnomaly detected: {state.get('anomaly_detected')}"
    )
    print(f"Severity: {state.get('severity')}")
    print(f"Anomaly score: {state.get('anomaly_score')}")
    print(f"Abnormal metrics: {state.get('abnormal_metrics')}")

    # --------------------------------------------------------
    # Map supervisor actions to agents.
    # --------------------------------------------------------

    action_to_agent = {
        "LOG_AGENT": log_agent,
        "METRIC_AGENT": metric_agent,
        "TRACE_AGENT": trace_agent,
        "DATABASE_AGENT": database_agent,
        "HYPOTHESIS_AGENT": hypothesis_agent,
        "CRITIC_AGENT": critic_agent,
    }

    # --------------------------------------------------------
    # Adaptive investigation loop.
    # --------------------------------------------------------

    for _ in range(MAX_ITERATIONS):

        decision = supervisor.decide_next_action(state)

        action = decision.get(
            "next_action",
            "STOP",
        )

        reason = decision.get(
            "reason",
            "No reason supplied.",
        )

        print(f"\nSupervisor decision: {action}")
        print(f"Reason: {reason}")

        if action == "STOP":
            print("Investigation stopped.")
            break

        agent = action_to_agent.get(action)

        if agent is None:
            print(
                f"Unknown supervisor action '{action}'. "
                "Stopping investigation."
            )
            break

        # Record the number of evidence items before the call.
        previous_evidence_count = len(
            state.get("evidence", [])
        )

        # ----------------------------------------------------
        # Execute the selected agent.
        # ----------------------------------------------------

        try:
            result = agent.analyze(state)

        except Exception as error:
            print(
                f"{action} failed with "
                f"{type(error).__name__}: {error}"
            )

            state["investigation_history"].append({
                "action": action,
                "evidence_added": 0,
                "error": str(error),
            })

            state["agents_called"].append(
                action.lower()
            )

            state["tool_calls"] += 1

            continue

        if not isinstance(result, dict):
            result = {}

        # ----------------------------------------------------
        # Merge evidence without discarding previous findings.
        # ----------------------------------------------------

        new_evidence = result.get("evidence", [])

        if isinstance(new_evidence, list) and new_evidence:
            state.setdefault("evidence", []).extend(
                new_evidence
            )

        # ----------------------------------------------------
        # Update hypotheses when the Hypothesis Agent runs.
        # ----------------------------------------------------

        if action == "HYPOTHESIS_AGENT":
            if "hypotheses" in result:
                state["hypotheses"] = result["hypotheses"]

        # ----------------------------------------------------
        # Update the evaluated results from the Critic Agent.
        # ----------------------------------------------------

        elif action == "CRITIC_AGENT":

            if "evaluated_hypotheses" in result:
                state["evaluated_hypotheses"] = (
                    result["evaluated_hypotheses"]
                )

            if "rejected_hypotheses" in result:
                state["rejected_hypotheses"] = (
                    result["rejected_hypotheses"]
                )

            if "root_cause" in result:
                state["root_cause"] = result["root_cause"]

            if "confidence" in result:
                state["confidence"] = float(
                    result["confidence"] or 0.0
                )

            if "impact" in result:
                state["impact"] = result["impact"]

        # ----------------------------------------------------
        # Record investigation history and budget usage.
        # ----------------------------------------------------

        evidence_added = (
            len(state.get("evidence", []))
            - previous_evidence_count
        )

        state["investigation_history"].append({
            "action": action,
            "evidence_added": evidence_added,
        })

        state["agents_called"].append(
            action.lower()
        )

        state["tool_calls"] += 1

        print(f"{action.lower()} completed.")
        print(f"New evidence items: {evidence_added}")

        if action == "CRITIC_AGENT":
            print(
                f"Current root cause: "
                f"{state.get('root_cause')}"
            )
            print(
                f"Current confidence: "
                f"{float(state.get('confidence') or 0.0):.3f}"
            )

    else:
        print(
            "\nMaximum loop iterations reached. "
            "Investigation stopped."
        )

    # --------------------------------------------------------
    # Apply confidence-aware reporting after the investigation.
    # --------------------------------------------------------

    state = apply_confidence_aware_reporting(state)

    # --------------------------------------------------------
    # Print the final report and return the complete state.
    # --------------------------------------------------------

    print_final_report(state)

    return state


# ============================================================
# 6. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
