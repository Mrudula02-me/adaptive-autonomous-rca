
from agents.supervisor_agent import SupervisorAgent


def test_supervisor_stops_when_no_anomaly():
    supervisor = SupervisorAgent()

    state = {
        "anomaly_detected": False,
        "evidence": [],
        "hypotheses": [],
        "investigation_history": [],
        "tool_calls": 0,
    }

    decision = supervisor.decide_next_action(state)

    assert decision["next_action"] == "STOP"
    assert "No anomaly" in decision["reason"]


def test_supervisor_selects_log_agent_when_logs_are_missing():
    supervisor = SupervisorAgent()

    state = {
        "anomaly_detected": True,
        "evidence": [],
        "hypotheses": [],
        "investigation_history": [],
        "tool_calls": 0,
    }

    decision = supervisor.decide_next_action(state)

    assert decision["next_action"] == "LOG_AGENT"
    assert "log" in decision["reason"].lower()


def test_supervisor_selects_database_agent_for_database_candidate():
    supervisor = SupervisorAgent()

    state = {
        "anomaly_detected": True,
        "evidence": [
            {
                "type": "logs",
                "finding": "Database connection timeout",
            },
            {
                "type": "metrics",
                "finding": "High latency detected",
            },
        ],
        "hypotheses": [
            {"cause": "Database-side failure"},
        ],
        "investigation_history": [
            {"action": "LOG_AGENT", "evidence_added": 1},
            {"action": "METRIC_AGENT", "evidence_added": 1},
            {"action": "HYPOTHESIS_AGENT", "evidence_added": 0},
        ],
        "tool_calls": 3,
    }

    decision = supervisor.decide_next_action(state)

    assert decision["next_action"] == "DATABASE_AGENT"
    assert "database" in decision["reason"].lower()


def test_supervisor_stops_when_tool_budget_is_exhausted():
    supervisor = SupervisorAgent()

    state = {
        "anomaly_detected": True,
        "evidence": [],
        "hypotheses": [],
        "investigation_history": [],
        "tool_calls": supervisor.max_tool_calls,
    }

    decision = supervisor.decide_next_action(state)

    assert decision["next_action"] == "STOP"
    assert "budget" in decision["reason"].lower()
