
from agents.critic_agent import CriticAgent


def make_state(evidence, abnormal_metrics=None):
    """Create a minimal RCA state for testing the critic."""
    return {
        "evidence": evidence,
        "hypotheses": [
            {
                "id": "H1",
                "cause": "Database-side failure",
            },
            {
                "id": "H2",
                "cause": "Network or dependency communication failure",
            },
            {
                "id": "H3",
                "cause": "Payment-service internal failure",
            },
            {
                "id": "H4",
                "cause": "Resource exhaustion",
            },
            {
                "id": "H5",
                "cause": "Traffic overload",
            },
        ],
        "abnormal_metrics": abnormal_metrics or [],
    }


def test_network_timeout_ranks_network_failure_first():
    critic = CriticAgent()

    state = make_state([
        {
            "type": "logs",
            "service": "payment-service",
            "finding": "Dependency communication timeout",
        }
    ])

    result = critic.analyze(state)

    assert (
        result["root_cause"]
        == "Network or dependency communication failure"
    )


def test_application_exception_ranks_application_failure_first():
    critic = CriticAgent()

    state = make_state([
        {
            "type": "logs",
            "service": "payment-service",
            "finding": "Unhandled exception in payment-service",
        }
    ])

    result = critic.analyze(state)

    assert (
        result["root_cause"]
        == "Payment-service internal failure"
    )


def test_database_evidence_ranks_database_failure_first():
    critic = CriticAgent()

    state = make_state([
        {
            "type": "database",
            "database": "payment-db",
            "finding": "High database query latency detected",
        },
        {
            "type": "database",
            "database": "payment-db",
            "finding": "Database connection errors detected",
        },
    ])

    result = critic.analyze(state)

    assert result["root_cause"] == "Database-side failure"


def test_generic_latency_does_not_prove_network_or_application_failure():
    critic = CriticAgent()

    state = make_state([
        {
            "type": "metrics",
            "service": "payment-service",
            "finding": "High latency detected",
        },
        {
            "type": "metrics",
            "service": "payment-service",
            "finding": "High error rate detected",
        },
    ])

    result = critic.analyze(state)

    # Without cause-specific evidence, the critic should not
    # confidently attribute symptoms to network or application failure.
    assert result["root_cause"] is None
    assert result["confidence"] == 0.0


def test_normal_database_latency_contradicts_database_failure():
    critic = CriticAgent()

    state = make_state([
        {
            "type": "traces",
            "finding": (
                "Payment-service latency elevated while "
                "database latency remains normal"
            ),
        }
    ])

    supporting, contradicting = critic._evaluate_evidence(
        CriticAgent.DATABASE_CAUSE,
        state["evidence"],
    )

    assert supporting == []
    assert len(contradicting) == 1
