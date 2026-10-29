
from typing import Any

from state.rca_state import RCAState


class HypothesisAgent:
    """
    Generate distinct root-cause hypotheses.

    Candidates are generated independently of ground-truth labels.
    The Critic Agent evaluates and ranks them using current evidence.
    """

    CANDIDATES = [
        {
            "id": "H1",
            "cause": "Database-side failure",
            "keywords": (
                "database",
                "query latency",
                "connection errors",
                "database connection",
                "database timeout",
                "active database connections",
            ),
            "reasoning": (
                "Database query performance or connection problems "
                "may be responsible for the incident."
            ),
        },
        {
            "id": "H2",
            "cause": "Network or dependency communication failure",
            "keywords": (
                "network",
                "dependency",
                "communication timeout",
                "connection refused",
                "communication failure",
                "dependency timeout",
            ),
            "reasoning": (
                "A communication problem between the service and "
                "a dependency may explain the incident."
            ),
        },
        {
            "id": "H3",
            "cause": "Payment-service internal failure",
            "keywords": (
                "unhandled exception",
                "application exception",
                "internal application error",
                "payment-service exception",
            ),
            "reasoning": (
                "An internal application error may explain the "
                "observed service failure."
            ),
        },
        {
            "id": "H4",
            "cause": "Resource exhaustion",
            "keywords": (
                "high cpu",
                "cpu saturation",
                "high memory",
                "memory exhaustion",
                "out of memory",
                "resource exhaustion",
                "resource saturation",
            ),
            "reasoning": (
                "Excessive CPU or memory consumption may be "
                "degrading service performance."
            ),
        },
        {
            "id": "H5",
            "cause": "Traffic overload",
            "keywords": (
                "traffic overload",
                "request overload",
                "high request rate",
                "too many requests",
                "request saturation",
                "traffic spike",
            ),
            "reasoning": (
                "Excessive incoming traffic may be overwhelming "
                "the service or one of its dependencies."
            ),
        },
    ]

    @staticmethod
    def _evidence_text(item: dict[str, Any]) -> str:
        """Build searchable text from an evidence record."""
        fields = (
            "type",
            "finding",
            "message",
            "metric",
            "name",
            "service",
            "database",
        )

        return " ".join(
            str(item.get(field, "") or "")
            for field in fields
        ).lower()

    def analyze(self, state: RCAState) -> dict[str, Any]:
        """Generate hypotheses by matching candidate keywords."""
        evidence = state.get("evidence", [])
        hypotheses = []

        for candidate in self.CANDIDATES:
            matching_evidence = []

            for item in evidence:
                text = self._evidence_text(item)

                if any(
                    keyword in text
                    for keyword in candidate["keywords"]
                ):
                    matching_evidence.append(item)

            hypotheses.append({
                "id": candidate["id"],
                "cause": candidate["cause"],
                "reasoning": candidate["reasoning"],
                "supporting_evidence": matching_evidence,
                "status": (
                    "EVIDENCE_MATCHED"
                    if matching_evidence
                    else "UNVERIFIED"
                ),
            })

        return {
            "hypotheses": hypotheses,
            "finding": (
                f"Generated {len(hypotheses)} distinct root-cause "
                "candidates for evidence-based evaluation."
            ),
        }
