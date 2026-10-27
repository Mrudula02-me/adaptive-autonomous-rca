
from typing import Any

from state.rca_state import RCAState


class CriticAgent:
    """
    Evaluate and rank competing root-cause hypotheses.

    The Critic reevaluates each hypothesis against all current
    evidence. Generic symptoms are weaker than cause-specific
    evidence. Confidence is a heuristic score, not a probability.
    """

    DATABASE_CAUSE = "Database-side failure"
    NETWORK_CAUSE = "Network or dependency communication failure"
    APPLICATION_CAUSE = "Payment-service internal failure"
    RESOURCE_CAUSE = "Resource exhaustion"
    TRAFFIC_CAUSE = "Traffic overload"

    @staticmethod
    def _text(item: dict[str, Any]) -> str:
        fields = (
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

    def _evaluate_evidence(
        self,
        cause: str,
        evidence: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        supporting = []
        contradicting = []

        for item in evidence:
            evidence_type = str(item.get("type", "")).lower()
            text = self._text(item)

            # H1: Database-side failure
            if cause == self.DATABASE_CAUSE:
                if evidence_type == "database" and any(
                    word in text for word in (
                        "query latency",
                        "connection errors",
                        "active database connections",
                        "database",
                    )
                ):
                    supporting.append(item)

                elif evidence_type == "logs" and any(
                    word in text for word in (
                        "database",
                        "query latency",
                        "connection timeout",
                    )
                ):
                    supporting.append(item)

                elif evidence_type == "traces" and any(
                    word in text for word in (
                        "database",
                        "query latency",
                    )
                ):
                    supporting.append(item)

            # H2: Network or dependency communication failure
            elif cause == self.NETWORK_CAUSE:
                if evidence_type == "logs" and any(
                    word in text for word in (
                        "network",
                        "dependency",
                        "communication timeout",
                        "connection refused",
                        "communication failure",
                    )
                ):
                    supporting.append(item)

                elif evidence_type == "traces" and any(
                    word in text for word in (
                        "network",
                        "dependency",
                        "communication",
                    )
                ):
                    supporting.append(item)

                elif evidence_type == "database" and any(
                    word in text for word in (
                        "query latency",
                        "connection errors",
                    )
                ):
                    contradicting.append(item)

            # H3: Internal application failure
            elif cause == self.APPLICATION_CAUSE:
                if evidence_type == "logs" and any(
                    word in text for word in (
                        "unhandled exception",
                        "application exception",
                        "internal application error",
                        "payment-service exception",
                    )
                ):
                    supporting.append(item)

                elif evidence_type == "metrics" and any(
                    word in text for word in (
                        "latency",
                        "error rate",
                    )
                ):
                    # Generic symptoms are weak evidence only.
                    supporting.append(item)

                elif evidence_type == "database" and any(
                    word in text for word in (
                        "query latency",
                        "connection errors",
                    )
                ):
                    contradicting.append(item)

            # H4: Resource exhaustion
            elif cause == self.RESOURCE_CAUSE:
                if any(
                    word in text for word in (
                        "cpu saturation",
                        "high cpu",
                        "memory exhaustion",
                        "high memory",
                        "out of memory",
                        "resource exhaustion",
                        "resource saturation",
                    )
                ):
                    supporting.append(item)

                elif (
                    evidence_type == "metrics"
                    and any(
                        word in text
                        for word in ("cpu", "memory")
                    )
                ):
                    supporting.append(item)

            # H5: Traffic overload
            elif cause == self.TRAFFIC_CAUSE:
                if any(
                    word in text for word in (
                        "traffic overload",
                        "request overload",
                        "high request rate",
                        "too many requests",
                        "request saturation",
                        "traffic spike",
                    )
                ):
                    supporting.append(item)

            # Do not infer traffic overload merely from high latency.
            # Request rate alone may need a baseline for comparison.

        return supporting, contradicting

    @staticmethod
    def _calculate_scores(
        supporting: list[dict[str, Any]],
        contradicting: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Calculate a transparent heuristic score.

        Evidence from different modalities contributes to agreement.
        Duplicate findings within one modality do not count as
        independent modalities.
        """
        modality_types = {
            str(item.get("type", "")).lower()
            for item in supporting
            if item.get("type")
        }

        support_count = len(supporting)
        contradiction_count = len(contradicting)
        modality_count = len(modality_types)

        # Count distinct modalities rather than only raw item totals.
        if support_count == 0:
            support_score = 0.0
        elif support_count == 1:
            support_score = 0.30
        elif support_count == 2:
            support_score = 0.55
        elif support_count == 3:
            support_score = 0.75
        elif support_count == 4:
            support_score = 0.90
        else:
            support_score = 1.0

        cross_modal_score = min(modality_count / 3, 1.0)
        contradiction_penalty = min(
            contradiction_count * 0.10,
            0.50,
        )

        confidence = max(
            0.0,
            min(
                0.50 * support_score
                + 0.50 * cross_modal_score
                - contradiction_penalty,
                1.0,
            ),
        )

        return {
            "support_count": support_count,
            "contradiction_count": contradiction_count,
            "modality_count": modality_count,
            "support_score": round(support_score, 3),
            "cross_modal_score": round(cross_modal_score, 3),
            "confidence": round(confidence, 3),
        }

    def analyze(self, state: RCAState) -> dict[str, Any]:
        hypotheses = state.get("hypotheses", [])
        evidence = state.get("evidence", [])

        if not hypotheses:
            return {
                "root_cause": None,
                "confidence": 0.0,
                "impact": "Unknown",
                "rejected_hypotheses": [],
                "evaluated_hypotheses": [],
                "finding": "No hypotheses available for evaluation.",
            }

        evaluated_hypotheses = []

        for hypothesis in hypotheses:
            cause = hypothesis.get("cause", "Unknown")
            supporting, contradicting = self._evaluate_evidence(
                cause,
                evidence,
            )

            scores = self._calculate_scores(
                supporting,
                contradicting,
            )

            evaluated_hypotheses.append({
                "id": hypothesis.get("id", "UNKNOWN"),
                "cause": cause,
                **scores,
                "supporting_evidence": supporting,
                "contradicting_evidence": contradicting,
            })

        # Rank candidates by heuristic confidence, then by evidence
        # count. Stable sorting preserves the input order for ties.
        evaluated_hypotheses.sort(
            key=lambda item: (
                item["confidence"],
                item["support_count"],
                -item["contradiction_count"],
            ),
            reverse=True,
        )

        for rank, hypothesis in enumerate(
            evaluated_hypotheses,
            start=1,
        ):
            hypothesis["rank"] = rank

        best_hypothesis = evaluated_hypotheses[0]

        rejected_hypotheses = []

        for hypothesis in evaluated_hypotheses[1:]:
            if hypothesis["support_count"] == 0:
                reason = (
                    "No matching supporting evidence was found."
                )
            elif hypothesis["contradiction_count"] > 0:
                reason = (
                    "Ranked lower due to weaker support and/or "
                    "contradicting evidence."
                )
            else:
                reason = (
                    "Ranked lower than the leading hypothesis "
                    "under the current heuristic scoring method."
                )

            rejected_hypotheses.append({
                "id": hypothesis["id"],
                "cause": hypothesis["cause"],
                "reason": reason,
                "confidence": hypothesis["confidence"],
            })

        impact = self._determine_impact(state)

        if best_hypothesis["support_count"] == 0:
            root_cause = None
            confidence = 0.0
            finding = (
                "No candidate has matching supporting evidence. "
                "The root cause remains undetermined."
            )
        else:
            root_cause = best_hypothesis["cause"]
            confidence = best_hypothesis["confidence"]
            finding = (
                f"Ranked {best_hypothesis['id']} "
                f"({root_cause}) first with heuristic score "
                f"{confidence}."
            )

        return {
            "root_cause": root_cause,
            "confidence": confidence,
            "impact": impact,
            "rejected_hypotheses": rejected_hypotheses,
            "evaluated_hypotheses": evaluated_hypotheses,
            "finding": finding,
        }

    def _determine_impact(self, state: RCAState) -> str:
        """Determine incident impact from abnormal metrics."""
        abnormal_metrics = state.get("abnormal_metrics", [])

        if (
            "latency" in abnormal_metrics
            and "error_rate" in abnormal_metrics
        ):
            return (
                "High impact: service latency and error rate "
                "are significantly elevated."
            )

        if "error_rate" in abnormal_metrics:
            return (
                "High impact: elevated error rate may affect "
                "request success."
            )

        if "latency" in abnormal_metrics:
            return (
                "Medium impact: elevated latency may degrade "
                "service responsiveness."
            )

        return "Impact could not be determined confidently."
