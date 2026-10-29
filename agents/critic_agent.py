
from typing import Any

from state.rca_state import RCAState


class CriticAgent:
    """
    Evaluate and rank competing root-cause hypotheses.

    Confidence is a heuristic score, not a calibrated probability.
    Cause-specific evidence is preferred over generic symptoms.
    """

    DATABASE_CAUSE = "Database-side failure"
    NETWORK_CAUSE = "Network or dependency communication failure"
    APPLICATION_CAUSE = "Payment-service internal failure"
    RESOURCE_CAUSE = "Resource exhaustion"
    TRAFFIC_CAUSE = "Traffic overload"

    DATABASE_ERROR_TYPES = {
        "database_timeout",
        "database_connection_error",
        "database_error",
    }

    NETWORK_ERROR_TYPES = {
        "dependency_timeout",
        "network_timeout",
        "connection_refused",
        "communication_failure",
    }

    APPLICATION_ERROR_TYPES = {
        "application_exception",
        "unhandled_exception",
        "internal_application_error",
    }

    @staticmethod
    def _text(item: dict[str, Any]) -> str:
        """
        Combine relevant evidence fields into searchable text.

        Structured trace fields are included so that cause-specific
        trace evidence can be evaluated by the critic.
        """

        fields = (
            "finding",
            "message",
            "metric",
            "name",
            "service",
            "database",
            "error_type",
            "span_status",
            "downstream_service",
            "source_service",
            "target_service",
        )

        return " ".join(
            str(item.get(field, "") or "")
            for field in fields
        ).lower()

    @staticmethod
    def _normalize(value: Any) -> str:
        """Normalize structured labels for consistent comparisons."""

        return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

    def _evaluate_evidence(
        self,
        cause: str,
        evidence: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Separate evidence supporting and contradicting a candidate."""

        supporting = []
        contradicting = []

        for item in evidence:
            evidence_type = self._normalize(item.get("type", ""))
            text = self._text(item)
            error_type = self._normalize(item.get("error_type", ""))

            # H1: Database-side failure
            if cause == self.DATABASE_CAUSE:
                if evidence_type == "database":
                    if any(
                        phrase in text
                        for phrase in (
                            "query latency",
                            "connection errors",
                            "active database connections",
                            "database timeout",
                            "database connection",
                        )
                    ):
                        supporting.append(item)

                elif evidence_type == "logs":
                    if any(
                        phrase in text
                        for phrase in (
                            "database connection",
                            "database timeout",
                            "database query",
                            "database connection error",
                            "database error",
                        )
                    ):
                        supporting.append(item)

                elif evidence_type == "traces":
                    if (
                        "database latency remains normal" in text
                        or "database latency is normal" in text
                        or (
                            "database_latency" in item
                            and self._is_normal_database_latency(item)
                        )
                    ):
                        contradicting.append(item)

                    elif (
                        error_type in self.DATABASE_ERROR_TYPES
                        or any(
                            phrase in text
                            for phrase in (
                                "high database latency",
                                "database latency elevated",
                                "database query latency",
                                "database timeout",
                            )
                        )
                    ):
                        supporting.append(item)

            # H2: Network or dependency communication failure
            elif cause == self.NETWORK_CAUSE:
                if evidence_type == "logs":
                    if any(
                        phrase in text
                        for phrase in (
                            "network timeout",
                            "communication timeout",
                            "connection refused",
                            "communication failure",
                            "dependency timeout",
                            "dependency communication failure",
                            "dependency connection failure",
                        )
                    ):
                        supporting.append(item)

                elif evidence_type == "traces":
                    if (
                        error_type in self.NETWORK_ERROR_TYPES
                        or any(
                            phrase in text
                            for phrase in (
                                "network timeout",
                                "dependency timeout",
                                "connection refused",
                                "communication failure",
                            )
                        )
                    ):
                        supporting.append(item)

                    elif (
                        "dependency_latency" in item
                        and self._is_high_dependency_latency(item)
                        and self._is_normal_database_latency(item)
                    ):
                        supporting.append(item)

                elif evidence_type == "database":
                    if any(
                        phrase in text
                        for phrase in (
                            "query latency",
                            "connection errors",
                            "high database latency",
                            "database timeout",
                        )
                    ):
                        contradicting.append(item)

            # H3: Payment-service internal failure
            elif cause == self.APPLICATION_CAUSE:
                if evidence_type == "logs":
                    if any(
                        phrase in text
                        for phrase in (
                            "unhandled exception",
                            "application exception",
                            "internal application error",
                            "payment-service exception",
                        )
                    ):
                        supporting.append(item)

                elif evidence_type == "traces":
                    if (
                        error_type in self.APPLICATION_ERROR_TYPES
                        or any(
                            phrase in text
                            for phrase in (
                                "application exception",
                                "unhandled exception",
                                "internal application error",
                            )
                        )
                    ):
                        supporting.append(item)

                    elif (
                        "dependency_latency" in item
                        and self._is_normal_database_latency(item)
                        and self._is_normal_dependency_latency(item)
                        and self._has_error_span(item)
                    ):
                        supporting.append(item)

                elif evidence_type == "database":
                    if any(
                        phrase in text
                        for phrase in (
                            "query latency",
                            "connection errors",
                            "high database latency",
                            "database timeout",
                        )
                    ):
                        contradicting.append(item)

            # H4: Resource exhaustion
            elif cause == self.RESOURCE_CAUSE:
                if any(
                    phrase in text
                    for phrase in (
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
                    and any(word in text for word in ("cpu", "memory"))
                ):
                    supporting.append(item)

            # H5: Traffic overload
            elif cause == self.TRAFFIC_CAUSE:
                if any(
                    phrase in text
                    for phrase in (
                        "traffic overload",
                        "request overload",
                        "high request rate",
                        "too many requests",
                        "request saturation",
                        "traffic spike",
                    )
                ):
                    supporting.append(item)

        return supporting, contradicting

    @staticmethod
    def _numeric_value(item: dict[str, Any], field: str) -> float | None:
        """Safely read a numeric value from evidence."""

        try:
            value = float(item[field])
        except (KeyError, TypeError, ValueError):
            return None

        if value != value or value in (float("inf"), float("-inf")):
            return None

        return value

    @classmethod
    def _is_normal_database_latency(
        cls,
        item: dict[str, Any],
    ) -> bool:
        value = cls._numeric_value(item, "database_latency")
        return value is not None and value <= 500

    @classmethod
    def _is_high_dependency_latency(
        cls,
        item: dict[str, Any],
    ) -> bool:
        value = cls._numeric_value(item, "dependency_latency")
        return value is not None and value > 500

    @classmethod
    def _is_normal_dependency_latency(
        cls,
        item: dict[str, Any],
    ) -> bool:
        value = cls._numeric_value(item, "dependency_latency")
        return value is not None and value <= 500

    @staticmethod
    def _has_error_span(item: dict[str, Any]) -> bool:
        return str(item.get("span_status", "")).strip().upper() in {
            "ERROR",
            "FAILED",
        }

    @staticmethod
    def _calculate_scores(
        supporting: list[dict[str, Any]],
        contradicting: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Calculate a heuristic confidence score.

        This score is not a statistically calibrated probability.
        """

        modality_types = {
            str(item.get("type", "")).lower()
            for item in supporting
            if item.get("type")
        }

        support_count = len(supporting)
        contradiction_count = len(contradicting)
        modality_count = len(modality_types)

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
        """Evaluate hypotheses and select the best-supported candidate."""

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

            evaluated_hypotheses.append(
                {
                    "id": hypothesis.get("id", "UNKNOWN"),
                    "cause": cause,
                    **scores,
                    "supporting_evidence": supporting,
                    "contradicting_evidence": contradicting,
                }
            )

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
                reason = "No matching supporting evidence was found."
            elif hypothesis["contradiction_count"] > 0:
                reason = (
                    "Ranked lower due to weaker support and/or "
                    "contradicting evidence."
                )
            else:
                reason = (
                    "Ranked lower under the current heuristic "
                    "scoring method."
                )

            rejected_hypotheses.append(
                {
                    "id": hypothesis["id"],
                    "cause": hypothesis["cause"],
                    "reason": reason,
                    "confidence": hypothesis["confidence"],
                }
            )

        impact = self._determine_impact(state)

        # Do not select a root cause without supporting evidence.
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
        """Describe likely impact using detected abnormal metrics."""

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
