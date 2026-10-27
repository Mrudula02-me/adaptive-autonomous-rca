
from typing import Any

import pandas as pd

from models.anomaly_detector import AnomalyDetector


class AnomalyAgent:
    """
    Detect statistical anomalies and identify actionable incidents.

    The agent does not determine the root cause.
    """

    def __init__(self):
        self.detector = AnomalyDetector()

    def analyze(
        self,
        data: pd.DataFrame,
        target_service: str = "payment-service",
    ) -> dict[str, Any]:
        """
        Analyze only the requested service.

        A statistical outlier is considered actionable only when
        at least one metric crosses its configured threshold.
        """

        if "service" not in data.columns:
            raise ValueError(
                "Observability data must contain a 'service' column."
            )

        # 1. Restrict analysis to the intended service.
        service_data = data.loc[
            data["service"] == target_service
        ].copy()

        if service_data.empty:
            return {
                "anomaly_detected": False,
                "message": (
                    f"No observations found for {target_service}."
                ),
                "affected_service": target_service,
                "abnormal_metrics": [],
                "anomaly_count": 0,
            }

        # 2. Fit and run the detector on this service's observations.
        self.detector.train(service_data)
        results = self.detector.detect(service_data)

        # 3. Keep statistical outliers.
        outliers = results.loc[
            results["anomaly_label"] == -1
        ].copy()

        if outliers.empty:
            return self._no_incident(
                target_service,
                "No statistical anomalies detected.",
            )

        # 4. Check whether each outlier has actionable metric signals.
        outliers["abnormal_metrics"] = outliers.apply(
            self._identify_abnormal_metrics,
            axis=1,
        )

        actionable = outliers.loc[
            outliers["abnormal_metrics"].map(bool)
        ].copy()

        # 5. Do not create an incident from an outlier alone.
        if actionable.empty:
            return self._no_incident(
                target_service,
                (
                    "Statistical outliers were found, but none "
                    "crossed the configured incident thresholds."
                ),
                statistical_outlier_count=len(outliers),
            )

        # 6. Choose the most anomalous actionable observation.
        incident_row = actionable.loc[
            actionable["anomaly_score"].idxmax()
        ]

        abnormal_metrics = incident_row["abnormal_metrics"]

        return {
            "anomaly_detected": True,
            "timestamp": str(incident_row["timestamp"]),
            "affected_service": target_service,
            "anomaly_score": float(
                incident_row["anomaly_score"]
            ),
            "severity": self._calculate_severity(
                abnormal_metrics
            ),
            "abnormal_metrics": abnormal_metrics,
            "values": {
                metric: float(incident_row[metric])
                for metric in (
                    "cpu",
                    "memory",
                    "latency",
                    "error_rate",
                    "request_rate",
                )
            },
            "anomaly_count": len(actionable),
            "statistical_outlier_count": len(outliers),
            "message": (
                "An actionable anomaly was detected using "
                "statistical and metric-threshold evidence."
            ),
        }

    @staticmethod
    def _no_incident(
        service: str,
        message: str,
        statistical_outlier_count: int = 0,
    ) -> dict[str, Any]:
        return {
            "anomaly_detected": False,
            "affected_service": service,
            "abnormal_metrics": [],
            "anomaly_count": 0,
            "statistical_outlier_count": statistical_outlier_count,
            "message": message,
        }

    @staticmethod
    def _identify_abnormal_metrics(
        observation: pd.Series,
    ) -> list[str]:
        """Apply the current proof-of-concept thresholds."""

        abnormal = []

        if observation["cpu"] > 80:
            abnormal.append("cpu")

        if observation["memory"] > 85:
            abnormal.append("memory")

        if observation["latency"] > 500:
            abnormal.append("latency")

        if observation["error_rate"] > 5:
            abnormal.append("error_rate")

        return abnormal

    @staticmethod
    def _calculate_severity(
        abnormal_metrics: list[str],
    ) -> str:
        """
        Assign severity based on the number and type
        of abnormal metrics, rather than the raw anomaly score.
        """

        if (
            "error_rate" in abnormal_metrics
            and "latency" in abnormal_metrics
        ):
            return "CRITICAL"

        if len(abnormal_metrics) >= 2:
            return "HIGH"

        return "MEDIUM"
