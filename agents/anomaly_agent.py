
from typing import Any

import numpy as np
import pandas as pd

from models.anomaly_detector import AnomalyDetector


class AnomalyAgent:
    """
    Detect actionable anomalies for a target service.

    Isolation Forest identifies statistical outliers.
    Absolute thresholds are the primary incident criteria.
    A joint latency/error-rate baseline rule is a fallback
    for substantial incidents below the absolute thresholds.
    """

    METRIC_THRESHOLDS = {
        "cpu": 80,
        "memory": 85,
        "latency": 500,
        "error_rate": 5,
    }

    REQUIRED_COLUMNS = {
        "service",
        "timestamp",
        "cpu",
        "memory",
        "latency",
        "error_rate",
        "request_rate",
    }

    FEATURE_COLUMNS = [
        "cpu",
        "memory",
        "latency",
        "error_rate",
        "request_rate",
    ]

    BASELINE_WINDOW = 5
    LATENCY_RATIO = 2.5
    ERROR_RATE_RATIO = 3.0
    MAD_MULTIPLIER = 3.0

    # Minimum scales prevent tiny baseline variation from making
    # insignificant changes look anomalous.
    BASELINE_SCALE_FLOORS = {
        "latency": 15.0,
        "error_rate": 0.15,
    }

    def __init__(self) -> None:
        self.detector = AnomalyDetector()

    def analyze(
        self,
        data: pd.DataFrame,
        target_service: str = "payment-service",
    ) -> dict[str, Any]:
        """Analyze observations and return an incident summary."""

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")

        missing_columns = self.REQUIRED_COLUMNS - set(data.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required columns: {sorted(missing_columns)}"
            )

        if not isinstance(target_service, str) or not target_service.strip():
            raise ValueError("target_service must be a non-empty string.")

        # Filter before doing any model fitting or baseline calculations.
        service_data = data.loc[
            data["service"] == target_service
        ].copy()

        if service_data.empty:
            return self._no_incident(
                message=f"No observations found for {target_service}.",
                affected_service=target_service,
            )

        service_data["timestamp"] = pd.to_datetime(
            service_data["timestamp"],
            errors="raise",
        )

        for column in self.FEATURE_COLUMNS:
            service_data[column] = pd.to_numeric(
                service_data[column],
                errors="coerce",
            )

        if service_data[self.FEATURE_COLUMNS].isnull().any().any():
            raise ValueError(
                "Metric columns contain missing or non-numeric values "
                f"for service {target_service}."
            )

        feature_values = service_data[
            self.FEATURE_COLUMNS
        ].to_numpy(dtype=float)

        if not np.isfinite(feature_values).all():
            raise ValueError(
                "Metric columns must contain finite numeric values."
            )

        # Sort chronologically before using previous observations
        # as a baseline.
        service_data = service_data.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        # Isolation Forest still identifies candidate outliers.
        self.detector.train(service_data)
        detected_data = self.detector.detect(service_data)

        statistical_outlier_count = int(
            (detected_data["anomaly_label"] == -1).sum()
        )

        actionable_records = []

        for index, row in detected_data.iterrows():
            if row["anomaly_label"] != -1:
                continue

            # First, use the existing absolute thresholds.
            abnormal_metrics = self._identify_abnormal_metrics(row)

            # Only use the baseline fallback if absolute thresholds
            # did not already identify an actionable metric.
            if not abnormal_metrics:
                abnormal_metrics = self._identify_joint_baseline_incident(
                    service_data,
                    index,
                )

            if abnormal_metrics:
                actionable_records.append((row, abnormal_metrics))

        if not actionable_records:
            if statistical_outlier_count:
                message = (
                    "Statistical outliers detected, but none met "
                    "the absolute thresholds or joint baseline rule."
                )
            else:
                message = "No statistical outliers detected."

            return self._no_incident(
                message=message,
                statistical_outlier_count=statistical_outlier_count,
                affected_service=target_service,
            )

        selected_row, abnormal_metrics = max(
            actionable_records,
            key=lambda item: float(item[0]["anomaly_score"]),
        )

        severity = self._calculate_severity(abnormal_metrics)

        return {
            "anomaly_detected": True,
            "timestamp": str(selected_row["timestamp"]),
            "affected_service": target_service,
            "anomaly_score": float(selected_row["anomaly_score"]),
            "severity": severity,
            "abnormal_metrics": abnormal_metrics,
            "values": {
                metric: float(selected_row[metric])
                for metric in self.FEATURE_COLUMNS
            },
            "anomaly_count": len(actionable_records),
            "statistical_outlier_count": statistical_outlier_count,
            "message": (
                f"{severity} incident detected in {target_service}; "
                f"abnormal metrics: {', '.join(abnormal_metrics)}."
            ),
        }

    def _identify_joint_baseline_incident(
        self,
        service_data: pd.DataFrame,
        index: int,
    ) -> list[str]:
        """
        Detect a joint latency and error-rate increase relative
        to the preceding five observations.

        Both metrics must satisfy their ratio and robust-deviation
        checks. This avoids flagging an incident from an isolated
        change in CPU, memory, or request rate.
        """
        if index < self.BASELINE_WINDOW:
            return []

        baseline = service_data.iloc[
            index - self.BASELINE_WINDOW:index
        ]
        current = service_data.iloc[index]

        checks = {}

        for metric, ratio in (
            ("latency", self.LATENCY_RATIO),
            ("error_rate", self.ERROR_RATE_RATIO),
        ):
            baseline_values = baseline[metric].to_numpy(dtype=float)
            baseline_median = float(np.median(baseline_values))

            # Ratios are not meaningful with a zero or negative baseline.
            if baseline_median <= 0:
                return []

            mad = float(
                np.median(np.abs(baseline_values - baseline_median))
            )
            robust_scale = max(
                1.4826 * mad,
                self.BASELINE_SCALE_FLOORS[metric],
            )

            value = float(current[metric])

            ratio_check = value >= ratio * baseline_median
            deviation_check = (
                value - baseline_median
                > self.MAD_MULTIPLIER * robust_scale
            )

            checks[metric] = ratio_check and deviation_check

        # Require both signals on the same observation.
        if checks.get("latency") and checks.get("error_rate"):
            return ["latency", "error_rate"]

        return []

    def _identify_abnormal_metrics(
        self,
        row: pd.Series,
    ) -> list[str]:
        """Identify metrics exceeding the existing absolute thresholds."""
        return [
            metric
            for metric, threshold in self.METRIC_THRESHOLDS.items()
            if float(row[metric]) > threshold
        ]

    @staticmethod
    def _calculate_severity(
        abnormal_metrics: list[str],
    ) -> str:
        """Classify severity from the number and type of abnormal metrics."""
        abnormal_set = set(abnormal_metrics)

        if {"latency", "error_rate"}.issubset(abnormal_set):
            return "CRITICAL"

        if len(abnormal_set) >= 2:
            return "HIGH"

        if abnormal_set:
            return "MEDIUM"

        return "NONE"

    @staticmethod
    def _no_incident(
        message: str,
        statistical_outlier_count: int = 0,
        affected_service: str | None = None,
    ) -> dict[str, Any]:
        """Return a consistent result when no actionable incident exists."""
        return {
            "anomaly_detected": False,
            "timestamp": None,
            "affected_service": affected_service,
            "anomaly_score": 0.0,
            "severity": "NONE",
            "abnormal_metrics": [],
            "values": {},
            "anomaly_count": 0,
            "statistical_outlier_count": statistical_outlier_count,
            "message": message,
        }
