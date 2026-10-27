
import pandas as pd
from sklearn.ensemble import IsolationForest


FEATURES = [
    "cpu",
    "memory",
    "latency",
    "error_rate",
    "request_rate",
]


class AnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42,
        )

    def train(self, data: pd.DataFrame) -> None:
        """Fit Isolation Forest to the supplied observations."""
        self._validate_data(data)

        self.model.fit(data[FEATURES])

    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Return observations with anomaly labels and scores.

        anomaly_label:
            -1 = statistical outlier
             1 = not classified as an outlier

        anomaly_score:
            Higher values indicate observations that are
            more anomalous relative to the fitted model.
        """
        self._validate_data(data)

        results = data.copy()
        X = results[FEATURES]

        results["anomaly_label"] = self.model.predict(X)
        results["anomaly_score"] = (
            -self.model.decision_function(X)
        )

        return results

    @staticmethod
    def _validate_data(data: pd.DataFrame) -> None:
        """Check that the model receives usable metric data."""
        if data.empty:
            raise ValueError(
                "Cannot detect anomalies in an empty dataset."
            )

        missing = set(FEATURES) - set(data.columns)

        if missing:
            raise ValueError(
                f"Missing required metric columns: {sorted(missing)}"
            )

        if data[FEATURES].isnull().any().any():
            raise ValueError(
                "Metric features contain missing values."
            )

        if not data[FEATURES].apply(
            lambda column: pd.api.types.is_numeric_dtype(column)
        ).all():
            raise ValueError(
                "All metric features must be numeric."
            )
