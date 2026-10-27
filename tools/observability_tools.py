
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_DIR = PROJECT_ROOT / "data" / "scenarios"


class ObservabilityTools:
    """
    Read observability data from a selected RCA scenario.

    Supported scenario directories:
        database_failure
        network_failure
        application_failure
        normal_operation
    """

    def __init__(self, scenario="database_failure"):
        self.scenario = scenario
        self.scenario_dir = SCENARIOS_DIR / scenario

        if not self.scenario_dir.is_dir():
            available = (
                sorted(
                    path.name
                    for path in SCENARIOS_DIR.iterdir()
                    if path.is_dir()
                )
                if SCENARIOS_DIR.is_dir()
                else []
            )

            raise FileNotFoundError(
                f"Scenario directory not found: {self.scenario_dir}\n"
                f"Available scenarios: {available}\n"
                "Run tools/generate_data.py first."
            )

        self.metrics_file = (
            self.scenario_dir / "simulated_observability.csv"
        )
        self.logs_file = self.scenario_dir / "simulated_logs.csv"
        self.traces_file = self.scenario_dir / "simulated_traces.csv"
        self.database_file = (
            self.scenario_dir / "simulated_database.csv"
        )

    def _read_csv(self, file_path):
        """Read a scenario CSV and report a clear error if it is missing."""

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Observability dataset not found: {file_path}"
            )

        return pd.read_csv(file_path)

    def query_metrics(self, service):
        """Retrieve metric observations for a specific service."""

        data = self._read_csv(self.metrics_file)

        return data.loc[data["service"] == service].copy()

    def query_logs(self, service):
        """Retrieve logs for a specific service."""

        data = self._read_csv(self.logs_file)

        return data.loc[data["service"] == service].copy()

    def query_traces(self, service):
        """Retrieve traces involving a particular service."""

        data = self._read_csv(self.traces_file)

        mask = (
            (data["source_service"] == service)
            | (data["target_service"] == service)
        )

        return data.loc[mask].copy()

    def query_database(self, database):
        """Retrieve database observability signals."""

        data = self._read_csv(self.database_file)

        return data.loc[data["database"] == database].copy()
