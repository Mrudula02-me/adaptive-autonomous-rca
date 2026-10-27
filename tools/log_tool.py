import pandas as pd


class LogTool:
    """
    Tool used to retrieve logs for a specific service.
    """

    def __init__(self, log_file="data/simulated_logs.csv"):
        self.log_file = log_file

    def query_logs(self, service: str) -> pd.DataFrame:
        """
        Retrieve logs belonging to the requested service.
        """

        logs = pd.read_csv(self.log_file)

        service_logs = logs[
            logs["service"] == service
        ].copy()

        return service_logs