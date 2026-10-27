
from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd


# ============================================================
# 1. Project configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SCENARIOS_DIR = DATA_DIR / "scenarios"

SERVICES = [
    "api-gateway",
    "auth-service",
    "payment-service",
    "order-service",
]

START_TIME = pd.Timestamp("2026-10-01 10:00:00")
INCIDENT_START = pd.Timestamp("2026-10-01 10:07:00")
INCIDENT_END = pd.Timestamp("2026-10-01 10:12:00")

N_SAMPLES = 200
RANDOM_SEED = 42


# Ground truth is stored separately from scenario observations.
# Do not pass this file to RCA agents.
SCENARIOS = {
    "database_failure": {
        "incident_id": "INC-DB-001",
        "affected_service": "payment-service",
        "ground_truth": "Database-side failure",
    },
    "network_failure": {
        "incident_id": "INC-NET-001",
        "affected_service": "payment-service",
        "ground_truth": "Network or dependency communication failure",
    },
    "application_failure": {
        "incident_id": "INC-APP-001",
        "affected_service": "payment-service",
        "ground_truth": "Payment-service internal failure",
    },
    "normal_operation": {
        "incident_id": "INC-NORMAL-001",
        "affected_service": "payment-service",
        "ground_truth": "No incident",
    },
}


# ============================================================
# 2. Generate baseline metrics
# ============================================================

def generate_service_data(
    service: str,
    rng: np.random.Generator,
    n_samples: int = N_SAMPLES,
) -> pd.DataFrame:
    """Generate normal metric observations for one service."""

    timestamps = pd.date_range(
        start=START_TIME,
        periods=n_samples,
        freq="min",
    )

    return pd.DataFrame({
        "timestamp": timestamps,
        "service": service,
        "cpu": rng.normal(45, 5, n_samples).clip(5, 100),
        "memory": rng.normal(55, 5, n_samples).clip(5, 100),
        "latency": rng.normal(120, 15, n_samples).clip(10, None),
        "error_rate": rng.normal(0.5, 0.15, n_samples).clip(0, None),
        "request_rate": rng.normal(500, 30, n_samples).clip(0, None),
    })


def inject_metric_incident(
    data: pd.DataFrame,
    scenario: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Inject a scenario-specific pattern into a copy of the metrics."""

    data = data.copy()

    if scenario == "normal_operation":
        return data

    affected_service = SCENARIOS[scenario]["affected_service"]

    mask = (
        (data["service"] == affected_service)
        & (data["timestamp"] >= INCIDENT_START)
        & (data["timestamp"] <= INCIDENT_END)
    )

    count = int(mask.sum())

    if count == 0:
        raise ValueError(
            f"No metric rows found for scenario {scenario!r}. "
            "Check the timestamps and affected service."
        )

    if scenario == "database_failure":
        data.loc[mask, "cpu"] = rng.normal(75, 5, count).clip(0, 100)
        data.loc[mask, "memory"] = rng.normal(80, 5, count).clip(0, 100)
        data.loc[mask, "latency"] = rng.normal(850, 80, count).clip(0, None)
        data.loc[mask, "error_rate"] = rng.normal(15, 2, count).clip(0, None)
        data.loc[mask, "request_rate"] = rng.normal(520, 25, count).clip(0, None)

    elif scenario == "network_failure":
        # Latency and errors rise, while CPU, memory, and request
        # volume remain near their normal levels.
        data.loc[mask, "latency"] = rng.normal(750, 60, count).clip(0, None)
        data.loc[mask, "error_rate"] = rng.normal(10, 1.5, count).clip(0, None)

    elif scenario == "application_failure":
        # A service-local problem causes high resource use,
        # latency, and errors. Database signals remain normal.
        data.loc[mask, "cpu"] = rng.normal(90, 3, count).clip(0, 100)
        data.loc[mask, "memory"] = rng.normal(85, 4, count).clip(0, 100)
        data.loc[mask, "latency"] = rng.normal(700, 60, count).clip(0, None)
        data.loc[mask, "error_rate"] = rng.normal(20, 2, count).clip(0, None)

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return data


def generate_metrics(
    scenario: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate the complete metrics dataset for one scenario."""

    service_frames = [
        generate_service_data(service, rng)
        for service in SERVICES
    ]

    metrics = pd.concat(service_frames, ignore_index=True)

    return inject_metric_incident(metrics, scenario, rng)


# ============================================================
# 3. Generate logs
# ============================================================

def generate_logs(scenario: str) -> pd.DataFrame:
    """Generate logs whose messages reflect the selected scenario."""

    timestamps = pd.date_range(
        start=START_TIME,
        periods=15,
        freq="min",
    )

    logs = []

    for minute, timestamp in enumerate(timestamps):
        is_incident_window = 7 <= minute <= 12

        if not is_incident_window or scenario == "normal_operation":
            level = "INFO"
            message = "Payment request processed successfully"

        elif scenario == "database_failure":
            level = "ERROR"
            message = "Database connection timeout"

        elif scenario == "network_failure":
            level = "ERROR"
            message = "Dependency communication timeout"

        elif scenario == "application_failure":
            level = "ERROR"
            message = "Unhandled exception in payment-service"

        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        logs.append({
            "timestamp": timestamp,
            "service": "payment-service",
            "level": level,
            "message": message,
        })

    return pd.DataFrame(logs)


# ============================================================
# 4. Generate distributed traces
# ============================================================

def generate_traces(scenario: str) -> pd.DataFrame:
    """Generate traces with different dependency latency patterns."""

    traces = []

    for minute in range(15):
        timestamp = START_TIME + pd.Timedelta(minutes=minute)
        is_incident_window = 7 <= minute <= 12

        payment_latency = 120
        database_latency = 80

        if is_incident_window:
            if scenario == "database_failure":
                payment_latency = 850
                database_latency = 800

            elif scenario == "network_failure":
                payment_latency = 750
                database_latency = 80

            elif scenario == "application_failure":
                payment_latency = 700
                database_latency = 80

        traces.append({
            "timestamp": timestamp,
            "trace_id": f"{scenario}-trace-{minute:03d}",
            "source_service": "api-gateway",
            "target_service": "payment-service",
            "payment_latency": payment_latency,
            "database_latency": database_latency,
        })

    return pd.DataFrame(traces)


# ============================================================
# 5. Generate database signals
# ============================================================

def generate_database_data(scenario: str) -> pd.DataFrame:
    """Generate database signals appropriate to each scenario."""

    timestamps = pd.date_range(
        start=START_TIME,
        periods=15,
        freq="min",
    )

    database_rows = []

    for minute, timestamp in enumerate(timestamps):
        is_incident_window = 7 <= minute <= 12

        if scenario == "database_failure" and is_incident_window:
            connection_errors = 8
            query_latency = 780
            active_connections = 95
        else:
            # Database stays healthy in network, application,
            # and normal-operation scenarios.
            connection_errors = 0
            query_latency = 80
            active_connections = 30

        database_rows.append({
            "timestamp": timestamp,
            "database": "payment-db",
            "connection_errors": connection_errors,
            "query_latency": query_latency,
            "active_connections": active_connections,
        })

    return pd.DataFrame(database_rows)


# ============================================================
# 6. Write one scenario's observability datasets
# ============================================================

def generate_scenario(
    scenario: str,
    seed: int,
) -> dict:
    """Generate and save all observability data for one scenario."""

    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")

    rng = np.random.default_rng(seed)
    scenario_dir = SCENARIOS_DIR / scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)

    metrics = generate_metrics(scenario, rng)
    logs = generate_logs(scenario)
    traces = generate_traces(scenario)
    database = generate_database_data(scenario)

    datasets = {
        "simulated_observability.csv": metrics,
        "simulated_logs.csv": logs,
        "simulated_traces.csv": traces,
        "simulated_database.csv": database,
    }

    for filename, dataframe in datasets.items():
        dataframe.to_csv(scenario_dir / filename, index=False)

    # Preserve compatibility with the existing main.py and agents:
    # the root-level files continue to represent database_failure.
    if scenario == "database_failure":
        for filename in datasets:
            shutil.copy2(
                scenario_dir / filename,
                DATA_DIR / filename,
            )

    print(f"\nScenario: {scenario}")
    print(f"Metrics:  {metrics.shape}")
    print(f"Logs:     {logs.shape}")
    print(f"Traces:   {traces.shape}")
    print(f"Database: {database.shape}")
    print(f"Saved to: {scenario_dir}")

    return {
        "scenario": scenario,
        "directory": str(scenario_dir),
        "metrics_rows": len(metrics),
        "logs_rows": len(logs),
        "traces_rows": len(traces),
        "database_rows": len(database),
    }


# ============================================================
# 7. Main generation pipeline
# ============================================================

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []

    for index, scenario in enumerate(SCENARIOS):
        # Separate reproducible seed for each scenario.
        seed = RANDOM_SEED + index

        summaries.append(generate_scenario(scenario, seed))

    # Ground-truth labels are stored separately from each
    # scenario's observability files. Agents must not read this.
    evaluation_dir = PROJECT_ROOT / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    ground_truth = {
        scenario: {
            "incident_id": details["incident_id"],
            "affected_service": details["affected_service"],
            "ground_truth": details["ground_truth"],
        }
        for scenario, details in SCENARIOS.items()
    }

    ground_truth_path = evaluation_dir / "ground_truth.json"

    with ground_truth_path.open("w", encoding="utf-8") as file:
        json.dump(ground_truth, file, indent=4)

    print("\n========== DATA GENERATION COMPLETE ==========")
    print(f"Scenarios generated: {len(summaries)}")
    print(f"Ground truth saved separately: {ground_truth_path}")
    print("Root-level CSV files remain set to database_failure.")
    print("===============================================")


if __name__ == "__main__":
    main()
