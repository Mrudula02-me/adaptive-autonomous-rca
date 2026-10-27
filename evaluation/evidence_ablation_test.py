import contextlib
import io
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import main

SCENARIOS_DIR = PROJECT_ROOT / "data" / "scenarios"
GROUND_TRUTH_FILE = PROJECT_ROOT / "evaluation" / "ground_truth.json"
RESULTS_FILE = PROJECT_ROOT / "evaluation" / "evidence_ablation_results.csv"

NO_INCIDENT = "No incident"
INCIDENT_SCENARIOS = [
    "database_failure",
    "network_failure",
    "application_failure",
]
EVIDENCE_FILES = {
    "logs": "simulated_logs.csv",
    "traces": "simulated_traces.csv",
    "database": "simulated_database.csv",
}


def normalize_label(label):
    if label is None:
        return NO_INCIDENT
    text = str(label).strip()
    if text.lower() in {
        "no incident", "no_incident",
        "normal operation", "normal_operation", "none",
    }:
        return NO_INCIDENT
    return text


def neutralize_evidence(path, source):
    """Replace one evidence source with neutral observations."""
    data = pd.read_csv(path)

    if source == "logs":
        data["level"] = "INFO"
        data["message"] = "Payment request processed successfully"
    elif source == "traces":
        if "payment_latency" in data.columns:
            data["payment_latency"] = 120.0
        if "database_latency" in data.columns:
            data["database_latency"] = 80.0
    elif source == "database":
        if "connection_errors" in data.columns:
            data["connection_errors"] = 0
        if "query_latency" in data.columns:
            data["query_latency"] = 80.0
        if "active_connections" in data.columns:
            data["active_connections"] = 30

    data.to_csv(path, index=False)


def extract_ranked_candidates(result):
    """Get candidate causes ordered by critic confidence."""
    hypotheses = result.get("evaluated_hypotheses", [])
    if not isinstance(hypotheses, list):
        return []

    valid = [
        item for item in hypotheses
        if isinstance(item, dict) and item.get("cause")
    ]
    ranked = sorted(
        valid,
        key=lambda item: float(item.get("confidence") or 0.0),
        reverse=True,
    )
    return [normalize_label(item["cause"]) for item in ranked]


def run_case(scenario, source, reference):
    """Run one ablation case and always restore the original file."""
    path = SCENARIOS_DIR / scenario / EVIDENCE_FILES[source]
    if not path.is_file():
        raise FileNotFoundError(f"Evidence file not found: {path}")

    original_bytes = path.read_bytes()

    try:
        neutralize_evidence(path, source)

        with contextlib.redirect_stdout(io.StringIO()):
            result = main({
                "scenario": scenario,
                "incident_id": reference["incident_id"],
                "affected_service": reference.get(
                    "affected_service", "payment-service"
                ),
            })

        expected = normalize_label(reference["ground_truth"])
        if not result.get("anomaly_detected", False):
            predicted = NO_INCIDENT
        else:
            predicted = normalize_label(result.get("root_cause"))

        ranked_causes = extract_ranked_candidates(result)
        top1_correct = (
            ranked_causes[0] == expected if ranked_causes else None
        )
        top3_correct = (
            expected in ranked_causes[:3] if ranked_causes else None
        )

        return {
            "scenario": scenario,
            "neutralized_source": source,
            "expected": expected,
            "predicted": predicted,
            "correct": predicted == expected,
            "ranked_candidates": " > ".join(ranked_causes),
            "candidate_count": len(ranked_causes),
            "top1_correct": top1_correct,
            "top3_correct": top3_correct,
            "anomaly_detected": bool(result.get("anomaly_detected", False)),
            "report_status": result.get("report_status", "UNKNOWN"),
            "confidence": float(result.get("confidence") or 0.0),
            "investigation_operations": int(result.get("tool_calls", 0)),
        }

    finally:
        path.write_bytes(original_bytes)


def print_topk_metric(results, column, label):
    """Exclude cases with no ranked candidate list from Top-K metrics."""
    valid = results[column].dropna()
    if valid.empty:
        print(f"{label}: unavailable; no ranked candidates returned.")
        return
    correct_count = int(valid.astype(bool).sum())
    total = len(valid)
    accuracy = valid.astype(bool).mean() * 100
    print(f"{label}: {accuracy:.1f}% ({correct_count}/{total} cases)")


def main_test():
    with GROUND_TRUTH_FILE.open("r", encoding="utf-8") as file:
        ground_truth = json.load(file)

    rows = []
    print("\n========== EVIDENCE ABLATION EVALUATION ==========")

    for scenario in INCIDENT_SCENARIOS:
        if scenario not in ground_truth:
            raise KeyError(f"Missing ground truth for {scenario}")

        for source in EVIDENCE_FILES:
            print(f"Testing {scenario} without {source} evidence ...")
            row = run_case(scenario, source, ground_truth[scenario])
            rows.append(row)

            status = "CORRECT" if row["correct"] else "INCORRECT"
            print(
                f"  Expected: {row['expected']} | "
                f"Predicted: {row['predicted']} | {status}"
            )
            print(
                f"  Status: {row['report_status']} | "
                f"Confidence: {row['confidence']:.3f}"
            )
            print(
                "  Ranked candidates: "
                + (row["ranked_candidates"] or "none returned")
            )

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_FILE, index=False)

    print("\n========== SUMMARY ==========")
    print(f"Cases evaluated: {len(results)}")
    print(
        f"Final-prediction accuracy: "
        f"{results['correct'].mean() * 100:.1f}%"
    )

    print("\nTop-K accuracy:")
    print_topk_metric(results, "top1_correct", "Top-1 accuracy")
    print_topk_metric(results, "top3_correct", "Top-3 accuracy")

    print("\nAccuracy by removed evidence source:")
    print(
        results.groupby("neutralized_source")["correct"]
        .mean().mul(100).round(1).to_string()
    )

    print(f"\nDetailed results saved to: {RESULTS_FILE}")
    print("Original evidence files have been restored.")


if __name__ == "__main__":
    main_test()
