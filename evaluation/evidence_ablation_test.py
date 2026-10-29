
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

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


# ============================================================
# 2. LABEL NORMALIZATION
# ============================================================

def normalize_label(label: Any) -> str:
    """Normalize equivalent labels before comparing predictions."""
    if label is None:
        return NO_INCIDENT

    text = str(label).strip()

    if text.lower() in {
        "no incident",
        "no_incident",
        "normal operation",
        "normal_operation",
        "none",
    }:
        return NO_INCIDENT

    return text


# ============================================================
# 3. EVIDENCE NEUTRALIZATION
# ============================================================

def neutralize_evidence(path: Path, source: str) -> None:
    """
    Replace one evidence source with neutral observations.

    This changes only the selected evidence CSV. The original
    file is restored by run_case(), even if the test fails.
    """
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

    else:
        raise ValueError(f"Unknown evidence source: {source}")

    data.to_csv(path, index=False)


# ============================================================
# 4. HYPOTHESIS DIAGNOSTICS
# ============================================================

def extract_ranked_hypotheses(
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return evaluated hypotheses sorted by confidence."""
    hypotheses = result.get("evaluated_hypotheses", [])

    if not isinstance(hypotheses, list):
        return []

    valid = [
        item
        for item in hypotheses
        if isinstance(item, dict) and item.get("cause")
    ]

    return sorted(
        valid,
        key=lambda item: (
            float(item.get("confidence") or 0.0),
            int(item.get("support_count") or 0),
            -int(item.get("contradiction_count") or 0),
        ),
        reverse=True,
    )


def safe_json(value: Any) -> str:
    """Serialize diagnostic values for CSV storage."""
    return json.dumps(value, ensure_ascii=False, default=str)


# ============================================================
# 5. RUN ONE ABLATION CASE
# ============================================================

def run_case(
    scenario: str,
    source: str,
    reference: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute one experiment and restore its original evidence file.

    Captures main() output to help explain incorrect predictions.
    """
    path = SCENARIOS_DIR / scenario / EVIDENCE_FILES[source]

    if not path.is_file():
        raise FileNotFoundError(f"Evidence file not found: {path}")

    # Keep the exact original bytes for safe restoration.
    original_bytes = path.read_bytes()
    captured_output = io.StringIO()

    try:
        neutralize_evidence(path, source)

        with contextlib.redirect_stdout(captured_output):
            result = main({
                "scenario": scenario,
                "incident_id": reference["incident_id"],
                "affected_service": reference.get(
                    "affected_service",
                    "payment-service",
                ),
            })

        if not isinstance(result, dict):
            raise TypeError(
                "main() must return a dictionary; received "
                f"{type(result).__name__}."
            )

        expected = normalize_label(reference["ground_truth"])
        anomaly_detected = bool(result.get("anomaly_detected", False))

        # Keep detection and diagnosis as separate evaluation stages.
        if not anomaly_detected:
            predicted = NO_INCIDENT
        elif not result.get("root_cause"):
            predicted = "Undetermined"
        else:
            predicted = normalize_label(result["root_cause"])

        ranked = extract_ranked_hypotheses(result)
        ranked_causes = [
            normalize_label(item["cause"])
            for item in ranked
        ]

        top1_correct = (
            ranked_causes[0] == expected if ranked_causes else None
        )
        top3_correct = (
            expected in ranked_causes[:3] if ranked_causes else None
        )

        leading = ranked[0] if ranked else {}

        row = {
            # Experiment identity
            "scenario": scenario,
            "incident_id": reference["incident_id"],
            "neutralized_source": source,
            "expected": expected,
            "predicted": predicted,
            "correct": predicted == expected,

            # Stage 1: anomaly detection
            "anomaly_detected": anomaly_detected,
            "anomaly_score": result.get("anomaly_score"),
            "severity": result.get("severity"),
            "abnormal_metrics": safe_json(
                result.get("abnormal_metrics", [])
            ),
            "anomaly_values": safe_json(
                result.get("anomaly_values", {})
            ),
            "detector_message": result.get("message", ""),

            # Stage 2: investigation
            "agents_called": safe_json(
                result.get("agents_called", [])
            ),
            "investigation_operations": int(
                result.get("tool_calls", 0)
            ),
            "evidence_count": len(result.get("evidence", [])),
            "evidence": safe_json(result.get("evidence", [])),
            "investigation_history": safe_json(
                result.get("investigation_history", [])
            ),

            # Stage 3: critic and final reporting
            "root_cause": result.get("root_cause"),
            "confidence": float(result.get("confidence") or 0.0),
            "report_status": result.get("report_status", "UNKNOWN"),
            "reported_root_cause": result.get("reported_root_cause"),
            "provisional_hypothesis": result.get(
                "provisional_hypothesis"
            ),
            "reporting_reason": result.get("reporting_reason", ""),

            # Hypothesis ranking
            "leading_hypothesis": leading.get("cause"),
            "leading_confidence": float(
                leading.get("confidence") or 0.0
            ),
            "leading_support_count": leading.get("support_count"),
            "leading_contradiction_count": leading.get(
                "contradiction_count"
            ),
            "ranked_candidates": " > ".join(ranked_causes),
            "candidate_count": len(ranked_causes),
            "top1_correct": top1_correct,
            "top3_correct": top3_correct,
        }

        # Store full workflow output only for incorrect predictions.
        if not row["correct"]:
            row["debug_output"] = captured_output.getvalue()

        return row

    finally:
        # Restore the selected evidence file even if an exception occurs.
        path.write_bytes(original_bytes)


# ============================================================
# 6. FAILURE DIAGNOSTICS
# ============================================================

def print_failure_diagnostics(row: dict[str, Any]) -> None:
    """Print information to distinguish detection from RCA failures."""
    print("\n  ---------- FAILURE DIAGNOSTICS ----------")
    print(f"  Expected: {row['expected']}")
    print(f"  Predicted: {row['predicted']}")
    print(f"  Anomaly detected: {row['anomaly_detected']}")
    print(f"  Detector message: {row['detector_message']}")
    print(f"  Anomaly score: {row['anomaly_score']}")
    print(f"  Severity: {row['severity']}")
    print(f"  Abnormal metrics: {row['abnormal_metrics']}")
    print(f"  Agents called: {row['agents_called']}")
    print(f"  Investigation operations: {row['investigation_operations']}")
    print(f"  Evidence count: {row['evidence_count']}")
    print(f"  Root cause: {row['root_cause']}")
    print(f"  Report status: {row['report_status']}")
    print(f"  Confidence: {row['confidence']:.3f}")
    print(f"  Leading hypothesis: {row['leading_hypothesis']}")
    print(f"  Reporting reason: {row['reporting_reason']}")

    debug_output = row.get("debug_output", "")

    if debug_output:
        print("\n  Captured main() output:")
        print(debug_output.rstrip())

    print("  -----------------------------------------")


# ============================================================
# 7. SUMMARY METRICS
# ============================================================

def print_topk_metric(
    results: pd.DataFrame,
    column: str,
    label: str,
) -> None:
    """Calculate Top-K accuracy for cases with ranked candidates."""
    valid = results[column].dropna()

    if valid.empty:
        print(f"{label}: unavailable; no ranked candidates.")
        return

    correct_count = int(valid.astype(bool).sum())
    total = len(valid)
    accuracy = valid.astype(bool).mean() * 100

    print(f"{label}: {accuracy:.1f}% ({correct_count}/{total})")


def print_grouped_accuracy(
    results: pd.DataFrame,
    group_column: str,
    metric_column: str,
    heading: str,
) -> None:
    """Print percentages grouped by evidence source."""
    print(f"\n{heading}")

    grouped = (
        results.groupby(group_column)[metric_column]
        .mean()
        .mul(100)
        .round(1)
    )

    print(grouped.to_string())


# ============================================================
# 8. RUN THE COMPLETE EXPERIMENT
# ============================================================

def main_test() -> None:
    with GROUND_TRUTH_FILE.open("r", encoding="utf-8") as file:
        ground_truth = json.load(file)

    rows = []

    print("\n========== EVIDENCE ABLATION EVALUATION ==========")

    for scenario in INCIDENT_SCENARIOS:
        if scenario not in ground_truth:
            raise KeyError(f"Missing ground truth for {scenario}")

        for source in EVIDENCE_FILES:
            print(f"\nTesting {scenario} without {source} evidence ...")

            row = run_case(
                scenario,
                source,
                ground_truth[scenario],
            )
            rows.append(row)

            status = "CORRECT" if row["correct"] else "INCORRECT"

            print(
                f"  Expected: {row['expected']} | "
                f"Predicted: {row['predicted']} | {status}"
            )
            print(
                f"  Anomaly detected: {row['anomaly_detected']} | "
                f"Status: {row['report_status']} | "
                f"Confidence: {row['confidence']:.3f}"
            )
            print(
                "  Ranked candidates: "
                + (row["ranked_candidates"] or "none returned")
            )

            if not row["correct"]:
                print_failure_diagnostics(row)

    results = pd.DataFrame(rows)

    # Save detailed diagnostics for later analysis.
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

    print_grouped_accuracy(
        results,
        "neutralized_source",
        "correct",
        "Final-prediction accuracy by neutralized source:",
    )

    print_grouped_accuracy(
        results,
        "neutralized_source",
        "anomaly_detected",
        "Anomaly-detection rate by neutralized source:",
    )

    print_grouped_accuracy(
        results,
        "neutralized_source",
        "top1_correct",
        "Top-1 hypothesis accuracy by neutralized source:",
    )

    print(f"\nDetailed results saved to: {RESULTS_FILE}")
    print("Original evidence files have been restored.")


if __name__ == "__main__":
    main_test()
