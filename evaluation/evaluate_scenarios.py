
import contextlib
import io
import json
import sys
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATH CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import main

GROUND_TRUTH_FILE = (
    PROJECT_ROOT / "evaluation" / "ground_truth.json"
)

RESULTS_FILE = (
    PROJECT_ROOT / "evaluation" / "scenario_evaluation_results.csv"
)


# --------------------------------------------------
# 2. LABEL NORMALIZATION
# --------------------------------------------------

def normalize_label(label):
    """Normalize equivalent labels for fair comparison."""
    if label is None:
        return "No incident"

    text = str(label).strip()

    if text.lower() in {
        "no incident",
        "no_incident",
        "normal operation",
        "normal_operation",
        "none",
    }:
        return "No incident"

    return text


# --------------------------------------------------
# 3. NETWORK HYPOTHESIS DIAGNOSTICS
# --------------------------------------------------

def print_hypothesis_details(result):
    """Print critic scores and evidence for every hypothesis."""
    print("\n--- NETWORK FAILURE: HYPOTHESIS DETAILS ---")

    hypotheses = result.get("evaluated_hypotheses", [])

    if not hypotheses:
        print("No evaluated hypotheses were returned.")
        return

    for hypothesis in hypotheses:
        print(f"\nRank: {hypothesis.get('rank', 'N/A')}")
        print(f"ID: {hypothesis.get('id', 'N/A')}")
        print(f"Cause: {hypothesis.get('cause', 'Unknown')}")
        print(
            f"Confidence: {hypothesis.get('confidence', 0.0)}"
        )
        print(
            f"Supporting evidence count: "
            f"{hypothesis.get('support_count', 0)}"
        )
        print(
            f"Contradiction count: "
            f"{hypothesis.get('contradiction_count', 0)}"
        )
        print(
            f"Evidence modalities: "
            f"{hypothesis.get('modality_count', 0)}"
        )

        print("Supporting evidence:")
        supporting = hypothesis.get("supporting_evidence", [])

        if supporting:
            for item in supporting:
                description = (
                    item.get("finding")
                    or item.get("message")
                    or item.get("metric")
                    or "No description available"
                )

                print(
                    f"  - Type: {item.get('type', 'unknown')}"
                    f" | Finding: {description}"
                )
        else:
            print("  None")

        print("Contradicting evidence:")
        contradicting = hypothesis.get("contradicting_evidence", [])

        if contradicting:
            for item in contradicting:
                description = (
                    item.get("finding")
                    or item.get("message")
                    or item.get("metric")
                    or "No description available"
                )

                print(
                    f"  - Type: {item.get('type', 'unknown')}"
                    f" | Finding: {description}"
                )
        else:
            print("  None")


# --------------------------------------------------
# 4. EVALUATE SCENARIOS
# --------------------------------------------------

def evaluate_scenarios():
    """Run the RCA pipeline against all ground-truth scenarios."""

    with GROUND_TRUTH_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        ground_truth = json.load(file)

    rows = []

    for scenario, reference in ground_truth.items():
        print(f"Evaluating scenario: {scenario}")

        incident_config = {
            "scenario": scenario,
            "incident_id": reference["incident_id"],
            "affected_service": reference.get(
                "affected_service",
                "payment-service",
            ),
        }

        # Capture the regular output from main() so the evaluation
        # summary remains readable.
        with contextlib.redirect_stdout(io.StringIO()):
            result = main(incident_config)

        # Show detailed scores only for the network scenario.
        if scenario == "network_failure":
            print_hypothesis_details(result)

        # Determine the final predicted label.
        expected = normalize_label(
            reference["ground_truth"]
        )

        if not result.get("anomaly_detected", False):
            predicted = "No incident"
        else:
            predicted = normalize_label(
                result.get("root_cause")
            )

        rows.append({
            "scenario": scenario,
            "incident_id": reference["incident_id"],
            "expected": expected,
            "predicted": predicted,
            "correct": predicted == expected,
            "anomaly_detected": bool(
                result.get("anomaly_detected", False)
            ),
            "report_status": result.get(
                "report_status",
                "UNKNOWN",
            ),
            "confidence": float(
                result.get("confidence") or 0.0
            ),
            "investigation_operations": int(
                result.get("tool_calls", 0)
            ),
            "agents_called": ", ".join(
                result.get("agents_called", [])
            ),
        })

    # --------------------------------------------------
    # 5. SAVE RESULTS
    # --------------------------------------------------

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_FILE, index=False)

    # --------------------------------------------------
    # 6. EVALUATION SUMMARY
    # --------------------------------------------------

    print("\n========== SCENARIO EVALUATION SUMMARY ==========")

    total_cases = len(results)
    correct_cases = int(results["correct"].sum())

    print(f"Cases evaluated: {total_cases}")
    print(
        f"Correct predictions: "
        f"{correct_cases}/{total_cases}"
    )

    if total_cases > 0:
        print(
            f"Accuracy: "
            f"{results['correct'].mean() * 100:.1f}%"
        )

    incident_rows = results[
        results["expected"] != "No incident"
    ]

    if not incident_rows.empty:
        print(
            f"Incident-only accuracy: "
            f"{incident_rows['correct'].mean() * 100:.1f}%"
        )

    normal_rows = results[
        results["expected"] == "No incident"
    ]

    if not normal_rows.empty:
        print(
            f"Normal-operation false-positive rate: "
            f"{normal_rows['anomaly_detected'].mean() * 100:.1f}%"
        )

    print("\nDetailed results:")
    print(results.to_string(index=False))

    print(f"\nResults saved to: {RESULTS_FILE}")


# --------------------------------------------------
# 7. SCRIPT ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    evaluate_scenarios()
