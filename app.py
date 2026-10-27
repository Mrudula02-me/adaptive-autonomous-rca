
import streamlit as st
import pandas as pd
from datetime import datetime

from main import main


st.set_page_config(
    page_title="Adaptive RCA Platform",
    page_icon="🔎",
    layout="wide",
)

st.title("Adaptive Autonomous Multi-Agent RCA Platform")
st.caption(
    "Investigate incidents using anomaly detection, "
    "observability evidence, and multi-agent reasoning."
)

SCENARIOS = {
    "Database Failure": "database_failure",
    "Network Failure": "network_failure",
    "Application Failure": "application_failure",
    "Normal Operation": "normal_operation",
}

with st.sidebar:
    st.header("Incident Configuration")

    selected_label = st.selectbox(
        "Select scenario",
        list(SCENARIOS.keys()),
    )

    scenario = SCENARIOS[selected_label]

    incident_id = st.text_input(
        "Incident ID",
        value="INC-DEMO-001",
    )

    affected_service = st.text_input(
        "Affected service",
        value="payment-service",
    )

    st.info(
        "The dashboard runs the existing RCA pipeline. "
        "Results depend on the configured scenario data "
        "and the current agent implementation."
    )

    run_clicked = st.button(
        "Run RCA Investigation",
        type="primary",
        use_container_width=True,
    )


def display_value(value, default="Not available"):
    if value is None or value == "":
        return default
    return value


if run_clicked:
    if not incident_id.strip() or not affected_service.strip():
        st.error("Enter an incident ID and affected service.")
        st.stop()

    try:
        with st.spinner("Agents are investigating the incident..."):
            result = main({
                "scenario": scenario,
                "incident_id": incident_id.strip(),
                "affected_service": affected_service.strip(),
            })

        if not isinstance(result, dict):
            st.error(
                "The RCA pipeline did not return a dictionary. "
                "Check the return value of main()."
            )
            st.stop()

        st.session_state["rca_result"] = result
        st.session_state["rca_scenario"] = selected_label
        st.session_state["rca_incident_id"] = incident_id.strip()

    except Exception as exc:
        st.error(f"Investigation failed: {exc}")
        st.exception(exc)


result = st.session_state.get("rca_result")

if result:
    st.divider()
    st.subheader("Investigation Results")

    anomaly_detected = result.get("anomaly_detected", False)
    severity = result.get("severity", "UNKNOWN")
    root_cause = result.get("root_cause")
    confidence = result.get("confidence", 0.0)
    report_status = result.get("report_status", "UNKNOWN")
    impact = result.get("impact", "Not available")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Anomaly detected",
        "Yes" if anomaly_detected else "No",
    )
    col2.metric("Severity", str(severity))
    col3.metric(
        "Report status",
        str(report_status),
    )

    try:
        confidence_percent = float(confidence) * 100
        col4.metric(
            "Leading confidence score",
            f"{confidence_percent:.1f}%",
        )
    except (ValueError, TypeError):
        col4.metric("Leading confidence score", "N/A")

    st.subheader("Root Cause Assessment")

    if root_cause:
        st.write(f"**Leading hypothesis:** {root_cause}")
    else:
        st.warning(
            "No supported root cause was returned by the pipeline."
        )

    st.write(f"**Impact:** {impact}")

    st.subheader("Ranked Hypotheses")

    hypotheses = result.get("evaluated_hypotheses", [])

    if hypotheses:
        ranked_rows = []

        for index, hypothesis in enumerate(hypotheses, start=1):
            score = hypothesis.get("confidence", 0.0)

            ranked_rows.append({
                "Rank": hypothesis.get("rank", index),
                "Hypothesis": hypothesis.get("cause", "Unknown"),
                "Score": score,
                "Supporting evidence": hypothesis.get(
                    "support_count", 0
                ),
                "Contradicting evidence": hypothesis.get(
                    "contradiction_count", 0
                ),
            })

        st.dataframe(
            pd.DataFrame(ranked_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "No evaluated hypothesis list was returned. "
            "Check whether main() preserves evaluated_hypotheses."
        )

    st.subheader("Collected Evidence")

    evidence = result.get("evidence", [])

    if evidence:
        st.dataframe(
            pd.DataFrame(evidence),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No evidence records were returned.")

    with st.expander("Investigation Details"):
        st.write("**Scenario:**", st.session_state.get("rca_scenario"))
        st.write(
            "**Incident ID:**",
            st.session_state.get("rca_incident_id"),
        )
        st.write(
            "**Agents called:**",
            result.get("agents_called", []),
        )
        st.write(
            "**Investigation operations:**",
            result.get("tool_calls", 0),
        )
        st.write(
            "**Abnormal metrics:**",
            result.get("abnormal_metrics", []),
        )
        st.write(
            "**Reporting reason:**",
            result.get("reporting_reason", "Not available"),
        )

    # Export the returned state as a JSON file.
    import json

    st.download_button(
        "Download RCA report (JSON)",
        data=json.dumps(result, indent=2, default=str),
        file_name=f"{incident_id.strip()}_rca_report.json",
        mime="application/json",
    )

else:
    st.info(
        "Choose a scenario from the sidebar and click "
        "'Run RCA Investigation' to begin."
    )

    st.subheader("Available Scenarios")

    left, right = st.columns(2)

    with left:
        st.markdown("**Database Failure**")
        st.write("Investigate database query latency and connection issues.")

        st.markdown("**Application Failure**")
        st.write("Investigate potential internal service errors.")

    with right:
        st.markdown("**Network Failure**")
        st.write("Investigate communication and dependency problems.")

        st.markdown("**Normal Operation**")
        st.write("Check whether the system avoids reporting a false incident.")
