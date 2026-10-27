from typing import TypedDict, List, Dict, Any


class RCAState(TypedDict, total=False):
    # -------------------------
    # Incident information
    # -------------------------
    incident_id: str
    anomaly_detected: bool
    timestamp: str
    affected_service: str
    severity: str
    anomaly_score: float
    abnormal_metrics: List[str]
    anomaly_values: Dict[str, float]

    # -------------------------
    # Investigation information
    # -------------------------
    evidence: List[Dict[str, Any]]
    hypotheses: List[Dict[str, Any]]
    rejected_hypotheses: List[Dict[str, Any]]
    investigation_history: List[Dict[str, Any]]

    # -------------------------
    # Agent / cost tracking
    # -------------------------
    agents_called: List[str]
    tool_calls: int
    estimated_tokens: int

    # -------------------------
    # RCA result
    # -------------------------
    root_cause: str
    confidence: float
    impact: str

    # -------------------------
    # Remediation
    # -------------------------
    remediation_action: str
    remediation_result: str
    verification_result: str