import re
from typing import Dict, List

from .schemas import GovernanceData


class GovernanceEngine:
    """
    Rule-based governance checks for AI interactions.

    Detects high-risk content domains and PII patterns using
    keyword matching and regex — no ML required for MVP.
    """

    _RISK_DOMAINS: Dict[str, List[str]] = {
        "financial_decision": [
            "credit", "loan", "mortgage", "investment", "banking", "risk score",
        ],
        "medical_content": [
            "diagnosis", "treatment", "prescription", "medical", "health condition",
        ],
        "hr_decision": [
            "hiring", "firing", "promotion", "performance review", "salary",
        ],
        "legal_advice": [
            "legal", "lawsuit", "attorney", "court", "contract law",
        ],
    }

    _PII_PATTERNS: Dict[str, str] = {
        "email":       r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "phone":       r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ssn":         r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
    }

    def analyze(self, prompt: str, response: str) -> GovernanceData:
        combined = (prompt + " " + response).lower()

        risk_flags: List[str] = []
        for domain, keywords in self._RISK_DOMAINS.items():
            if any(kw in combined for kw in keywords):
                risk_flags.append(domain)

        pii_detected = any(
            re.search(pattern, prompt) or re.search(pattern, response)
            for pattern in self._PII_PATTERNS.values()
        )

        if risk_flags:
            policy_status = "requires_review"
        elif pii_detected:
            policy_status = "restricted"
        else:
            policy_status = "reviewable"

        return GovernanceData(
            risk_flags=risk_flags,
            pii_detected=pii_detected,
            policy_status=policy_status,
        )
