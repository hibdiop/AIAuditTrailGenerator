import json
import tempfile
from pathlib import Path

import pytest

from src.audit_logger import AuditLogger
from src.explanation import ExplanationEngine
from src.governance import GovernanceEngine
from src.schemas import (
    AuditEvent,
    ContextDocument,
    ExplanationData,
    GovernanceData,
    ModelMetadata,
    RequestData,
    ResponseData,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _minimal_event(session_id: str = "test_session") -> AuditEvent:
    return AuditEvent(
        session_id=session_id,
        model=ModelMetadata(),
        request=RequestData(system_prompt="Test system", user_prompt="Hello"),
        response=ResponseData(text="Hi there", latency_ms=100),
        explanation=ExplanationData(rationale_summary="Test explanation"),
        governance=GovernanceData(),
    )


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------
class TestSchemas:
    def test_audit_event_auto_fields(self):
        event = _minimal_event()
        assert event.audit_id is not None
        assert len(event.audit_id) == 36  # UUID4
        assert event.timestamp_utc is not None

    def test_model_metadata_defaults(self):
        m = ModelMetadata()
        assert m.provider == "ollama"
        assert m.model_name == "llama3.1:8b"
        assert m.temperature == 0.2

    def test_temperature_validation(self):
        with pytest.raises(Exception):
            ModelMetadata(temperature=1.5)  # exceeds max

    def test_governance_defaults(self):
        g = GovernanceData()
        assert g.risk_flags == []
        assert g.pii_detected is False
        assert g.policy_status == "reviewable"

    def test_context_document_fields(self):
        doc = ContextDocument(doc_id="d1", title="Report", snippet="Key finding…")
        assert doc.doc_id == "d1"


# ---------------------------------------------------------------------------
# Governance tests
# ---------------------------------------------------------------------------
class TestGovernanceEngine:
    engine = GovernanceEngine()

    def test_pii_email_detected(self):
        result = self.engine.analyze("My email is alice@example.com", "OK")
        assert result.pii_detected is True

    def test_pii_phone_detected(self):
        result = self.engine.analyze("Call me at 555-123-4567", "OK")
        assert result.pii_detected is True

    def test_no_pii_clean_text(self):
        result = self.engine.analyze("What is the weather today?", "It's sunny.")
        assert result.pii_detected is False

    def test_financial_risk_flag(self):
        result = self.engine.analyze("I need a mortgage loan", "Here are options.")
        assert "financial_decision" in result.risk_flags

    def test_medical_risk_flag(self):
        result = self.engine.analyze("Suggest a diagnosis", "Consult a doctor.")
        assert "medical_content" in result.risk_flags

    def test_clean_text_no_flags(self):
        result = self.engine.analyze("What is 2 + 2?", "Four.")
        assert result.risk_flags == []
        assert result.policy_status == "reviewable"

    def test_policy_status_requires_review_on_risk(self):
        result = self.engine.analyze("hiring decisions for the role", "Noted.")
        assert result.policy_status == "requires_review"

    def test_policy_status_restricted_on_pii_no_risk(self):
        result = self.engine.analyze("My SSN is 123-45-6789", "Noted.")
        # SSN triggers PII but no domain risk → restricted
        assert result.pii_detected is True
        assert result.policy_status in ("restricted", "requires_review")


# ---------------------------------------------------------------------------
# Audit logger tests
# ---------------------------------------------------------------------------
class TestAuditLogger:
    def _logger_with_tmp(self) -> tuple[AuditLogger, Path]:
        tmp = Path(tempfile.mkdtemp()) / "test_audit.jsonl"
        return AuditLogger(str(tmp)), tmp

    def test_write_creates_file(self):
        logger, path = self._logger_with_tmp()
        assert logger.write_event(_minimal_event()) is True
        assert path.exists()

    def test_write_and_read_roundtrip(self):
        logger, _ = self._logger_with_tmp()
        event = _minimal_event("roundtrip_session")
        logger.write_event(event)

        events = logger.read_events()
        assert len(events) == 1
        assert events[0]["audit_id"] == event.audit_id
        assert events[0]["session_id"] == "roundtrip_session"

    def test_multiple_events_ordered(self):
        logger, _ = self._logger_with_tmp()
        for i in range(3):
            logger.write_event(_minimal_event(f"s{i}"))

        events = logger.read_events()
        assert len(events) == 3
        assert [e["session_id"] for e in events] == ["s0", "s1", "s2"]

    def test_read_limit(self):
        logger, _ = self._logger_with_tmp()
        for _ in range(5):
            logger.write_event(_minimal_event())
        assert len(logger.read_events(limit=3)) == 3

    def test_search_events_match(self):
        logger, _ = self._logger_with_tmp()
        event = AuditEvent(
            session_id="search_test",
            model=ModelMetadata(),
            request=RequestData(system_prompt="s", user_prompt="unique_keyword_xyz"),
            response=ResponseData(text="answer", latency_ms=10),
            explanation=ExplanationData(rationale_summary="r"),
            governance=GovernanceData(),
        )
        logger.write_event(event)
        results = logger.search_events("unique_keyword_xyz")
        assert len(results) == 1

    def test_search_events_no_match(self):
        logger, _ = self._logger_with_tmp()
        logger.write_event(_minimal_event())
        assert logger.search_events("zzz_not_present") == []

    def test_statistics_empty(self):
        logger, _ = self._logger_with_tmp()
        stats = logger.get_statistics()
        assert stats["total_interactions"] == 0

    def test_statistics_with_events(self):
        logger, _ = self._logger_with_tmp()
        gov_event = AuditEvent(
            session_id="stats_test",
            model=ModelMetadata(),
            request=RequestData(system_prompt="s", user_prompt="loan application"),
            response=ResponseData(text="r", latency_ms=50),
            explanation=ExplanationData(rationale_summary="e"),
            governance=GovernanceData(risk_flags=["financial_decision"]),
        )
        logger.write_event(gov_event)
        stats = logger.get_statistics()
        assert stats["total_interactions"] == 1
        assert "financial_decision" in stats["risk_flags_summary"]

    def test_jsonl_is_valid_json_per_line(self):
        logger, path = self._logger_with_tmp()
        logger.write_event(_minimal_event())
        lines = path.read_text().strip().splitlines()
        for line in lines:
            json.loads(line)  # must not raise


# ---------------------------------------------------------------------------
# Explanation engine tests
# ---------------------------------------------------------------------------
class TestExplanationEngine:
    def _engine(self) -> ExplanationEngine:
        return ExplanationEngine(artifacts_dir=tempfile.mkdtemp())

    def test_rationale_contains_user_ask(self):
        exp = self._engine().generate_rationale(
            user_prompt="Why is the sky blue?",
            response_text="Rayleigh scattering.",
        )
        assert "User asked" in exp.rationale_summary

    def test_no_context_message(self):
        exp = self._engine().generate_rationale(
            user_prompt="Hi", response_text="Hello"
        )
        assert "No external context" in exp.rationale_summary

    def test_context_docs_appear_in_rationale(self):
        docs = [ContextDocument(doc_id="1", title="Physics 101", snippet="light scattering")]
        exp = self._engine().generate_rationale(
            user_prompt="Why sky blue?",
            response_text="Because of scattering.",
            context_docs=docs,
        )
        assert "Physics 101" in exp.rationale_summary

    def test_tool_calls_appear_in_rationale(self):
        exp = self._engine().generate_rationale(
            user_prompt="Search for X",
            response_text="Found X.",
            tool_calls=["web_search"],
        )
        assert "web_search" in exp.rationale_summary

    def test_explanation_method_set(self):
        exp = self._engine().generate_rationale("q", "a")
        assert exp.explanation_method == "observable-evidence-trace"

    def test_artifact_file_created(self):
        exp = self._engine().generate_rationale("q", "a")
        assert exp.aix360_artifact_path is not None
        assert Path(exp.aix360_artifact_path).exists()

    def test_artifact_is_valid_json(self):
        exp = self._engine().generate_rationale("q", "a")
        with open(exp.aix360_artifact_path) as fh:
            data = json.load(fh)
        assert data["method"] == "rule-based"
        assert "evidence_trace" in data
