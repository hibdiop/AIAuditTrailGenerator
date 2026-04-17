from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import List, Optional
import uuid


class ModelMetadata(BaseModel):
    provider: str = "ollama"
    model_name: str = "llama3.1:8b"
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    max_tokens: int = 800


class ContextDocument(BaseModel):
    doc_id: str
    title: str
    snippet: str


class RequestData(BaseModel):
    system_prompt: str
    user_prompt: str
    context_documents: List[ContextDocument] = []


class ResponseData(BaseModel):
    text: str
    latency_ms: int


class ExplanationData(BaseModel):
    rationale_summary: str
    evidence_used: List[str] = []
    explanation_method: str = "rule-based summary + retrieved evidence trace"
    aix360_artifact_path: Optional[str] = None


class GovernanceData(BaseModel):
    risk_flags: List[str] = []
    pii_detected: bool = False
    policy_status: str = "reviewable"


class AuditEvent(BaseModel):
    audit_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_utc: datetime = Field(default_factory=datetime.utcnow)
    session_id: str
    user_id: str = "demo_user"
    app_version: str = "0.1.0"
    model: ModelMetadata
    request: RequestData
    response: ResponseData
    explanation: ExplanationData
    governance: GovernanceData

    @field_validator("timestamp_utc")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        # Store as naive UTC
        return v.replace(tzinfo=None)
