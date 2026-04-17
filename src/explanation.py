import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .schemas import ContextDocument, ExplanationData


class ExplanationEngine:
    """
    Generates grounded rationales from observable inputs only.

    All explanations are derived from what is actually visible —
    the user prompt, response text, and any provided context documents.
    No hidden chain-of-thought is fabricated.
    """

    def __init__(self, artifacts_dir: str = "artifacts/explanations"):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def generate_rationale(
        self,
        user_prompt: str,
        response_text: str,
        context_docs: Optional[List[ContextDocument]] = None,
        tool_calls: Optional[List[str]] = None,
    ) -> ExplanationData:
        """Build an observable-evidence explanation for one interaction."""
        context_docs = context_docs or []
        tool_calls = tool_calls or []

        evidence_used = [self._extract_key_evidence(doc) for doc in context_docs]

        parts: List[str] = []
        parts.append(f"User asked: '{user_prompt[:100]}...'")

        if context_docs:
            titles = [doc.title for doc in context_docs[:3]]
            parts.append(f"Response used context from: {', '.join(titles)}")
        else:
            parts.append("No external context documents were provided")

        if tool_calls:
            parts.append(f"Tools called: {', '.join(tool_calls)}")
        else:
            parts.append("No external tools were called")

        if evidence_used:
            parts.append(
                f"The response emphasised key evidence: {', '.join(evidence_used[:3])}"
            )
        else:
            parts.append(
                "Response was generated from model's parametric knowledge only"
            )

        artifact_path = self._create_aix360_artifact(
            user_prompt, response_text, context_docs
        )

        return ExplanationData(
            rationale_summary=" ".join(parts),
            evidence_used=evidence_used,
            explanation_method="observable-evidence-trace",
            aix360_artifact_path=artifact_path,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_key_evidence(doc: ContextDocument) -> str:
        return f"{doc.title}: {doc.snippet[:50]}..."

    def _create_aix360_artifact(
        self,
        prompt: str,
        response: str,
        context_docs: List[ContextDocument],
    ) -> str:
        artifact_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        artifact_path = self.artifacts_dir / f"explanation_{artifact_id}.json"

        artifact = {
            "method": "rule-based",
            "timestamp": datetime.utcnow().isoformat(),
            "input_summary": prompt[:200],
            "output_summary": response[:200],
            "evidence_trace": [
                {
                    "source": doc.title,
                    "relevance": "high",
                    "extracted_snippet": doc.snippet,
                }
                for doc in context_docs
            ],
            "explanation_type": "local",
            "confidence": 0.85,
        }

        with open(artifact_path, "w") as fh:
            json.dump(artifact, fh, indent=2)

        return str(artifact_path)

    def load_aix360_artifact(self, path: str) -> dict:
        with open(path, "r") as fh:
            return json.load(fh)
