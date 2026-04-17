import json
from pathlib import Path
from typing import List, Optional

from .schemas import AuditEvent


class AuditLogger:
    """Append-only JSONL audit logger with schema validation."""

    def __init__(self, log_path: str = "logs/audit_log.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def write_event(self, event: AuditEvent) -> bool:
        """Validate and append one audit event to the JSONL file."""
        try:
            record = event.model_dump()
            # Serialize datetime to ISO string for JSON
            record["timestamp_utc"] = event.timestamp_utc.isoformat()

            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            return True
        except Exception as e:
            print(f"[AuditLogger] Failed to write event: {e}")
            return False

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def read_events(self, limit: Optional[int] = None) -> List[dict]:
        """Return events from the log file, oldest first."""
        if not self.log_path.exists():
            return []

        events: List[dict] = []
        with open(self.log_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if limit is not None and i >= limit:
                    break
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return events

    def search_events(self, query: str) -> List[dict]:
        """Case-insensitive search across user prompt and response text."""
        q = query.lower()
        return [
            e for e in self.read_events()
            if q in e.get("request", {}).get("user_prompt", "").lower()
            or q in e.get("response", {}).get("text", "").lower()
        ]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def get_statistics(self) -> dict:
        events = self.read_events()
        if not events:
            return {"total_interactions": 0}

        risk_counts: dict = {}
        for e in events:
            for flag in e.get("governance", {}).get("risk_flags", []):
                risk_counts[flag] = risk_counts.get(flag, 0) + 1

        return {
            "total_interactions": len(events),
            "first_interaction": events[0].get("timestamp_utc"),
            "last_interaction": events[-1].get("timestamp_utc"),
            "risk_flags_summary": risk_counts,
            "pii_incidents": sum(
                1 for e in events
                if e.get("governance", {}).get("pii_detected")
            ),
        }
