# AI Audit Trail Generator

A compliance-ready Streamlit app that creates auditable JSONL records for every AI interaction.

## Features

- Every interaction logged with full model metadata
- Grounded explanations based solely on observable inputs/outputs
- Governance flags for risk domains (financial, medical, HR, legal) and PII
- Append-only JSONL audit log for compliance review
- Download logs as CSV directly from the sidebar
- Runs entirely with free, local, open-source tools (Ollama)

## Quick Start

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Pull the model via Ollama**
```bash
ollama pull llama3.1:8b
```

**3. Run the app**
```bash
streamlit run app.py
```

**4. Run tests**
```bash
pytest tests/ -v
```

## Project Structure

```
ai-audit-trail/
├── app.py                    # Streamlit UI
├── src/
│   ├── schemas.py            # Pydantic data models
│   ├── llm_client.py         # Ollama wrapper
│   ├── explanation.py        # Observable-evidence rationale engine
│   ├── governance.py         # Rule-based PII & risk detection
│   └── audit_logger.py       # Append-only JSONL logger
├── tests/
│   └── test_audit_system.py  # Unit tests
├── logs/                     # Created automatically at runtime
│   └── audit_log.jsonl
└── artifacts/explanations/   # AIX360-style JSON artifacts
```

## Audit Event Schema

Each line in `logs/audit_log.jsonl` is a JSON object with these top-level keys:

| Field | Description |
|---|---|
| `audit_id` | UUID4, unique per interaction |
| `timestamp_utc` | ISO 8601 naive UTC |
| `session_id` | 8-char hex, resets on "New Session" |
| `model` | Provider, model name, temperature, max_tokens |
| `request` | System prompt, user prompt, context documents |
| `response` | Response text, latency in ms |
| `explanation` | Rationale summary, evidence used, artifact path |
| `governance` | Risk flags, PII detected, policy status |
