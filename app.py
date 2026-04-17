import uuid
from datetime import datetime

import pandas as pd
import streamlit as st

from src.audit_logger import AuditLogger
from src.explanation import ExplanationEngine
from src.governance import GovernanceEngine
from src.llm_client import LLMClient
from src.schemas import (
    AuditEvent,
    ModelMetadata,
    RequestData,
    ResponseData,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Audit Trail Generator",
    page_icon="📋",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []
if "audit_logger" not in st.session_state:
    st.session_state.audit_logger = AuditLogger()


# ---------------------------------------------------------------------------
# Cached component initialisation (survives re-runs)
# ---------------------------------------------------------------------------
@st.cache_resource
def init_components():
    return LLMClient(), ExplanationEngine(), GovernanceEngine()


llm_client, explanation_engine, governance_engine = init_components()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📋 Audit Controls")
    st.metric("Session ID", st.session_state.session_id)

    with st.expander("Model Settings"):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, step=0.05)
        max_tokens = st.slider("Max Tokens", 100, 2000, 800, step=100)
        llm_client.model_metadata.temperature = temperature
        llm_client.model_metadata.max_tokens = max_tokens

    system_prompt = st.text_area(
        "System Prompt",
        value=(
            "You are a governed assistant for analytics QA. "
            "Provide clear, evidence-based responses."
        ),
        height=100,
    )

    with st.expander("Audit Log Viewer"):
        if st.button("Refresh — show latest entry"):
            events = st.session_state.audit_logger.read_events(limit=100)
            if events:
                st.json(events[-1])
            else:
                st.info("No events logged yet.")

    with st.expander("Download Logs"):
        events_all = st.session_state.audit_logger.read_events()
        if events_all:
            df_all = pd.json_normalize(events_all)
            st.download_button(
                label="Download CSV",
                data=df_all.to_csv(index=False),
                file_name=f"audit_log_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
        else:
            st.caption("No log entries yet.")

    with st.expander("Statistics"):
        if st.button("Show Stats"):
            st.json(st.session_state.audit_logger.get_statistics())

# ---------------------------------------------------------------------------
# Main chat interface
# ---------------------------------------------------------------------------
st.title("🤖 AI Audit Trail Generator")
st.caption(
    f"Session: **{st.session_state.session_id}** — "
    "Every interaction is audited for compliance."
)

# Replay chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "audit_id" in msg:
            with st.expander("📋 View Audit Trail"):
                st.caption(f"Audit ID: `{msg['audit_id']}`")
                if msg.get("risk_flags"):
                    st.warning(f"Risk Flags: {', '.join(msg['risk_flags'])}")
                if msg.get("pii_detected"):
                    st.error("🔒 PII detected — interaction flagged.")
                if msg.get("explanation"):
                    st.info(msg["explanation"][:300])

# Chat input
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating response…"):
            # Exclude the current user turn from history
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]

            response_text, latency_ms, _token_usage = llm_client.generate(
                user_prompt=prompt,
                system_prompt=system_prompt,
                history=history,
            )

            st.markdown(response_text)

            explanation = explanation_engine.generate_rationale(
                user_prompt=prompt,
                response_text=response_text,
                context_docs=[],
                tool_calls=[],
            )

            governance = governance_engine.analyze(prompt, response_text)

            audit_event = AuditEvent(
                session_id=st.session_state.session_id,
                model=ModelMetadata(
                    provider="ollama",
                    model_name=llm_client.model_metadata.model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
                request=RequestData(
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    context_documents=[],
                ),
                response=ResponseData(
                    text=response_text,
                    latency_ms=latency_ms,
                ),
                explanation=explanation,
                governance=governance,
            )

            success = st.session_state.audit_logger.write_event(audit_event)

            if success:
                st.caption(
                    f"✅ Audited | ID: `{audit_event.audit_id[:8]}` | "
                    f"Latency: {latency_ms} ms"
                )
            else:
                st.error("❌ Failed to write audit log.")

            if governance.risk_flags:
                st.warning(f"⚠️ Risk Flags: {', '.join(governance.risk_flags)}")
            if governance.pii_detected:
                st.error("🔒 PII Detected — this interaction has been flagged.")

            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "audit_id": audit_event.audit_id,
                "risk_flags": governance.risk_flags,
                "pii_detected": governance.pii_detected,
                "explanation": explanation.rationale_summary,
            })

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "🔒 Compliance Ready: every interaction is logged with model metadata, "
    "observable explanations, and governance flags. "
    "Audit logs are stored in JSONL format."
)
