"""
app.py — AgriBot Streamlit UI
Wires all loaded models → agents → SupervisorBrain → chat interface.

Run
---
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st
from PIL import Image

# ── Page config (must be first Streamlit call) ─────────────
st.set_page_config(
    page_title="AgriBot — Precision Agriculture AI",
    page_icon="🌾",
    layout="wide",
)


# ── Lazy-load everything behind st.cache_resource ──────────
@st.cache_resource(show_spinner="Loading AI models…")
def load_all():
    """
    Loads all models once per server process and caches them.
    Returns the fully-initialised SupervisorBrain.
    """
    import torch
    from models.models_loader import (
        DEVICE,
        load_soil_models,
        load_soil_vision_model,
        load_agronomy_strategy_models,
        load_weather_models,
        load_pathology_vision_models,
    )
    from agents.soil    import SoilIntelligenceAgent, MasterAgronomyAgent
    from agents.weather import MicroClimateAgent
    from agents.vision  import CropPathologyAgent
    from core.memory    import MemoryNode
    from core.supervisor import SupervisorBrain

    # ── Load model weights ──────────────────────────────────
    soil_weights      = load_soil_models()
    vision_model      = load_soil_vision_model()
    agronomy_weights  = load_agronomy_strategy_models()
    weather_weights   = load_weather_models()
    pathology_weights = load_pathology_vision_models()

    # ── Instantiate agents ──────────────────────────────────
    soil_agent     = SoilIntelligenceAgent(soil_weights)
    agronomy_agent = MasterAgronomyAgent(vision_model, agronomy_weights)
    weather_agent  = MicroClimateAgent(
        models_dict={
            "gru":  weather_weights["gru"],
            "lstm": weather_weights["lstm"],
            "cnn":  weather_weights["cnn"],
        },
        scaler=weather_weights["scaler"],
        device=DEVICE,
    )

    # Vision agent — Team Delta's tomato pathology model
    try:
        vision_agent = CropPathologyAgent(pathology_weights)
    except Exception:
        vision_agent = None

    # ── Memory ──────────────────────────────────────────────
    try:
        memory = MemoryNode()
    except Exception:
        memory = None

    # ── Supervisor ──────────────────────────────────────────
    brain = SupervisorBrain(
        agents={
            "soil":     soil_agent,
            "weather":  weather_agent,
            "vision":   vision_agent,
            "agronomy": agronomy_agent,
        },
        memory=memory,
    )
    return brain


brain = load_all()


# ── Session state ───────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "tool_log" not in st.session_state:
    st.session_state.tool_log = []
if "queued_message" not in st.session_state:
    st.session_state.queued_message = None


# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.title("🌾 AgriBot")
    st.caption("Agentic AI Decision Support System")
    st.divider()

    st.subheader("📷 Tomato Vision Scanner")
    uploaded_img = st.file_uploader(
        "Upload a tomato photo",
        type=["jpg", "jpeg", "png"],
        key="leaf_upload",
    )

    if uploaded_img:
        import tempfile, os
        img = Image.open(uploaded_img)
        st.image(img, caption="Uploaded leaf", use_container_width=True)

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".jpg", dir="."
        ) as tmp:
            img.save(tmp.name)
            tmp_path = tmp.name

        if st.button("🔍 Scan Tomato"):
            with st.spinner("Running vision model…"):
                result = brain.chat(
                    f"Analyze this tomato image. image_path={tmp_path}",
                    history=None,
                    verbose=False,
                )
            os.unlink(tmp_path)
            st.session_state.queued_message = (
                f"I scanned a tomato photo. The result: {result['reply']} "
                "What should I do next?"
            )
            st.rerun()

    st.divider()
    st.subheader("🔧 Tool Activity")
    if st.session_state.tool_log:
        for t in st.session_state.tool_log[-10:]:
            st.code(t, language=None)
    else:
        st.caption("No tools called yet.")

    if st.button("🗑️ Clear chat"):
        st.session_state.messages  = []
        st.session_state.tool_log  = []
        st.session_state.queued_message = None
        st.rerun()


# ── Chat ────────────────────────────────────────────────────
st.title("AgriBot — Precision Agriculture AI")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.session_state.pop("queued_message", None) or st.chat_input(
    "Ask about soil health, weather forecast, or rice disease…"
)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = brain.chat(
                prompt,
                history=st.session_state.messages[:-1] or None,
                verbose=False,
            )

        reply = result["reply"]

        # Safety block banner
        if result["safety_blocked"]:
            st.error(f"⛔ Action blocked by safety guardrail:\n{result['safety_details']}")

        st.markdown(reply)

        if result["tool_calls_made"]:
            tools_used = [t["name"] for t in result["tool_calls_made"]]
            st.caption(f"🔧 Tools called: {', '.join(tools_used)}")
            st.session_state.tool_log.extend(tools_used)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()