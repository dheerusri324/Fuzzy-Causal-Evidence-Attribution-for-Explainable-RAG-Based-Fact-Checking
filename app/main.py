"""
Streamlit UI for End-to-End Fact-Checking Application.
Presentation Demo: Partial Implementation Showcase.
"""

import sys
from pathlib import Path
import streamlit as st

# Ensure the root project directory is in the path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from config.settings import Config
from app.pipeline import FactCheckingPipeline

# Set page config
st.set_page_config(
    page_title="Fuzzy-Causal Evidence Attribution",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def render_in_development_card(heading: str, description: str, status: str = "Under Development"):
    """Renders a polished, dark-themed 'IN DEVELOPMENT' presentation card."""
    card_html = f"""
    <div style="
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 20px 24px;
        margin: 18px 0 24px 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.25), 0 2px 4px -2px rgba(0, 0, 0, 0.2);
    ">
        <div style="
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(51, 65, 85, 0.6);
            padding-bottom: 12px;
            margin-bottom: 14px;
        ">
            <div style="
                display: flex;
                align-items: center;
                gap: 10px;
                font-size: 1.12rem;
                font-weight: 700;
                color: #f1f5f9;
                letter-spacing: 0.5px;
            ">
                <span style="font-size: 1.15rem;">🔒</span>
                <span>{heading}</span>
            </div>
            <div style="
                background: rgba(245, 158, 11, 0.15);
                color: #fbbf24;
                border: 1px solid rgba(245, 158, 11, 0.4);
                padding: 4px 12px;
                border-radius: 9999px;
                font-size: 0.75rem;
                font-weight: 700;
                letter-spacing: 0.5px;
                display: flex;
                align-items: center;
                gap: 6px;
            ">
                <span>🔒</span>
                <span>IN DEVELOPMENT</span>
            </div>
        </div>
        <p style="
            font-size: 0.94rem;
            line-height: 1.6;
            color: #cbd5e1;
            margin: 0 0 16px 0;
        ">
            {description}
        </p>
        <div style="
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.82rem;
            color: #94a3b8;
            font-weight: 500;
        ">
            <span style="
                width: 8px;
                height: 8px;
                background: #f59e0b;
                border-radius: 50%;
                display: inline-block;
            "></span>
            <span>Status: {status}</span>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_project_status_notice():
    """Renders the mandatory project status notice at the bottom of the dashboard."""
    notice_html = """
    <div style="
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid #334155;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 40px;
        margin-bottom: 24px;
    ">
        <div style="
            font-weight: 700;
            font-size: 0.88rem;
            color: #93c5fd;
            letter-spacing: 0.6px;
            margin-bottom: 6px;
        ">
            PROJECT STATUS — PARTIAL IMPLEMENTATION DEMO
        </div>
        <div style="
            font-size: 0.88rem;
            color: #cbd5e1;
            line-height: 1.55;
        ">
            The core fact-checking, evidence retrieval, and attribution components are available for demonstration. Global explanation robustness, fuzzy reliability metrics, natural-language explanation integration, and research experiments including comparative evaluation remain under development.
        </div>
    </div>
    """
    st.markdown(notice_html, unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Load the full pipeline. Cached so models only load once."""
    cfg = Config(top_k=3)
    return FactCheckingPipeline(config=cfg)


st.title("Fuzzy-Causal Fact-Checking & Explanation Pipeline")

# Initialize pipeline
with st.spinner("Loading models and pipeline (this may take a minute on first run)..."):
    pipeline = load_pipeline()

# 1. Claim Input and Run Analysis (Functional Component 1)
claim_input = st.text_input("Enter a claim to fact-check:", "Lincoln Motor Company was named by Henry M. Leland.")
run_button = st.button("Run Analysis")

if run_button and claim_input.strip():
    with st.spinner("Running pipeline (Retrieval -> Verification -> Attribution -> Intervention)..."):
        try:
            res = pipeline.process(claim_input)
            st.session_state["pipeline_result"] = res
            st.session_state["evaluated_claim"] = claim_input
        except Exception as e:
            st.error(f"Pipeline failed: {str(e)}")
            st.stop()

# Retrieve stored pipeline result if available
res = st.session_state.get("pipeline_result")

if res is not None:
    # 2. Fact-Checking Verdict (Functional Component 2)
    st.header("FACT-CHECKING VERDICT")
    st.info(f"**Verdict:** {res.verification.verdict}")
    st.write(
        f"**Probabilities:** SUPPORTS: {res.verification.probabilities.get('SUPPORTS', 0):.4f} | "
        f"REFUTES: {res.verification.probabilities.get('REFUTES', 0):.4f} | "
        f"NEI: {res.verification.probabilities.get('NOT_ENOUGH_INFO', 0):.4f}"
    )

    # 3. Retrieved Evidence (Functional Component 3)
    st.header("Retrieved Evidence")
    for ev_res in res.evidence_results:
        ev = ev_res.evidence
        with st.expander(f"[{ev.evidence_id}] {ev.doc_title}"):
            st.write(ev.text)
            st.caption(f"Score: {ev.retrieval_score:.4f} | Rank: {ev.retrieval_rank} | Source: {ev.source_type}")

    # 4. Explanation & Attribution (Functional Component 4)
    st.header("Explanation & Attribution")
    st.write("Displays the contribution of each evidence piece.")

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown("**Evidence ID**")
    col2.markdown("**Attribution (SHAP / LIME)**")
    col3.markdown("**Intervention Influence (ΔP)**")
    col4.markdown("**Role**")

    for ev_res in res.evidence_results:
        c1, c2, c3, c4 = st.columns(4)
        c1.write(ev_res.evidence.evidence_id)
        c2.write(f"S: {ev_res.shap_target:.4f} | L: {ev_res.lime_target:.4f}")
        c3.write(f"{ev_res.delta_p:.4f} {'(Flipped)' if ev_res.verdict_changed else ''}")

        # Color code role
        role_color = "green" if ev_res.role == "SUPPORTING" else "red" if ev_res.role == "REFUTING" else "gray"
        c4.markdown(f"<span style='color:{role_color}'>{ev_res.role}</span>", unsafe_allow_html=True)

    # 5. Global Explanation Robustness (Locked Module 1 - IN DEVELOPMENT)
    render_in_development_card(
        heading="GLOBAL EXPLANATION ROBUSTNESS",
        description="Conflict detection and explanation stability analysis are planned for integration into the presentation dashboard. This module will provide insights into conflicting evidence signals and the robustness of generated explanations.",
        status="Under Development"
    )

    # 6. Evidence / Explanation Reliability Metrics (Fuzzy) (Locked Module 2 - IN DEVELOPMENT)
    render_in_development_card(
        heading="EVIDENCE / EXPLANATION RELIABILITY METRICS (FUZZY)",
        description="Fuzzy-based evidence reliability assessment and explanation confidence metrics are being integrated into the dashboard. This module will provide a structured presentation of fuzzy inference outputs and evidence reliability assessments.",
        status="Under Development"
    )

    # 7. Final Natural-Language Explanation (Locked Module 3 - IN DEVELOPMENT)
    render_in_development_card(
        heading="FINAL NATURAL-LANGUAGE EXPLANATION",
        description="Natural-language synthesis of the verifier verdict, retrieved evidence, attribution signals, and analytical outputs is planned for a future development iteration. This module is intended to combine the pipeline's outputs into a concise, human-readable fact-checking explanation.",
        status="Under Development"
    )

# Render mandatory project status notice near the bottom of the dashboard
render_project_status_notice()
