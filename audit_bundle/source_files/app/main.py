"""
Streamlit UI for End-to-End Fact-Checking Application.
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
st.set_page_config(page_title="Fuzzy-Causal Evidence Attribution", layout="wide")

@st.cache_resource
def load_pipeline():
    """Load the full pipeline. Cached so models only load once."""
    cfg = Config(top_k=3)
    return FactCheckingPipeline(config=cfg)

st.title("Fuzzy-Causal Fact-Checking & Explanation Pipeline")

# Initialize pipeline
with st.spinner("Loading models and pipeline (this may take a minute on first run)..."):
    pipeline = load_pipeline()

# 1. Claim Input
claim_input = st.text_input("Enter a claim to fact-check:", "Lincoln Motor Company was named by Henry M. Leland.")
run_button = st.button("Run Analysis")

if run_button and claim_input.strip():
    with st.spinner("Running full pipeline (Retrieval -> Verification -> Attribution -> Intervention -> Conflict -> Stability -> Fuzzy)..."):
        try:
            res = pipeline.process(claim_input)
        except Exception as e:
            st.error(f"Pipeline failed: {str(e)}")
            st.stop()
            
    # 2. Verdict Summary
    st.header("FACT-CHECKING VERDICT")
    st.info(f"**Verdict:** {res.verification.verdict}")
    st.write(f"**Probabilities:** SUPPORTS: {res.verification.probabilities.get('SUPPORTS',0):.4f} | REFUTES: {res.verification.probabilities.get('REFUTES',0):.4f} | NEI: {res.verification.probabilities.get('NOT_ENOUGH_INFO',0):.4f}")

    # 3. Retrieved Evidence
    st.header("Retrieved Evidence")
    for ev_res in res.evidence_results:
        ev = ev_res.evidence
        with st.expander(f"[{ev.evidence_id}] {ev.doc_title}"):
            st.write(ev.text)
            st.caption(f"Score: {ev.retrieval_score:.4f} | Rank: {ev.retrieval_rank} | Source: {ev.source_type}")

    # 4 & 5 & 8. Attribution, Intervention, and Role Analysis
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

    # 6. Conflict / Stability Analysis
    st.header("Global Explanation Robustness")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Conflict Detection")
        st.write(f"**Detected:** {'YES' if res.conflict.conflict_detected else 'NO'}")
        st.write(f"**Score:** {res.conflict.conflict_score:.4f}")
    with c2:
        st.subheader("Explanation Stability")
        st.write(f"**Score:** {res.stability.stability_score:.4f}")
        st.write(f"**Correlation (rho):** {res.stability.spearman_correlation:.4f}")

    # 7. Fuzzy Reliability
    st.header("Evidence / Explanation Reliability Metrics (Fuzzy)")
    for ev_res in res.evidence_results:
        with st.expander(f"Fuzzy Metrics for: {ev_res.evidence.evidence_id}"):
            fuz = ev_res.fuzzy_output
            st.write(f"- **Evidence Reliability:** {fuz.evidence_reliability:.4f}")
            st.write(f"- **Evidence Relevance:** {fuz.evidence_relevance:.4f}")
            st.write(f"- **Support Degree:** {fuz.support_degree:.4f}")
            st.write(f"- **Contradiction Degree:** {fuz.contradiction_degree:.4f}")
            st.write(f"- **Explanation Confidence:** {fuz.explanation_confidence:.4f}")
            
    # 8. Final Natural-Language Summary
    st.header("Final Natural-Language Explanation")
    st.success(res.final_explanation)
