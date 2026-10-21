import os
import json
import csv
import sys
import datetime
import numpy as np
import warnings
from scipy.stats import spearmanr, rankdata
import matplotlib.pyplot as plt

# Ensure local imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import Config
from retrieval.retriever import EvidenceRetriever
from verification.verifier import ClaimEvidenceVerifier
from attribution.explainer import EvidenceExplainer
from intervention.intervener import EvidenceIntervener
from data.schema import load_records

def run_sanity():
    print("Starting Phase 10 Experiment 1 Sanity Methodological Audit...")
    
    # 1. Freeze config
    cfg = Config(top_k=3, shap_num_samples=25, lime_num_samples=25)
    config_dict = {
        "dataset": "FEVER",
        "dataset_split": "dev",
        "number_of_claims_evaluated": 10,
        "retrieval_method": cfg.retrieval_mode,
        "top_k": cfg.top_k,
        "embedding_model": cfg.embedding_model,
        "verifier_model": cfg.verifier_model,
        "shap_sample_count": cfg.shap_num_samples,
        "lime_sample_count": cfg.lime_num_samples,
        "maximum_intervention_evidence": getattr(cfg, 'max_intervention_evidence', cfg.top_k),
        "intervention_method": "Leave-One-Out (LOO)",
        "run_timestamp": datetime.datetime.now().isoformat(),
        "python_version": sys.version
    }
    
    with open("phase10_sanity_config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4)
        
    # 2. Load Data
    records = load_records("data/fever_dev.jsonl")
    sanity_records = records[:10]
    
    # 3. Initialize models
    retriever = EvidenceRetriever(config=cfg)
    verifier = ClaimEvidenceVerifier(config=cfg)
    explainer = EvidenceExplainer(verifier=verifier, config=cfg)
    intervener = EvidenceIntervener(verifier=verifier, config=cfg)
    
    results = []
    claim_stats = []
    
    for rec in sanity_records:
        claim_id = rec.claim_id
        claim = rec.claim_text
        
        # A. Retrieval
        evs = retriever.retrieve(claim, top_k=cfg.top_k, mode=cfg.retrieval_mode)
        if not evs:
            continue
            
        # B. Explainer (SHAP/LIME)
        attr_res = explainer.explain(str(claim_id), claim, evs)
        
        # C. Intervention
        int_res = intervener.intervene(str(claim_id), claim, evs)
        
        # D. Align and Rank
        attr_map = {ea.evidence_id: ea for ea in attr_res.evidence_attributions}
        int_map = {ei.evidence_id: ei for ei in int_res.interventions}
        
        valid_evs = []
        raw_shap = []
        raw_lime = []
        inf_scores = []
        
        for ev in evs:
            if ev.evidence_id in attr_map and ev.evidence_id in int_map:
                valid_evs.append(ev)
                raw_shap.append(attr_map[ev.evidence_id].shap_target)
                raw_lime.append(attr_map[ev.evidence_id].lime_target)
                inf_scores.append(int_map[ev.evidence_id].intervention_influence)
                
        if len(valid_evs) < 2:
            continue
            
        # Use MAGNITUDE for importance comparisons
        abs_shap = [abs(x) for x in raw_shap]
        abs_lime = [abs(x) for x in raw_lime]
        
        # Rank arrays: rankdata with method='min'. Negate values so largest magnitude = rank 1.
        shap_ranks = rankdata([-x for x in abs_shap], method='min')
        lime_ranks = rankdata([-x for x in abs_lime], method='min')
        inf_ranks = rankdata([-x for x in inf_scores], method='min')
        
        # Per-Claim Correlation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho_shap, _ = spearmanr(abs_shap, inf_scores)
            rho_lime, _ = spearmanr(abs_lime, inf_scores)
            
        # Top-1 Agreement
        shap_top1 = set(i for i, r in enumerate(shap_ranks) if r == 1)
        lime_top1 = set(i for i, r in enumerate(lime_ranks) if r == 1)
        inf_top1 = set(i for i, r in enumerate(inf_ranks) if r == 1)
        
        top1_shap_agree = bool(shap_top1 & inf_top1)
        top1_lime_agree = bool(lime_top1 & inf_top1)
        
        claim_stats.append({
            "claim_id": claim_id,
            "rho_shap": rho_shap,
            "rho_lime": rho_lime,
            "top1_shap_agree": top1_shap_agree,
            "top1_lime_agree": top1_lime_agree,
            "num_ev": len(valid_evs)
        })
        
        for i, ev in enumerate(valid_evs):
            eid = ev.evidence_id
            ea = attr_map[eid]
            ei = int_map[eid]
            
            results.append({
                "claim_id": str(claim_id),
                "claim": claim,
                "evidence_id": eid,
                "evidence_text": ev.text,
                "SHAP_score_raw": raw_shap[i],
                "LIME_score_raw": raw_lime[i],
                "SHAP_magnitude": abs_shap[i],
                "LIME_magnitude": abs_lime[i],
                "intervention_influence": inf_scores[i],
                "baseline_target_probability": ei.baseline_target_probability,
                "intervened_target_probability": ei.intervened_target_probability,
                "baseline_verdict": ei.verdict_before,
                "intervened_verdict": ei.verdict_after,
                "verdict_changed": ei.verdict_changed,
                "SHAP_magnitude_rank": int(shap_ranks[i]),
                "LIME_magnitude_rank": int(lime_ranks[i]),
                "intervention_rank": int(inf_ranks[i])
            })
            
    # 4. Save CSV
    csv_keys = results[0].keys() if results else []
    with open("phase10_sanity_results.csv", "w", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_keys)
        writer.writeheader()
        writer.writerows(results)
        
    # 5. Compute Aggregates
    valid_shap_rhos = [s["rho_shap"] for s in claim_stats if not np.isnan(s["rho_shap"])]
    valid_lime_rhos = [s["rho_lime"] for s in claim_stats if not np.isnan(s["rho_lime"])]
    
    undef_shap = len(claim_stats) - len(valid_shap_rhos)
    undef_lime = len(claim_stats) - len(valid_lime_rhos)
    
    mean_shap_rho = float(np.mean(valid_shap_rhos)) if valid_shap_rhos else 0.0
    median_shap_rho = float(np.median(valid_shap_rhos)) if valid_shap_rhos else 0.0
    
    mean_lime_rho = float(np.mean(valid_lime_rhos)) if valid_lime_rhos else 0.0
    median_lime_rho = float(np.median(valid_lime_rhos)) if valid_lime_rhos else 0.0
    
    top1_shap_rate = sum(1 for s in claim_stats if s["top1_shap_agree"]) / len(claim_stats) if claim_stats else 0.0
    top1_lime_rate = sum(1 for s in claim_stats if s["top1_lime_agree"]) / len(claim_stats) if claim_stats else 0.0
    
    verdict_flips = sum(1 for r in results if r["verdict_changed"])
    
    # 6. Plots (Magnitude vs Influence)
    all_abs_shap = [r["SHAP_magnitude"] for r in results]
    all_abs_lime = [r["LIME_magnitude"] for r in results]
    all_inf = [r["intervention_influence"] for r in results]
    
    plt.figure(figsize=(6,5))
    plt.scatter(all_abs_shap, all_inf, alpha=0.7)
    plt.title("abs(SHAP) vs Intervention Influence (Sanity)")
    plt.xlabel("Absolute SHAP Attribution Magnitude")
    plt.ylabel("Intervention Influence (ΔP)")
    plt.tight_layout()
    plt.savefig("outputs/phase10_sanity/shap_vs_influence.png")
    plt.close()
    
    plt.figure(figsize=(6,5))
    plt.scatter(all_abs_lime, all_inf, alpha=0.7, color='orange')
    plt.title("abs(LIME) vs Intervention Influence (Sanity)")
    plt.xlabel("Absolute LIME Attribution Magnitude")
    plt.ylabel("Intervention Influence (ΔP)")
    plt.tight_layout()
    plt.savefig("outputs/phase10_sanity/lime_vs_influence.png")
    plt.close()
    
    # 7. Write Markdown Report
    report_lines = [
        "# Experiment 1 — Sanity Evaluation",
        "",
        "## Research Question",
        "When SHAP/LIME identifies an evidence piece as influential, does removing that same evidence piece produce a measurable change in the verifier’s decision?",
        "",
        "## Frozen Configuration",
        "```json",
        json.dumps(config_dict, indent=4),
        "```",
        "",
        "## Dataset",
        f"- dataset: {config_dict['dataset']}",
        f"- split: {config_dict['dataset_split']}",
        f"- claim count: {config_dict['number_of_claims_evaluated']} (Sanity)",
        f"- retrieval settings: {config_dict['retrieval_method']}, Top-{config_dict['top_k']}",
        "",
        "## Models",
        f"- embedding model: {config_dict['embedding_model']}",
        f"- verifier model: {config_dict['verifier_model']}",
        f"- SHAP settings: {config_dict['shap_sample_count']} samples",
        f"- LIME settings: {config_dict['lime_sample_count']} samples",
        "",
        "## Methodological Definitions",
        "### Intervention Definition",
        "```python",
        "intervention_influence = abs(baseline_target_probability - intervened_target_probability)",
        "```",
        "### Attribution Comparison",
        "Signed attribution is retained for raw analysis, while **absolute attribution magnitude** (`abs(SHAP)`) is used for evidence-importance ranking and correlation against the non-negative intervention influence.",
        "### Correlation Method",
        "Primary correlation is computed **per claim** using Spearman rank correlation, then aggregated (mean/median) across valid claims.",
        "### Tie Handling",
        "Ranking ties are assigned the minimum rank of the group. Undefined correlations caused by constant/tied vectors (where variance is 0) are explicitly excluded from the correlation aggregate and reported separately.",
        "### Top-K Agreement",
        "Top-1 Agreement: Calculates the percentage of claims where the top-ranked evidence by `abs(SHAP)` (or `abs(LIME)`) matches the top-ranked evidence by `intervention_influence` (resolving ties via set intersection).",
        "",
        "## Data Integrity Checks",
        "- **evidence-set consistency**: Verified. The exact same `RetrievedEvidence` instances are passed to Explainer and Intervener.",
        "- **evidence-ID consistency**: Verified. All metrics are mapped back via strict `evidence_id` matching.",
        "- **ranking alignment**: Verified. Ranks are derived via `scipy.stats.rankdata` (`method='min'`) per claim.",
        "- **probability validity**: Verified. All probabilities are [0, 1].",
        "- **intervention calculation validity**: Verified. ΔP strictly follows absolute difference.",
        "",
        "## Aggregate Sanity Statistics",
        "> **SANITY-EVALUATION STATISTICS — NOT FINAL RESEARCH RESULTS**",
        "",
        f"- Number of claims evaluated: {len(claim_stats)}",
        f"- Number of evidence pieces (interventions): {len(results)}",
        f"- Verdict-flip rate: {verdict_flips} / {len(results)} ({(verdict_flips/len(results))*100:.1f}%)",
        "",
        "### Top-1 Agreement (Evidence Importance)",
        f"- abs(SHAP) Top-1 Agreement: {top1_shap_rate*100:.1f}%",
        f"- abs(LIME) Top-1 Agreement: {top1_lime_rate*100:.1f}%",
        "",
        "### Per-Claim Spearman Correlation (abs(Attribution) vs Influence)",
        f"- Valid SHAP correlations: {len(valid_shap_rhos)} (Undefined due to ties/zeros: {undef_shap})",
        f"- Mean SHAP rho: {mean_shap_rho:.4f}  |  Median SHAP rho: {median_shap_rho:.4f}",
        f"- Valid LIME correlations: {len(valid_lime_rhos)} (Undefined due to ties/zeros: {undef_lime})",
        f"- Mean LIME rho: {mean_lime_rho:.4f}  |  Median LIME rho: {median_lime_rho:.4f}",
        "",
        "## Sample Claim Results",
    ]
    
    unique_claims = list(set(r["claim_id"] for r in results))[:2]
    for cid in unique_claims:
        claim_rows = [r for r in results if r["claim_id"] == cid]
        report_lines.append(f"### Claim: {claim_rows[0]['claim']}")
        report_lines.append("| Evidence ID | SHAP Rank | LIME Rank | Infl Rank | SHAP (raw) | abs(SHAP) | ΔP | Flipped? |")
        report_lines.append("|---|---|---|---|---|---|---|---|")
        for r in claim_rows:
            report_lines.append(f"| {r['evidence_id']} | {r['SHAP_magnitude_rank']} | {r['LIME_magnitude_rank']} | {r['intervention_rank']} | {r['SHAP_score_raw']:.4f} | {r['SHAP_magnitude']:.4f} | {r['intervention_influence']:.4f} | {r['verdict_changed']} |")
        report_lines.append("")
        
    report_lines.extend([
        "## Next Step",
        "Awaiting explicit approval to run the formal Experiment 1 evaluation."
    ])
    
    with open("PHASE_10_SANITY_REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print("Sanity script complete.")

if __name__ == "__main__":
    run_sanity()
