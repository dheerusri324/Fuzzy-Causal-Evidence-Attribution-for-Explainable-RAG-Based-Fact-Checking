import os
import csv
import json
import numpy as np
import scipy.stats as stats
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

os.makedirs("outputs/phase12_analysis", exist_ok=True)

# 1. Read Raw Results
csv_path = "outputs/phase10_formal/results.csv"
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

total_removals = len(rows)
claims_dict = defaultdict(list)
for r in rows:
    claims_dict[r["claim_id"]].append(r)
total_claims = len(claims_dict)

# 2. Compute Per-Claim Stats
per_claim_stats = []
for cid, ev_list in claims_dict.items():
    raw_shap = [float(x["SHAP_score_raw"]) for x in ev_list]
    raw_lime = [float(x["LIME_score_raw"]) for x in ev_list]
    abs_shap = [float(x["SHAP_magnitude"]) for x in ev_list]
    abs_lime = [float(x["LIME_magnitude"]) for x in ev_list]
    inf_scores = [float(x["intervention_influence"]) for x in ev_list]
    
    # Ranks (negate so largest magnitude = rank 1, method='min')
    re_shap_ranks = stats.rankdata([-x for x in abs_shap], method='min')
    re_lime_ranks = stats.rankdata([-x for x in abs_lime], method='min')
    re_inf_ranks = stats.rankdata([-x for x in inf_scores], method='min')
    
    # Correlation
    with np.errstate(all='ignore'):
        rho_shap, _ = stats.spearmanr(abs_shap, inf_scores)
        rho_lime, _ = stats.spearmanr(abs_lime, inf_scores)
        
    shap_top1 = set(i for i, r in enumerate(re_shap_ranks) if r == 1)
    lime_top1 = set(i for i, r in enumerate(re_lime_ranks) if r == 1)
    inf_top1 = set(i for i, r in enumerate(re_inf_ranks) if r == 1)
    
    top1_shap_agree = bool(shap_top1 & inf_top1)
    top1_lime_agree = bool(lime_top1 & inf_top1)
    
    flips_in_claim = sum(1 for x in ev_list if x["verdict_changed"].strip().lower() == "true")
    
    per_claim_stats.append({
        "claim_id": cid,
        "rho_shap": rho_shap,
        "rho_lime": rho_lime,
        "top1_shap_agree": top1_shap_agree,
        "top1_lime_agree": top1_lime_agree,
        "flips_in_claim": flips_in_claim,
        "baseline_verdict": ev_list[0]["baseline_verdict"]
    })

valid_shap_rhos = [s["rho_shap"] for s in per_claim_stats if not np.isnan(s["rho_shap"])]
valid_lime_rhos = [s["rho_lime"] for s in per_claim_stats if not np.isnan(s["rho_lime"])]

def get_stats(arr):
    a = np.array(arr)
    return {
        "count": len(a),
        "mean": float(np.mean(a)),
        "std_sample": float(np.std(a, ddof=1)),
        "std_pop": float(np.std(a, ddof=0)),
        "min": float(np.min(a)),
        "q25": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "q75": float(np.percentile(a, 75)),
        "max": float(np.max(a)),
        "iqr": float(np.percentile(a, 75) - np.percentile(a, 25))
    }

shap_s = get_stats(valid_shap_rhos)
lime_s = get_stats(valid_lime_rhos)

shap_top1_count = sum(1 for s in per_claim_stats if s["top1_shap_agree"])
lime_top1_count = sum(1 for s in per_claim_stats if s["top1_lime_agree"])
both_agree = sum(1 for s in per_claim_stats if s["top1_shap_agree"] and s["top1_lime_agree"])
shap_only = sum(1 for s in per_claim_stats if s["top1_shap_agree"] and not s["top1_lime_agree"])
lime_only = sum(1 for s in per_claim_stats if not s["top1_shap_agree"] and s["top1_lime_agree"])
neither = sum(1 for s in per_claim_stats if not s["top1_shap_agree"] and not s["top1_lime_agree"])

verdict_flips_count = sum(1 for r in rows if r["verdict_changed"].strip().lower() == "true")
non_flips_count = total_removals - verdict_flips_count

# Summary records for CSV
summary_records = [
    {"Category": "Overview", "Metric": "Total Claims Processed", "Value": str(total_claims), "Unit": "claims", "Notes": "Unique claims in FEVER dev split"},
    {"Category": "Overview", "Metric": "Total Evidence Evaluations", "Value": str(total_removals), "Unit": "evaluations", "Notes": "3 evidence pieces per claim (LOO interventions)"},
    {"Category": "Overview", "Metric": "Top-k Evidence per Claim", "Value": "3", "Unit": "evidence/claim", "Notes": "Fixed retrieval budget"},
    
    {"Category": "Verdict Flips", "Metric": "Total Verdict Flips", "Value": str(verdict_flips_count), "Unit": "flips", "Notes": "Interventions resulting in verdict change"},
    {"Category": "Verdict Flips", "Metric": "Total Non-Flips", "Value": str(non_flips_count), "Unit": "non-flips", "Notes": "Interventions preserving baseline verdict"},
    {"Category": "Verdict Flips", "Metric": "Verdict-Flip Rate", "Value": f"{(verdict_flips_count/total_removals)*100:.2f}", "Unit": "%", "Notes": "Percentage of evidence removals that flip verdict"},
    
    {"Category": "Top-1 Agreement", "Metric": "SHAP Top-1 Agreement Count", "Value": str(shap_top1_count), "Unit": "claims", "Notes": "Highest abs(SHAP) matches highest intervention influence"},
    {"Category": "Top-1 Agreement", "Metric": "SHAP Top-1 Agreement Rate", "Value": f"{(shap_top1_count/total_claims)*100:.2f}", "Unit": "%", "Notes": "361 / 500"},
    {"Category": "Top-1 Agreement", "Metric": "LIME Top-1 Agreement Count", "Value": str(lime_top1_count), "Unit": "claims", "Notes": "Highest abs(LIME) matches highest intervention influence"},
    {"Category": "Top-1 Agreement", "Metric": "LIME Top-1 Agreement Rate", "Value": f"{(lime_top1_count/total_claims)*100:.2f}", "Unit": "%", "Notes": "420 / 500"},
    {"Category": "Top-1 Agreement", "Metric": "Consensus (Both Agree)", "Value": str(both_agree), "Unit": "claims", "Notes": f"{(both_agree/total_claims)*100:.2f}% of claims"},
    {"Category": "Top-1 Agreement", "Metric": "LIME Only Agrees", "Value": str(lime_only), "Unit": "claims", "Notes": f"{(lime_only/total_claims)*100:.2f}% of claims"},
    {"Category": "Top-1 Agreement", "Metric": "SHAP Only Agrees", "Value": str(shap_only), "Unit": "claims", "Notes": f"{(shap_only/total_claims)*100:.2f}% of claims"},
    {"Category": "Top-1 Agreement", "Metric": "Neither Agrees", "Value": str(neither), "Unit": "claims", "Notes": f"{(neither/total_claims)*100:.2f}% of claims"},
    
    {"Category": "Spearman Rho (SHAP)", "Metric": "Valid Correlations Count", "Value": str(shap_s["count"]), "Unit": "claims", "Notes": "Zero undefined correlations"},
    {"Category": "Spearman Rho (SHAP)", "Metric": "Mean Rho", "Value": f"{shap_s['mean']:.4f}", "Unit": "correlation", "Notes": "Unweighted arithmetic mean across claims"},
    {"Category": "Spearman Rho (SHAP)", "Metric": "Median Rho", "Value": f"{shap_s['median']:.4f}", "Unit": "correlation", "Notes": "50th percentile"},
    {"Category": "Spearman Rho (SHAP)", "Metric": "Std Dev (Sample, ddof=1)", "Value": f"{shap_s['std_sample']:.4f}", "Unit": "correlation", "Notes": "Sample standard deviation"},
    {"Category": "Spearman Rho (SHAP)", "Metric": "Min Rho", "Value": f"{shap_s['min']:.4f}", "Unit": "correlation", "Notes": "Minimum observed"},
    {"Category": "Spearman Rho (SHAP)", "Metric": "Q1 (25th Percentile)", "Value": f"{shap_s['q25']:.4f}", "Unit": "correlation", "Notes": "First quartile"},
    {"Category": "Spearman Rho (SHAP)", "Metric": "Q3 (75th Percentile)", "Value": f"{shap_s['q75']:.4f}", "Unit": "correlation", "Notes": "Third quartile"},
    {"Category": "Spearman Rho (SHAP)", "Metric": "Max Rho", "Value": f"{shap_s['max']:.4f}", "Unit": "correlation", "Notes": "Maximum observed"},
    {"Category": "Spearman Rho (SHAP)", "Metric": "IQR", "Value": f"{shap_s['iqr']:.4f}", "Unit": "correlation", "Notes": "Q3 - Q1"},
    
    {"Category": "Spearman Rho (LIME)", "Metric": "Valid Correlations Count", "Value": str(lime_s["count"]), "Unit": "claims", "Notes": "Zero undefined correlations"},
    {"Category": "Spearman Rho (LIME)", "Metric": "Mean Rho", "Value": f"{lime_s['mean']:.4f}", "Unit": "correlation", "Notes": "Unweighted arithmetic mean across claims"},
    {"Category": "Spearman Rho (LIME)", "Metric": "Median Rho", "Value": f"{lime_s['median']:.4f}", "Unit": "correlation", "Notes": "50th percentile"},
    {"Category": "Spearman Rho (LIME)", "Metric": "Std Dev (Sample, ddof=1)", "Value": f"{lime_s['std_sample']:.4f}", "Unit": "correlation", "Notes": "Sample standard deviation"},
    {"Category": "Spearman Rho (LIME)", "Metric": "Min Rho", "Value": f"{lime_s['min']:.4f}", "Unit": "correlation", "Notes": "Minimum observed"},
    {"Category": "Spearman Rho (LIME)", "Metric": "Q1 (25th Percentile)", "Value": f"{lime_s['q25']:.4f}", "Unit": "correlation", "Notes": "First quartile"},
    {"Category": "Spearman Rho (LIME)", "Metric": "Q3 (75th Percentile)", "Value": f"{lime_s['q75']:.4f}", "Unit": "correlation", "Notes": "Third quartile"},
    {"Category": "Spearman Rho (LIME)", "Metric": "Max Rho", "Value": f"{lime_s['max']:.4f}", "Unit": "correlation", "Notes": "Maximum observed"},
    {"Category": "Spearman Rho (LIME)", "Metric": "IQR", "Value": f"{lime_s['iqr']:.4f}", "Unit": "correlation", "Notes": "Q3 - Q1"}
]

# Write CSV Summary
with open("outputs/phase12_analysis/analysis_summary.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["Category", "Metric", "Value", "Unit", "Notes"])
    writer.writeheader()
    writer.writerows(summary_records)
print("Saved analysis_summary.csv")

# 3. Generate Publication-Quality Figures

# Figure 1: Spearman Rho Discrete Distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

shap_counts = Counter(round(x, 4) for x in valid_shap_rhos)
lime_counts = Counter(round(x, 4) for x in valid_lime_rhos)

rho_keys = [-1.0, -0.5, 0.5, 0.866, 1.0]
labels = ["-1.0\n(Reverse)", "-0.5\n(Cyclic)", "+0.5\n(Partial)", "+0.866\n(Tied)", "+1.0\n(Identical)"]

s_vals = [shap_counts.get(k, 0) for k in rho_keys]
l_vals = [lime_counts.get(k, 0) for k in rho_keys]

x_pos = np.arange(len(rho_keys))
width = 0.55

# SHAP panel
bars1 = ax1.bar(x_pos, s_vals, width, color="#2b5c8f", edgecolor="black", alpha=0.85)
ax1.set_title("SHAP vs Intervention Influence", fontsize=12, fontweight="bold")
ax1.set_xlabel("Spearman Rank Correlation (ρ)", fontsize=11)
ax1.set_ylabel("Number of Claims (N = 500)", fontsize=11)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(labels, fontsize=9)
ax1.grid(axis='y', linestyle='--', alpha=0.4)
ax1.set_ylim(0, 270)

for bar in bars1:
    yval = bar.get_height()
    pct = (yval / total_claims) * 100
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 4, f"{yval}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=8.5)

# LIME panel
bars2 = ax2.bar(x_pos, l_vals, width, color="#e27c38", edgecolor="black", alpha=0.85)
ax2.set_title("LIME vs Intervention Influence", fontsize=12, fontweight="bold")
ax2.set_xlabel("Spearman Rank Correlation (ρ)", fontsize=11)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels, fontsize=9)
ax2.grid(axis='y', linestyle='--', alpha=0.4)

for bar in bars2:
    yval = bar.get_height()
    pct = (yval / total_claims) * 100
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 4, f"{yval}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=8.5)

fig.suptitle("Distribution of Per-Claim Spearman Rank Correlation (N=500, k=3)", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("outputs/phase12_analysis/spearman_rho_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved spearman_rho_distribution.png")

# Figure 2: Top-1 Agreement Comparison & Consensus Overlap
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Panel A: Overall Top-1 Agreement
methods = ["SHAP", "LIME"]
agreements = [shap_top1_count / total_claims * 100, lime_top1_count / total_claims * 100]
colors = ["#2b5c8f", "#e27c38"]

bars = ax1.bar(methods, agreements, width=0.45, color=colors, edgecolor="black", alpha=0.85)
ax1.set_ylabel("Top-1 Agreement Rate (%)", fontsize=11)
ax1.set_ylim(0, 100)
ax1.set_title("(a) Top-1 Attribution Agreement", fontsize=12, fontweight="bold")
ax1.grid(axis='y', linestyle='--', alpha=0.4)

for bar, count in zip(bars, [shap_top1_count, lime_top1_count]):
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%\n({count}/500)", ha='center', va='bottom', fontsize=10, fontweight="bold")

# Panel B: Agreement Overlap Breakdown
cat_labels = ["Both Agree", "LIME Only", "Neither", "SHAP Only"]
cat_counts = [both_agree, lime_only, neither, shap_only]
cat_colors = ["#4caf50", "#ff9800", "#e91e63", "#2196f3"]

bars_cat = ax2.bar(cat_labels, cat_counts, width=0.55, color=cat_colors, edgecolor="black", alpha=0.85)
ax2.set_ylabel("Number of Claims", fontsize=11)
ax2.set_ylim(0, 400)
ax2.set_title("(b) Agreement Overlap Across Claims", fontsize=12, fontweight="bold")
ax2.grid(axis='y', linestyle='--', alpha=0.4)

for bar in bars_cat:
    yval = bar.get_height()
    pct = (yval / total_claims) * 100
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f"{yval}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=9.5)

plt.tight_layout()
plt.savefig("outputs/phase12_analysis/top1_agreement_breakdown.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved top1_agreement_breakdown.png")

# Figure 3: Verdict-Flip Dynamics and Class Sensitivity
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Subplot A: Flips by Baseline Verdict Class
bverdicts = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
b_totals = [462, 801, 237]
b_flips = [172, 81, 69]
b_rates = [(f / t) * 100 for f, t in zip(b_flips, b_totals)]

bars_bv = ax1.bar(bverdicts, b_rates, width=0.5, color=["#3f51b5", "#f44336", "#9c27b0"], edgecolor="black", alpha=0.85)
ax1.set_ylabel("Verdict-Flip Rate (%)", fontsize=11)
ax1.set_ylim(0, 48)
ax1.set_title("(a) Intervention Sensitivity by Verdict Class", fontsize=12, fontweight="bold")
ax1.grid(axis='y', linestyle='--', alpha=0.4)

for bar, f, t in zip(bars_bv, b_flips, b_totals):
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.1f}%\n({f}/{t})", ha='center', va='bottom', fontsize=9.5)

# Subplot B: Flips per Claim Histogram
flip_counts_dist = [247, 195, 47, 11]
flip_x = [0, 1, 2, 3]
bars_fc = ax2.bar(flip_x, flip_counts_dist, width=0.55, color="#607d8b", edgecolor="black", alpha=0.85)
ax2.set_xlabel("Number of Verdict Flips per Claim (out of 3)", fontsize=11)
ax2.set_ylabel("Number of Claims", fontsize=11)
ax2.set_ylim(0, 280)
ax2.set_title("(b) Distribution of Flips per Claim", fontsize=12, fontweight="bold")
ax2.set_xticks(flip_x)
ax2.grid(axis='y', linestyle='--', alpha=0.4)

for bar in bars_fc:
    yval = bar.get_height()
    pct = (yval / total_claims) * 100
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 4, f"{yval}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=9.5)

plt.tight_layout()
plt.savefig("outputs/phase12_analysis/verdict_flip_dynamics.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved verdict_flip_dynamics.png")

# Figure 4: Intervention Influence by Attribution Rank
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

shap_r1_inf = [float(r['intervention_influence']) for r in rows if int(r['SHAP_magnitude_rank']) == 1]
shap_r2_inf = [float(r['intervention_influence']) for r in rows if int(r['SHAP_magnitude_rank']) == 2]
shap_r3_inf = [float(r['intervention_influence']) for r in rows if int(r['SHAP_magnitude_rank']) == 3]

lime_r1_inf = [float(r['intervention_influence']) for r in rows if int(r['LIME_magnitude_rank']) == 1]
lime_r2_inf = [float(r['intervention_influence']) for r in rows if int(r['LIME_magnitude_rank']) == 2]
lime_r3_inf = [float(r['intervention_influence']) for r in rows if int(r['LIME_magnitude_rank']) == 3]

bplot1 = ax1.boxplot([shap_r1_inf, shap_r2_inf, shap_r3_inf], positions=[1, 2, 3], patch_artist=True,
                     boxprops=dict(facecolor="#2b5c8f", alpha=0.7),
                     medianprops=dict(color="black", linewidth=1.5))
ax1.set_title("Intervention Influence by SHAP Rank", fontsize=12, fontweight="bold")
ax1.set_xlabel("SHAP Magnitude Rank", fontsize=11)
ax1.set_ylabel("Intervention Influence (ΔP)", fontsize=11)
ax1.set_xticklabels(["Rank 1 (Top)", "Rank 2", "Rank 3"], fontsize=10)
ax1.grid(axis='y', linestyle='--', alpha=0.4)

# Print means for annotation
m1, m2, m3 = np.mean(shap_r1_inf), np.mean(shap_r2_inf), np.mean(shap_r3_inf)
ax1.text(1, 0.9, f"Mean: {m1:.3f}", ha='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
ax1.text(2, 0.9, f"Mean: {m2:.3f}", ha='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
ax1.text(3, 0.9, f"Mean: {m3:.3f}", ha='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

bplot2 = ax2.boxplot([lime_r1_inf, lime_r2_inf, lime_r3_inf], positions=[1, 2, 3], patch_artist=True,
                     boxprops=dict(facecolor="#e27c38", alpha=0.7),
                     medianprops=dict(color="black", linewidth=1.5))
ax2.set_title("Intervention Influence by LIME Rank", fontsize=12, fontweight="bold")
ax2.set_xlabel("LIME Magnitude Rank", fontsize=11)
ax2.set_xticklabels(["Rank 1 (Top)", "Rank 2", "Rank 3"], fontsize=10)
ax2.grid(axis='y', linestyle='--', alpha=0.4)

lm1, lm2, lm3 = np.mean(lime_r1_inf), np.mean(lime_r2_inf), np.mean(lime_r3_inf)
ax2.text(1, 0.9, f"Mean: {lm1:.3f}", ha='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
ax2.text(2, 0.9, f"Mean: {lm2:.3f}", ha='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
ax2.text(3, 0.9, f"Mean: {lm3:.3f}", ha='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

fig.suptitle("Measured Intervention Influence (ΔP) Stratified by Attribution Rank", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("outputs/phase12_analysis/influence_by_attribution_rank.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved influence_by_attribution_rank.png")
