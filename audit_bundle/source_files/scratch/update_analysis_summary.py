import csv
import shutil

summary_records = [
    # Overview & Corpus
    {"Category": "Overview", "Metric": "Evaluated Claims", "Value": "500", "Unit": "claims", "Notes": "Claims evaluated from FEVER dev split (first 500 of 750 records)"},
    {"Category": "Overview", "Metric": "Evidence Corpus Size", "Value": "808", "Unit": "passages", "Notes": "Closed candidate pool of unique evidence passages extracted from the 750 dev claims (copenlu/fever_gold_evidence)"},
    {"Category": "Overview", "Metric": "Corpus Construction Nature", "Value": "Gold-Conditioned Closed Pool", "Unit": "text", "Notes": "Retrieval searched 808-passage candidate pool (534 gold + 274 distractors), not full 5M Wikipedia dump"},
    {"Category": "Overview", "Metric": "Top-k Evidence per Claim", "Value": "3", "Unit": "evidence/claim", "Notes": "Fixed retrieval budget for claim-evidence verification"},
    {"Category": "Overview", "Metric": "Total Evidence Evaluations", "Value": "1500", "Unit": "removals", "Notes": "Total leave-one-out (LOO) evidence removals (500 claims x 3 evidence)"},

    # Gold Evidence Representation & Top-3 Retrieval Coverage
    {"Category": "Corpus Availability", "Metric": "Gold Passages in Corpus", "Value": "534", "Unit": "passages", "Notes": "534 / 534 annotated gold passages present in candidate corpus (100.0%)"},
    {"Category": "Retrieval Coverage", "Metric": "Claims with Non-Empty Gold Annotations", "Value": "500", "Unit": "claims", "Notes": "All 500 evaluated claims possess non-empty gold annotations"},
    {"Category": "Retrieval Coverage", "Metric": "All Annotated Gold IDs Retrieved in Top-3", "Value": "409", "Unit": "claims", "Notes": "81.80% of claims (409 / 500 with non-empty gold annotations)"},
    {"Category": "Retrieval Coverage", "Metric": "Partial Gold-Evidence Coverage in Top-3", "Value": "21", "Unit": "claims", "Notes": "4.20% of claims (at least 1, but not all, annotated gold IDs retrieved)"},
    {"Category": "Retrieval Coverage", "Metric": "Zero Gold-Evidence Coverage in Top-3", "Value": "70", "Unit": "claims", "Notes": "14.00% of claims (annotated gold exists, but zero matches in Top-3)"},
    {"Category": "Retrieval Coverage", "Metric": "At Least One Gold ID Retrieved in Top-3", "Value": "430", "Unit": "claims", "Notes": "86.00% of claims (409 full + 21 partial = 430 / 500)"},
    {"Category": "Retrieval Coverage", "Metric": "Retrieved Gold Evidence Instances", "Value": "443", "Unit": "instances", "Notes": "29.53% of all 1500 retrieved evidence pieces (443 / 1500)"},
    {"Category": "Retrieval Coverage", "Metric": "Retrieved Distractor Instances", "Value": "1057", "Unit": "instances", "Notes": "70.47% of all 1500 retrieved evidence pieces (1057 / 1500)"},

    # Baseline Verdict Distribution across Claims
    {"Category": "Baseline Distribution", "Metric": "SUPPORTS Claims", "Value": "154", "Unit": "claims", "Notes": "30.80% of evaluated claims (154 / 500)"},
    {"Category": "Baseline Distribution", "Metric": "REFUTES Claims", "Value": "267", "Unit": "claims", "Notes": "53.40% of evaluated claims (267 / 500)"},
    {"Category": "Baseline Distribution", "Metric": "NOT_ENOUGH_INFO Claims", "Value": "79", "Unit": "claims", "Notes": "15.80% of evaluated claims (79 / 500)"},

    # Verdict Flips (Evaluations & Denominators Clarified)
    {"Category": "Verdict Flips (Evaluations)", "Metric": "Total Verdict Flips", "Value": "322", "Unit": "flips", "Notes": "Total removals that changed baseline verdict"},
    {"Category": "Verdict Flips (Evaluations)", "Metric": "Total Non-Flips", "Value": "1178", "Unit": "non-flips", "Notes": "Total removals that preserved baseline verdict"},
    {"Category": "Verdict Flips (Evaluations)", "Metric": "Overall Verdict-Flip Rate", "Value": "21.47", "Unit": "%", "Notes": "322 flips / 1500 total removals"},
    {"Category": "Verdict Flips (Evaluations)", "Metric": "SUPPORTS Removal Flip Rate", "Value": "37.23", "Unit": "%", "Notes": "172 flips / 462 removal evaluations in SUPPORTS claims (denominator = removals)"},
    {"Category": "Verdict Flips (Evaluations)", "Metric": "REFUTES Removal Flip Rate", "Value": "10.11", "Unit": "%", "Notes": "81 flips / 801 removal evaluations in REFUTES claims (denominator = removals)"},
    {"Category": "Verdict Flips (Evaluations)", "Metric": "NOT_ENOUGH_INFO Removal Flip Rate", "Value": "29.11", "Unit": "%", "Notes": "69 flips / 237 removal evaluations in NEI claims (denominator = removals)"},

    # Verdict Flips (Claim-Level Sensitivity)
    {"Category": "Verdict Flips (Claim-Level)", "Metric": "Claims with >= 1 Flip", "Value": "253", "Unit": "claims", "Notes": "50.60% of all claims (253 / 500)"},
    {"Category": "Verdict Flips (Claim-Level)", "Metric": "Claims with 0 Flips", "Value": "247", "Unit": "claims", "Notes": "49.40% of all claims (247 / 500)"},
    {"Category": "Verdict Flips (Claim-Level)", "Metric": "SUPPORTS Claims with >= 1 Flip", "Value": "135", "Unit": "claims", "Notes": "87.66% of SUPPORTS claims (135 / 154)"},
    {"Category": "Verdict Flips (Claim-Level)", "Metric": "REFUTES Claims with >= 1 Flip", "Value": "70", "Unit": "claims", "Notes": "26.22% of REFUTES claims (70 / 267)"},
    {"Category": "Verdict Flips (Claim-Level)", "Metric": "NOT_ENOUGH_INFO Claims with >= 1 Flip", "Value": "48", "Unit": "claims", "Notes": "60.76% of NEI claims (48 / 79)"},

    # Direct SHAP-LIME Pairwise Comparison
    {"Category": "Attribution Agreement", "Metric": "Direct SHAP-LIME Top-1 Agreement", "Value": "412", "Unit": "claims", "Notes": "82.40% of claims (412 / 500); abs(SHAP) and abs(LIME) rank identical evidence piece #1"},
    {"Category": "Attribution Agreement", "Metric": "Direct SHAP-LIME Top-1 Disagreement", "Value": "88", "Unit": "claims", "Notes": "17.60% of claims (88 / 500); abs(SHAP) and abs(LIME) select different evidence pieces for rank #1"},

    # Top-1 Agreement with Intervention Influence
    {"Category": "Top-1 Agreement (Intervention)", "Metric": "SHAP Top-1 Agreement Count", "Value": "361", "Unit": "claims", "Notes": "Highest abs(SHAP) matches highest intervention influence (361 / 500)"},
    {"Category": "Top-1 Agreement (Intervention)", "Metric": "SHAP Top-1 Agreement Rate", "Value": "72.20", "Unit": "%", "Notes": "361 / 500 claims"},
    {"Category": "Top-1 Agreement (Intervention)", "Metric": "LIME Top-1 Agreement Count", "Value": "420", "Unit": "claims", "Notes": "Highest abs(LIME) matches highest intervention influence (420 / 500)"},
    {"Category": "Top-1 Agreement (Intervention)", "Metric": "LIME Top-1 Agreement Rate", "Value": "84.00", "Unit": "%", "Notes": "420 / 500 claims"},
    {"Category": "Top-1 Agreement (Intervention)", "Metric": "Consensus (Both Agree with Intervention)", "Value": "354", "Unit": "claims", "Notes": "70.80% of claims (354 / 500)"},
    {"Category": "Top-1 Agreement (Intervention)", "Metric": "LIME Only Agrees with Intervention", "Value": "66", "Unit": "claims", "Notes": "13.20% of claims (66 / 500)"},
    {"Category": "Top-1 Agreement (Intervention)", "Metric": "SHAP Only Agrees with Intervention", "Value": "7", "Unit": "claims", "Notes": "1.40% of claims (7 / 500)"},
    {"Category": "Top-1 Agreement (Intervention)", "Metric": "Neither Agrees with Intervention", "Value": "73", "Unit": "claims", "Notes": "14.60% of claims (73 / 500; 58 agree with each other, 15 disagree with each other)"},

    # Spearman Rho (SHAP)
    {"Category": "Spearman Rho (SHAP)", "Metric": "Valid Correlations Count", "Value": "500", "Unit": "claims", "Notes": "Zero undefined correlations across 500 claims"},
    {"Category": "Spearman Rho (SHAP)", "Metric": "Mean Rho", "Value": "0.4517", "Unit": "correlation", "Notes": "Coarse measure of within-claim rank agreement"},
    {"Category": "Spearman Rho (SHAP)", "Metric": "Median Rho", "Value": "0.5000", "Unit": "correlation", "Notes": "50th percentile across claims"},
    {"Category": "Spearman Rho (SHAP)", "Metric": "Std Dev (Sample, ddof=1)", "Value": "0.6397", "Unit": "correlation", "Notes": "Sample standard deviation"},
    {"Category": "Spearman Rho (SHAP)", "Metric": "Min Rho", "Value": "-1.0000", "Unit": "correlation", "Notes": "Minimum observed"},
    {"Category": "Spearman Rho (SHAP)", "Metric": "Q1 (25th Percentile)", "Value": "0.5000", "Unit": "correlation", "Notes": "First quartile"},
    {"Category": "Spearman Rho (SHAP)", "Metric": "Q3 (75th Percentile)", "Value": "1.0000", "Unit": "correlation", "Notes": "Third quartile"},
    {"Category": "Spearman Rho (SHAP)", "Metric": "Max Rho", "Value": "1.0000", "Unit": "correlation", "Notes": "Maximum observed"},
    {"Category": "Spearman Rho (SHAP)", "Metric": "IQR", "Value": "0.5000", "Unit": "correlation", "Notes": "Q3 - Q1"},
    {"Category": "Spearman Rho (SHAP)", "Metric": "Positive Correlations (rho > 0)", "Value": "78.80", "Unit": "%", "Notes": "394 / 500 claims"},

    # Spearman Rho (LIME)
    {"Category": "Spearman Rho (LIME)", "Metric": "Valid Correlations Count", "Value": "500", "Unit": "claims", "Notes": "Zero undefined correlations across 500 claims"},
    {"Category": "Spearman Rho (LIME)", "Metric": "Mean Rho", "Value": "0.5877", "Unit": "correlation", "Notes": "Coarse measure of within-claim rank agreement"},
    {"Category": "Spearman Rho (LIME)", "Metric": "Median Rho", "Value": "0.5000", "Unit": "correlation", "Notes": "50th percentile across claims"},
    {"Category": "Spearman Rho (LIME)", "Metric": "Std Dev (Sample, ddof=1)", "Value": "0.5565", "Unit": "correlation", "Notes": "Sample standard deviation"},
    {"Category": "Spearman Rho (LIME)", "Metric": "Min Rho", "Value": "-1.0000", "Unit": "correlation", "Notes": "Minimum observed"},
    {"Category": "Spearman Rho (LIME)", "Metric": "Q1 (25th Percentile)", "Value": "0.5000", "Unit": "correlation", "Notes": "First quartile"},
    {"Category": "Spearman Rho (LIME)", "Metric": "Q3 (75th Percentile)", "Value": "1.0000", "Unit": "correlation", "Notes": "Third quartile"},
    {"Category": "Spearman Rho (LIME)", "Metric": "Max Rho", "Value": "1.0000", "Unit": "correlation", "Notes": "Maximum observed"},
    {"Category": "Spearman Rho (LIME)", "Metric": "IQR", "Value": "0.5000", "Unit": "correlation", "Notes": "Q3 - Q1"},
    {"Category": "Spearman Rho (LIME)", "Metric": "Positive Correlations (rho > 0)", "Value": "87.00", "Unit": "%", "Notes": "435 / 500 claims"}
]

out_path = "outputs/phase12_analysis/analysis_summary.csv"
with open(out_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["Category", "Metric", "Value", "Unit", "Notes"])
    writer.writeheader()
    writer.writerows(summary_records)

shutil.copyfile(out_path, "analysis_summary.csv")
print("Updated outputs/phase12_analysis/analysis_summary.csv and analysis_summary.csv")
