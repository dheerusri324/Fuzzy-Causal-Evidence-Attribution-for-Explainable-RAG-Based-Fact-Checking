import os
import csv
import json
import numpy as np
import scipy.stats as stats
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

# Set style for publication-quality plots
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300
})

def analyze():
    csv_path = "outputs/phase10_formal/results.csv"
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Total evidence rows: {len(rows)}")

    # 1. Anomaly checks
    anomalies = []
    
    # Trackers
    claims_dict = defaultdict(list)
    verdict_flips_count = 0
    non_flips_count = 0
    verdict_transitions = Counter()
    
    for idx, r in enumerate(rows):
        cid = r["claim_id"]
        eid = r["evidence_id"]
        
        # Check numerical fields
        for field in ["SHAP_score_raw", "LIME_score_raw", "SHAP_magnitude", "LIME_magnitude",
                      "intervention_influence", "baseline_target_probability", "intervened_target_probability"]:
            val_str = r[field]
            if val_str is None or val_str == "":
                anomalies.append(f"Row {idx}: Missing {field}")
            else:
                try:
                    val = float(val_str)
                    if "probability" in field and not (0.0 <= val <= 1.0 + 1e-6):
                        anomalies.append(f"Row {idx}: Probability {field} out of bounds: {val}")
                    if "magnitude" in field and val < -1e-6:
                        anomalies.append(f"Row {idx}: Negative magnitude {field}: {val}")
                    if field == "intervention_influence" and val < -1e-6:
                        anomalies.append(f"Row {idx}: Negative intervention influence: {val}")
                except ValueError:
                    anomalies.append(f"Row {idx}: Cannot parse {field} as float: {val_str}")
                    
        # Check ranks
        for rfield in ["SHAP_magnitude_rank", "LIME_magnitude_rank", "intervention_rank"]:
            try:
                rval = int(r[rfield])
                if rval not in [1, 2, 3]:
                    anomalies.append(f"Row {idx}: Rank {rfield} unexpected: {rval}")
            except ValueError:
                anomalies.append(f"Row {idx}: Cannot parse rank {rfield}: {r[rfield]}")
                
        # Check verdict flip boolean
        v_changed = r["verdict_changed"].strip().lower() == "true"
        if v_changed:
            verdict_flips_count += 1
        else:
            non_flips_count += 1
            
        b_verdict = r["baseline_verdict"]
        i_verdict = r["intervened_verdict"]
        verdict_transitions[(b_verdict, i_verdict)] += 1
        
        claims_dict[cid].append(r)

    print(f"Total unique claims: {len(claims_dict)}")
    print(f"Anomalies found in raw data check: {len(anomalies)}")
    if anomalies:
        for a in anomalies[:10]:
            print("  ", a)

    # Check evidence duplicates per claim and count per claim
    claims_with_diff_counts = {}
    claim_duplicate_eids = {}
    for cid, ev_list in claims_dict.items():
        if len(ev_list) != 3:
            claims_with_diff_counts[cid] = len(ev_list)
        eids = [x["evidence_id"] for x in ev_list]
        if len(set(eids)) != len(eids):
            claim_duplicate_eids[cid] = eids

    print(f"Claims with evidence count != 3: {len(claims_with_diff_counts)}")
    print(f"Claims with duplicate evidence IDs: {len(claim_duplicate_eids)}")

    # 2. Per-claim re-evaluation and verification
    per_claim_stats = []
    
    for cid, ev_list in claims_dict.items():
        raw_shap = [float(x["SHAP_score_raw"]) for x in ev_list]
        raw_lime = [float(x["LIME_score_raw"]) for x in ev_list]
        abs_shap = [float(x["SHAP_magnitude"]) for x in ev_list]
        abs_lime = [float(x["LIME_magnitude"]) for x in ev_list]
        inf_scores = [float(x["intervention_influence"]) for x in ev_list]
        
        # Verify stored ranks match recomputed ranks
        re_shap_ranks = stats.rankdata([-x for x in abs_shap], method='min')
        re_lime_ranks = stats.rankdata([-x for x in abs_lime], method='min')
        re_inf_ranks = stats.rankdata([-x for x in inf_scores], method='min')
        
        stored_shap_ranks = [int(x["SHAP_magnitude_rank"]) for x in ev_list]
        stored_lime_ranks = [int(x["LIME_magnitude_rank"]) for x in ev_list]
        stored_inf_ranks = [int(x["intervention_rank"]) for x in ev_list]
        
        if list(re_shap_ranks) != stored_shap_ranks:
            anomalies.append(f"Claim {cid}: stored SHAP ranks {stored_shap_ranks} != recomputed {list(re_shap_ranks)}")
        if list(re_lime_ranks) != stored_lime_ranks:
            anomalies.append(f"Claim {cid}: stored LIME ranks {stored_lime_ranks} != recomputed {list(re_lime_ranks)}")
        if list(re_inf_ranks) != stored_inf_ranks:
            anomalies.append(f"Claim {cid}: stored Infl ranks {stored_inf_ranks} != recomputed {list(re_inf_ranks)}")

        # Correlation
        with np.errstate(all='ignore'):
            rho_shap, _ = stats.spearmanr(abs_shap, inf_scores)
            rho_lime, _ = stats.spearmanr(abs_lime, inf_scores)
            
        # Top-1 agreement
        shap_top1 = set(i for i, r in enumerate(re_shap_ranks) if r == 1)
        lime_top1 = set(i for i, r in enumerate(re_lime_ranks) if r == 1)
        inf_top1 = set(i for i, r in enumerate(re_inf_ranks) if r == 1)
        
        top1_shap_agree = bool(shap_top1 & inf_top1)
        top1_lime_agree = bool(lime_top1 & inf_top1)
        
        # Verdict flips count in this claim
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

    # 3. Analyze distributions
    valid_shap_rhos = [s["rho_shap"] for s in per_claim_stats if not np.isnan(s["rho_shap"])]
    valid_lime_rhos = [s["rho_lime"] for s in per_claim_stats if not np.isnan(s["rho_lime"])]
    
    print("\n--- CORRELATION VERIFICATION ---")
    print(f"Valid SHAP rhos: {len(valid_shap_rhos)}, Undefined: {len(per_claim_stats) - len(valid_shap_rhos)}")
    print(f"Valid LIME rhos: {len(valid_lime_rhos)}, Undefined: {len(per_claim_stats) - len(valid_lime_rhos)}")
    
    def get_distribution_stats(arr):
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

    shap_dist = get_distribution_stats(valid_shap_rhos)
    lime_dist = get_distribution_stats(valid_lime_rhos)
    
    print("SHAP Distribution:", json.dumps(shap_dist, indent=2))
    print("LIME Distribution:", json.dumps(lime_dist, indent=2))
    
    # Discrete values breakdown for n=3 Spearman rho
    print("\nSHAP rho value counts (rounded to 4 decimals):")
    shap_counts = Counter(round(x, 4) for x in valid_shap_rhos)
    for val, cnt in sorted(shap_counts.items(), key=lambda x: -x[1]):
        print(f"  rho = {val:+.4f}: {cnt} claims ({cnt/len(valid_shap_rhos)*100:.1f}%)")

    print("\nLIME rho value counts (rounded to 4 decimals):")
    lime_counts = Counter(round(x, 4) for x in valid_lime_rhos)
    for val, cnt in sorted(lime_counts.items(), key=lambda x: -x[1]):
        print(f"  rho = {val:+.4f}: {cnt} claims ({cnt/len(valid_lime_rhos)*100:.1f}%)")

    # 4. Top-1 Agreement Analysis
    shap_top1_count = sum(1 for s in per_claim_stats if s["top1_shap_agree"])
    lime_top1_count = sum(1 for s in per_claim_stats if s["top1_lime_agree"])
    total_claims = len(per_claim_stats)
    
    both_agree = sum(1 for s in per_claim_stats if s["top1_shap_agree"] and s["top1_lime_agree"])
    shap_only = sum(1 for s in per_claim_stats if s["top1_shap_agree"] and not s["top1_lime_agree"])
    lime_only = sum(1 for s in per_claim_stats if not s["top1_shap_agree"] and s["top1_lime_agree"])
    neither = sum(1 for s in per_claim_stats if not s["top1_shap_agree"] and not s["top1_lime_agree"])
    
    print("\n--- TOP-1 AGREEMENT VERIFICATION ---")
    print(f"SHAP Top-1 Agreement: {shap_top1_count} / {total_claims} ({shap_top1_count/total_claims*100:.2f}%)")
    print(f"LIME Top-1 Agreement: {lime_top1_count} / {total_claims} ({lime_top1_count/total_claims*100:.2f}%)")
    print(f"Both agree: {both_agree} ({both_agree/total_claims*100:.2f}%)")
    print(f"SHAP only agrees: {shap_only} ({shap_only/total_claims*100:.2f}%)")
    print(f"LIME only agrees: {lime_only} ({lime_only/total_claims*100:.2f}%)")
    print(f"Neither agrees: {neither} ({neither/total_claims*100:.2f}%)")

    # 5. Verdict Flips Analysis
    print("\n--- VERDICT FLIPS VERIFICATION ---")
    total_removals = len(rows)
    print(f"Total removals: {total_removals}")
    print(f"Flips: {verdict_flips_count} ({verdict_flips_count/total_removals*100:.2f}%)")
    print(f"Non-flips: {non_flips_count} ({non_flips_count/total_removals*100:.2f}%)")
    
    # Flips per claim breakdown
    claim_flips_dist = Counter(s["flips_in_claim"] for s in per_claim_stats)
    print("Flips per claim distribution (out of 3 removals):")
    for k in sorted(claim_flips_dist.keys()):
        cnt = claim_flips_dist[k]
        print(f"  {k} flips: {cnt} claims ({cnt/total_claims*100:.2f}%)")
        
    print("\nVerdict transition matrix (baseline -> intervened):")
    for (bv, iv), cnt in sorted(verdict_transitions.items(), key=lambda x: -x[1]):
        print(f"  {bv} -> {iv}: {cnt} ({cnt/total_removals*100:.2f}%)")

if __name__ == "__main__":
    analyze()
