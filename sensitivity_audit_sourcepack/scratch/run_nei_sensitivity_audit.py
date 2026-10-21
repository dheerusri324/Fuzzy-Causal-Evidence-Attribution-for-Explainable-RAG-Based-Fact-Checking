import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

out_dir = 'outputs/sensitivity_nei_label_audit'
os.makedirs(out_dir, exist_ok=True)

# 1. Load FEVER claim labels
claims_map = {}
with open('data/fever_dev.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 500:
            break
        d = json.loads(line)
        claims_map[d['claim_id']] = d['label']

# 2. Load original Experiment 2 results
df_orig = pd.read_csv('outputs/phase11_formal/results.csv')

# 3. Construct corrected-label dataset
df_sens = df_orig.copy()
df_sens['claim_label'] = df_sens['claim_id'].map(claims_map)

# Protocol A: All 500 claims; NEI evidence treated as 0
df_sens['is_gold_protocol_a'] = df_sens['is_annotated_gold'].copy()
df_sens.loc[df_sens['claim_label'] == 'NOT_ENOUGH_INFO', 'is_gold_protocol_a'] = 0

# Protocol B indicator: True for SUPPORTS/REFUTES, False for NEI
df_sens['in_protocol_b'] = df_sens['claim_label'].isin(['SUPPORTS', 'REFUTES'])
df_sens['is_gold_protocol_b'] = np.where(df_sens['in_protocol_b'], df_sens['is_annotated_gold'], np.nan)

# Save corrected dataset
csv_dest = os.path.join(out_dir, 'results_nei_sensitivity.csv')
df_sens.to_csv(csv_dest, index=False)
print(f'Saved corrected dataset to {csv_dest} ({len(df_sens)} rows)')

# Define evaluation functions
def run_cluster_bootstrap(df_sub, target_col, score_fuzzy_col, score_base_col, n_bootstraps=1000, seed=42):
    np.random.seed(seed)
    claim_ids = df_sub['claim_id'].unique()
    claim_dict = {cid: df_sub[df_sub['claim_id'] == cid] for cid in claim_ids}
    
    y_true = df_sub[target_col].values
    s_f = df_sub[score_fuzzy_col].values
    s_b = df_sub[score_base_col].values
    
    hat_auroc_f = roc_auc_score(y_true, s_f)
    hat_auroc_b = roc_auc_score(y_true, s_b)
    hat_delta_auroc = hat_auroc_f - hat_auroc_b
    
    hat_ap_f = average_precision_score(y_true, s_f)
    hat_ap_b = average_precision_score(y_true, s_b)
    hat_delta_ap = hat_ap_f - hat_ap_b
    
    boot_delta_auroc = []
    boot_delta_ap = []
    
    for b in range(n_bootstraps):
        sampled_cids = np.random.choice(claim_ids, size=len(claim_ids), replace=True)
        boot_df = pd.concat([claim_dict[cid] for cid in sampled_cids], ignore_index=True)
        
        y_boot = boot_df[target_col].values
        # Check if at least one positive and one negative exists
        if len(np.unique(y_boot)) < 2:
            continue
            
        b_auroc_f = roc_auc_score(y_boot, boot_df[score_fuzzy_col].values)
        b_auroc_b = roc_auc_score(y_boot, boot_df[score_base_col].values)
        boot_delta_auroc.append(b_auroc_f - b_auroc_b)
        
        b_ap_f = average_precision_score(y_boot, boot_df[score_fuzzy_col].values)
        b_ap_b = average_precision_score(y_boot, boot_df[score_base_col].values)
        boot_delta_ap.append(b_ap_f - b_ap_b)
        
    boot_delta_auroc = np.array(boot_delta_auroc)
    boot_delta_ap = np.array(boot_delta_ap)
    n_valid = len(boot_delta_auroc)
    
    def get_stats(boot_deltas, hat_delta):
        # Null shift centering: delta_tilde = delta* - hat_delta
        tilde_deltas = boot_deltas - hat_delta
        # Two sided empirical p-value
        p_val = (1.0 + np.sum(np.abs(tilde_deltas) >= np.abs(hat_delta))) / (n_valid + 1.0)
        ci_95 = [float(np.percentile(boot_deltas, 2.5)), float(np.percentile(boot_deltas, 97.5))]
        ci_975 = [float(np.percentile(boot_deltas, 1.25)), float(np.percentile(boot_deltas, 98.75))]
        return p_val, ci_95, ci_975

    p_auroc, ci95_auroc, ci975_auroc = get_stats(boot_delta_auroc, hat_delta_auroc)
    p_ap, ci95_ap, ci975_ap = get_stats(boot_delta_ap, hat_delta_ap)
    
    # Holm-Bonferroni correction for co-primary metrics (m=2)
    hypotheses = [
        ('H1_AUROC', p_auroc, hat_delta_auroc, hat_auroc_f, hat_auroc_b, ci95_auroc, ci975_auroc),
        ('H2_AP', p_ap, hat_delta_ap, hat_ap_f, hat_ap_b, ci95_ap, ci975_ap)
    ]
    # Sort by raw p-value ascending
    hypotheses.sort(key=lambda x: x[1])
    
    # Step-down
    p_adj_1 = min(1.0, 2.0 * hypotheses[0][1])
    p_adj_2 = min(1.0, max(p_adj_1, 1.0 * hypotheses[1][1]))
    
    results = {}
    for i, (name, raw_p, delta, f_pt, b_pt, ci95, ci975) in enumerate(hypotheses):
        adj_p = p_adj_1 if i == 0 else p_adj_2
        step_alpha = 0.025 if i == 0 else 0.05
        results[name] = {
            'point_estimate_fuzzy': float(f_pt),
            'point_estimate_base': float(b_pt),
            'delta_hat': float(delta),
            'ci_95_marginal': ci95,
            'ci_975_bonferroni': ci975,
            'p_value_raw': float(raw_p),
            'p_value_holm_adjusted': float(adj_p),
            'step_alpha': step_alpha,
            'statistically_significant': bool(adj_p < step_alpha),
            'replicates_valid': n_valid
        }
    return results

def compute_top1_metrics(df_sub, target_col):
    n_claims = df_sub['claim_id'].nunique()
    
    # Rank 1 selections
    top1_fuzzy = df_sub[df_sub['fuzzy_top1'] == True]
    top1_base = df_sub[df_sub['baseline_top1'] == True]
    
    # Metric A: Denominator = all claims in set
    hit_a_fuzzy = int(top1_fuzzy[target_col].sum())
    hit_a_base = int(top1_base[target_col].sum())
    rate_a_fuzzy = hit_a_fuzzy / n_claims
    rate_a_base = hit_a_base / n_claims
    
    # Metric B: Claims with gold present in top-3
    claims_with_gold = set(df_sub[df_sub[target_col] == 1]['claim_id'].unique())
    n_gold_claims = len(claims_with_gold)
    
    top1_fuzzy_gold = top1_fuzzy[top1_fuzzy['claim_id'].isin(claims_with_gold)]
    top1_base_gold = top1_base[top1_base['claim_id'].isin(claims_with_gold)]
    
    hit_b_fuzzy = int(top1_fuzzy_gold[target_col].sum())
    hit_b_base = int(top1_base_gold[target_col].sum())
    rate_b_fuzzy = hit_b_fuzzy / n_gold_claims if n_gold_claims > 0 else 0.0
    rate_b_base = hit_b_base / n_gold_claims if n_gold_claims > 0 else 0.0
    
    # Metric C: Direct agreement
    agreed_claims = 0
    for cid, grp in df_sub.groupby('claim_id'):
        f_top1_id = grp[grp['fuzzy_top1'] == True]['evidence_id'].iloc[0]
        b_top1_id = grp[grp['baseline_top1'] == True]['evidence_id'].iloc[0]
        if f_top1_id == b_top1_id:
            agreed_claims += 1
    rate_c = agreed_claims / n_claims
    
    return {
        'metric_a_all_claims': {
            'denominator': n_claims,
            'fuzzy_hits': hit_a_fuzzy,
            'fuzzy_rate': float(rate_a_fuzzy),
            'baseline_hits': hit_a_base,
            'baseline_rate': float(rate_a_base),
            'difference': float(rate_a_fuzzy - rate_a_base)
        },
        'metric_b_gold_present_claims': {
            'denominator': n_gold_claims,
            'fuzzy_hits': hit_b_fuzzy,
            'fuzzy_rate': float(rate_b_fuzzy),
            'baseline_hits': hit_b_base,
            'baseline_rate': float(rate_b_base),
            'difference': float(rate_b_fuzzy - rate_b_base)
        },
        'metric_c_direct_agreement': {
            'denominator': n_claims,
            'agreed_claims': agreed_claims,
            'agreement_rate': float(rate_c)
        }
    }

print('Running bootstrap for Protocol A (500 claims)...')
proto_a_dyn = run_cluster_bootstrap(df_sens, 'is_gold_protocol_a', 'fuzzy_evidence_reliability', 'baseline_adjusted_score')
proto_a_stat = run_cluster_bootstrap(df_sens, 'is_gold_protocol_a', 'fuzzy_evidence_reliability_static', 'baseline_adjusted_score_static')
top1_a = compute_top1_metrics(df_sens, 'is_gold_protocol_a')

print('Running bootstrap for Protocol B (334 claims)...')
df_proto_b = df_sens[df_sens['in_protocol_b']].copy()
proto_b_dyn = run_cluster_bootstrap(df_proto_b, 'is_annotated_gold', 'fuzzy_evidence_reliability', 'baseline_adjusted_score')
proto_b_stat = run_cluster_bootstrap(df_proto_b, 'is_annotated_gold', 'fuzzy_evidence_reliability_static', 'baseline_adjusted_score_static')
top1_b = compute_top1_metrics(df_proto_b, 'is_annotated_gold')

summary = {
    'audit_title': 'NEI Evidence Label Sensitivity Audit',
    'timestamp': '2026-10-20T15:35:00Z',
    'evaluation_scope': 'Closed candidate pool of 808 FEVER passages',
    'protocol_a_all_500_claims_nei_as_zero': {
        'description': 'All 500 claims evaluated; 147 NEI evidence rows reclassified as distractor (is_gold=0)',
        'sample_sizes': {
            'total_claims': 500,
            'total_evidence_rows': 1500,
            'gold_instances': int((df_sens['is_gold_protocol_a'] == 1).sum()),
            'distractor_instances': int((df_sens['is_gold_protocol_a'] == 0).sum()),
            'claims_with_gold_in_top3': int(df_sens[df_sens['is_gold_protocol_a'] == 1]['claim_id'].nunique()),
            'claims_with_zero_gold_in_top3': int(500 - df_sens[df_sens['is_gold_protocol_a'] == 1]['claim_id'].nunique())
        },
        'dynamic_pipeline_results': proto_a_dyn,
        'static_ablation_results': proto_a_stat,
        'top1_alignment_metrics': top1_a
    },
    'protocol_b_supports_refutes_only_334_claims': {
        'description': 'Excludes all 166 NEI claims; evaluates only 168 SUPPORTS and 166 REFUTES claims with verifiable gold evidence',
        'sample_sizes': {
            'total_claims': 334,
            'total_evidence_rows': 1002,
            'gold_instances': int((df_proto_b['is_annotated_gold'] == 1).sum()),
            'distractor_instances': int((df_proto_b['is_annotated_gold'] == 0).sum()),
            'claims_with_gold_in_top3': int(df_proto_b[df_proto_b['is_annotated_gold'] == 1]['claim_id'].nunique()),
            'claims_with_zero_gold_in_top3': int(334 - df_proto_b[df_proto_b['is_annotated_gold'] == 1]['claim_id'].nunique())
        },
        'dynamic_pipeline_results': proto_b_dyn,
        'static_ablation_results': proto_b_stat,
        'top1_alignment_metrics': top1_b
    }
}

summary_dest = os.path.join(out_dir, 'summary_sensitivity.json')
with open(summary_dest, 'w', encoding='utf-8') as fp:
    json.dump(summary, fp, indent=2)

print(f'Saved sensitivity summary to {summary_dest}')
