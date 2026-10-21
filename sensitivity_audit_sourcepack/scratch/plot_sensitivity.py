import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
import pandas as pd

plots_dir = 'outputs/sensitivity_nei_label_audit/plots'
os.makedirs(plots_dir, exist_ok=True)

df = pd.read_csv('outputs/sensitivity_nei_label_audit/results_nei_sensitivity.csv')
df_b = df[df['in_protocol_b']].copy()

# Plot 1: ROC Curves for Protocol A & Protocol B
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Protocol A
y_a = df['is_gold_protocol_a'].values
fpr_f_a, tpr_f_a, _ = roc_curve(y_a, df['fuzzy_evidence_reliability'])
fpr_b_a, tpr_b_a, _ = roc_curve(y_a, df['baseline_adjusted_score'])
auc_f_a = roc_auc_score(y_a, df['fuzzy_evidence_reliability'])
auc_b_a = roc_auc_score(y_a, df['baseline_adjusted_score'])

ax1.plot(fpr_f_a, tpr_f_a, color='#1f77b4', lw=2.5, label=f'Fuzzy Reliability (AUROC = {auc_f_a:.4f})')
ax1.plot(fpr_b_a, tpr_b_a, color='#ff7f0e', lw=2.5, linestyle='--', label=f'Adjusted Baseline (AUROC = {auc_b_a:.4f})')
ax1.plot([0, 1], [0, 1], color='gray', linestyle=':')
ax1.set_title('Protocol A: All 500 Claims (NEI as Distractor)', fontsize=12, fontweight='bold')
ax1.set_xlabel('False Positive Rate', fontsize=11)
ax1.set_ylabel('True Positive Rate', fontsize=11)
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)

# Protocol B
y_b = df_b['is_annotated_gold'].values
fpr_f_b, tpr_f_b, _ = roc_curve(y_b, df_b['fuzzy_evidence_reliability'])
fpr_b_b, tpr_b_b, _ = roc_curve(y_b, df_b['baseline_adjusted_score'])
auc_f_b = roc_auc_score(y_b, df_b['fuzzy_evidence_reliability'])
auc_b_b = roc_auc_score(y_b, df_b['baseline_adjusted_score'])

ax2.plot(fpr_f_b, tpr_f_b, color='#1f77b4', lw=2.5, label=f'Fuzzy Reliability (AUROC = {auc_f_b:.4f})')
ax2.plot(fpr_b_b, tpr_b_b, color='#ff7f0e', lw=2.5, linestyle='--', label=f'Adjusted Baseline (AUROC = {auc_b_b:.4f})')
ax2.plot([0, 1], [0, 1], color='gray', linestyle=':')
ax2.set_title('Protocol B: Verifiable Claims Only (SUPPORTS & REFUTES)', fontsize=12, fontweight='bold')
ax2.set_xlabel('False Positive Rate', fontsize=11)
ax2.set_ylabel('True Positive Rate', fontsize=11)
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
roc_path = os.path.join(plots_dir, 'sensitivity_roc_curves.png')
plt.savefig(roc_path, dpi=300)
plt.close()
print(f'Saved sensitivity ROC curves to {roc_path}')

# Plot 2: Precision-Recall Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Protocol A PR
p_f_a, r_f_a, _ = precision_recall_curve(y_a, df['fuzzy_evidence_reliability'])
p_b_a, r_b_a, _ = precision_recall_curve(y_a, df['baseline_adjusted_score'])
ap_f_a = average_precision_score(y_a, df['fuzzy_evidence_reliability'])
ap_b_a = average_precision_score(y_a, df['baseline_adjusted_score'])
base_rate_a = y_a.mean()

ax1.plot(r_f_a, p_f_a, color='#1f77b4', lw=2.5, label=f'Fuzzy (AP = {ap_f_a:.4f})')
ax1.plot(r_b_a, p_b_a, color='#ff7f0e', lw=2.5, linestyle='--', label=f'Baseline (AP = {ap_b_a:.4f})')
ax1.axhline(base_rate_a, color='gray', linestyle=':', label=f'Base Rate ({base_rate_a:.3f})')
ax1.set_title('Protocol A: Precision-Recall (500 Claims)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Recall', fontsize=11)
ax1.set_ylabel('Precision', fontsize=11)
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)

# Protocol B PR
p_f_b, r_f_b, _ = precision_recall_curve(y_b, df_b['fuzzy_evidence_reliability'])
p_b_b, r_b_b, _ = precision_recall_curve(y_b, df_b['baseline_adjusted_score'])
ap_f_b = average_precision_score(y_b, df_b['fuzzy_evidence_reliability'])
ap_b_b = average_precision_score(y_b, df_b['baseline_adjusted_score'])
base_rate_b = y_b.mean()

ax2.plot(r_f_b, p_f_b, color='#1f77b4', lw=2.5, label=f'Fuzzy (AP = {ap_f_b:.4f})')
ax2.plot(r_b_b, p_b_b, color='#ff7f0e', lw=2.5, linestyle='--', label=f'Baseline (AP = {ap_b_b:.4f})')
ax2.axhline(base_rate_b, color='gray', linestyle=':', label=f'Base Rate ({base_rate_b:.3f})')
ax2.set_title('Protocol B: Precision-Recall (334 Verifiable Claims)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Recall', fontsize=11)
ax2.set_ylabel('Precision', fontsize=11)
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
pr_path = os.path.join(plots_dir, 'sensitivity_pr_curves.png')
plt.savefig(pr_path, dpi=300)
plt.close()
print(f'Saved sensitivity PR curves to {pr_path}')
