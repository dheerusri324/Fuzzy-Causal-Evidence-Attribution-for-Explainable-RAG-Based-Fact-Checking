"""
Experiment 2 Formal Evaluation Script.
Phase 11: Comparative Evaluation of Fuzzy Evidence Reasoning vs. Linear Baseline.

Protocol: outputs/phase11/proposed_protocol.json & PHASE_11_EXPERIMENT_2_DESIGN_AUDIT.md
Authoritative Frozen Data:
  - outputs/phase10_formal/results.csv
  - outputs/phase10_formal/config.json
  - data/fever_dev.jsonl
  - data/evidence_corpus.json
"""

import os
import sys
import json
import csv
import time
import datetime
import warnings
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

# Ensure local imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import Config
from data.schema import load_records, ClaimRecord
from retrieval.retriever import EvidenceRetriever
from verification.verifier import ClaimEvidenceVerifier
from attribution.explainer import EvidenceExplainer
from conflict.detector import ConflictDetector
from stability.analyzer import StabilityAnalyzer
from fuzzy.schema import FuzzyInput
from fuzzy.normalization import compute_attribution_strength
from fuzzy.engine import FuzzyEvidenceEngine
from fuzzy.baseline import compute_baseline_score


# ---------------------------------------------------------------------------
# 1. Pre-Run Integrity Validation
# ---------------------------------------------------------------------------

def validate_pre_run_integrity(
    df_exp1: pd.DataFrame, 
    records: List[ClaimRecord], 
    retriever: EvidenceRetriever, 
    cfg: Config
) -> Tuple[Dict[str, Any], Dict[str, List[Any]]]:
    """
    Validates the frozen reference dataset and confirms 100% retrieval alignment
    across all 500 claims before computing any scores.
    """
    print("\n" + "=" * 70)
    print("STEP 1: PRE-RUN INTEGRITY CHECKS")
    print("=" * 70)

    # Check A.1: Unique claims count
    claims = df_exp1['claim_id'].unique()
    assert len(claims) == 500, f"Integrity check failed: Expected 500 claims, found {len(claims)}"
    print(f"[PASS] Unique claims evaluated in Experiment 1: {len(claims)} (expected: 500)")

    # Check A.2: Total rows count
    assert len(df_exp1) == 1500, f"Integrity check failed: Expected 1500 rows, found {len(df_exp1)}"
    print(f"[PASS] Total evidence records in Experiment 1: {len(df_exp1)} (expected: 1500)")

    # Check A.3: Exactly 3 records per claim
    claim_counts = df_exp1.groupby('claim_id').size()
    assert (claim_counts == 3).all(), "Integrity check failed: Claims do not all have exactly 3 evidence records"
    print("[PASS] Exactly 3 evidence records present for every evaluated claim")

    # Check A.4: No duplicate evidence IDs within an individual claim
    dup_counts = df_exp1.groupby('claim_id')['evidence_id'].nunique()
    assert (dup_counts == 3).all(), "Integrity check failed: Duplicate evidence IDs found within claim"
    print("[PASS] Zero duplicate evidence IDs within any claim")

    # Check A.5: Attribution and intervention values are finite and valid
    assert df_exp1['SHAP_score_raw'].notna().all(), "NaN in SHAP_score_raw"
    assert df_exp1['LIME_score_raw'].notna().all(), "NaN in LIME_score_raw"
    assert df_exp1['intervention_influence'].notna().all(), "NaN in intervention_influence"
    assert (df_exp1['intervention_influence'] >= 0.0).all() and (df_exp1['intervention_influence'] <= 1.0).all(), \
        "intervention_influence out of bounds [0, 1]"
    print("[PASS] All SHAP, LIME, and intervention influence values are finite and valid")

    # Check A.6: Gold evidence derivation
    record_map = {str(r.claim_id): r for r in records}
    gold_matches = 0
    gold_flags = []
    claims_with_gold = set()

    for _, row in df_exp1.iterrows():
        cid = str(row['claim_id'])
        eid = str(row['evidence_id'])
        rec = record_map[cid]
        gold_ids = {ep.evidence_id for ep in rec.evidence}
        is_gold = 1 if eid in gold_ids else 0
        gold_flags.append(is_gold)
        if is_gold == 1:
            gold_matches += 1
            claims_with_gold.add(cid)

    assert gold_matches == 443, f"Integrity check failed: Expected 443 gold matches, found {gold_matches}"
    assert len(claims_with_gold) == 430, f"Integrity check failed: Expected 430 claims with >= 1 gold, found {len(claims_with_gold)}"
    print(f"[PASS] Gold-evidence proxy labels: exactly {gold_matches} gold instances (29.53%) and {1500 - gold_matches} distractors")
    print(f"[PASS] Claims containing >= 1 gold in Top-3: {len(claims_with_gold)} / 500 (86.00%)")

    # Check B: 100% hybrid retrieval alignment across ALL 500 claims
    print("\nValidating 100% hybrid retrieval alignment across all 500 claims...")
    retrieved_map = {}
    mismatches = []

    for rec in records:
        cid = str(rec.claim_id)
        claim_df = df_exp1[df_exp1['claim_id'] == cid]
        csv_eids = list(claim_df['evidence_id'])
        
        evs = retriever.retrieve(rec.claim_text, top_k=cfg.top_k, mode=cfg.retrieval_mode)
        ret_eids = [ev.evidence_id for ev in evs]
        
        if csv_eids != ret_eids:
            mismatches.append((cid, csv_eids, ret_eids))
        
        retrieved_map[cid] = evs

    if mismatches:
        print(f"\n[FATAL ERROR] Alignment check failed! {len(mismatches)} claims had retrieval mismatches:")
        for cid, c_eids, r_eids in mismatches[:5]:
            print(f"  Claim {cid}: CSV={c_eids} vs RETRIEVED={r_eids}")
        sys.exit(1)

    print("[PASS] Full 500-claim retrieval alignment PASSED with 100.0% concordant ordering!")
    
    validation_info = {
        "status": "PASSED_100_PERCENT",
        "claims_validated": len(claims),
        "evidence_records_validated": len(df_exp1),
        "gold_evidence_matches": gold_matches,
        "claims_with_gold_in_top3": len(claims_with_gold),
        "claims_with_zero_gold_in_top3": len(claims) - len(claims_with_gold),
        "all_checks_passed": True
    }
    
    return validation_info, retrieved_map


# ---------------------------------------------------------------------------
# 2. Dynamic Conflict and Stability Computation with Caching
# ---------------------------------------------------------------------------

def compute_dynamic_signals(
    records: List[ClaimRecord],
    retrieved_map: Dict[str, List[Any]],
    cfg: Config,
    cache_path: str = "outputs/phase11_formal/.signals_cache.json"
) -> Dict[str, Dict[str, float]]:
    """
    Computes dynamic contradiction degree (ConflictDetector) and explanation
    order-permutation stability (StabilityAnalyzer) for all 500 claims.
    Uses disk caching to support safe resumption and reproducible execution.
    """
    print("\n" + "=" * 70)
    print("STEP 2: DYNAMIC CONFLICT & STABILITY SENSING")
    print("=" * 70)

    cached_signals = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_signals = json.load(f)
            print(f"Loaded {len(cached_signals)} pre-computed dynamic signal records from cache: {cache_path}")
        except Exception as e:
            print(f"Warning: could not read cache file ({e}), recomputing...")

    remaining_records = [r for r in records if str(r.claim_id) not in cached_signals]
    
    if remaining_records:
        print(f"Computing dynamic signals for {len(remaining_records)} remaining claims...")
        verifier = ClaimEvidenceVerifier(config=cfg)
        explainer = EvidenceExplainer(verifier=verifier, config=cfg)
        conflict_det = ConflictDetector(verifier=verifier, config=cfg)
        stability_ana = StabilityAnalyzer(explainer=explainer, config=cfg)

        start_time = time.time()
        for i, rec in enumerate(remaining_records, start=1):
            cid = str(rec.claim_id)
            evs = retrieved_map[cid]

            # 1. Conflict detection
            conf_res = conflict_det.detect(cid, rec.claim_text, evs)
            c_score = float(conf_res.conflict_score)

            # 2. Stability analysis (order permutation)
            stab_res = stability_ana.analyze(cid, rec.claim_text, evs)
            s_score = float(stab_res.stability_score)

            cached_signals[cid] = {
                "conflict_score": c_score,
                "stability_score": s_score
            }

            if i % 25 == 0 or i == len(remaining_records):
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(remaining_records) - i) / rate if rate > 0 else 0
                print(f"  Processed {i}/{len(remaining_records)} claims ({i/len(remaining_records)*100:.1f}%) | "
                      f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
                # Flush to disk periodically
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cached_signals, f, indent=2)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cached_signals, f, indent=2)

    print(f"[PASS] Dynamic signals fully assembled for all {len(cached_signals)} claims.")
    return cached_signals


# ---------------------------------------------------------------------------
# 3. Deterministic Within-Claim Ranking & Tie-Breaking
# ---------------------------------------------------------------------------

def compute_within_claim_ranks(
    items: List[Dict[str, Any]], 
    score_key: str
) -> List[int]:
    """
    Computes deterministic within-claim ranks (1 = highest priority, 3 = lowest priority).
    Deterministic tie-breaking order:
      1. Primary: Score descending (higher score = better rank)
      2. Secondary: Normalized retrieval score descending (higher relevance = better rank)
      3. Tertiary: Original retrieval rank ascending (lower index = better rank)
    """
    # Sort items based on the 3 keys:
    # Key: (-score, -retrieval_score, retrieval_rank)
    indexed_items = list(enumerate(items))
    sorted_items = sorted(
        indexed_items,
        key=lambda pair: (
            -pair[1][score_key],
            -pair[1]["retrieval_score"],
            pair[1]["retrieval_rank"]
        )
    )
    
    ranks = [0] * len(items)
    for rank_idx, (orig_idx, _) in enumerate(sorted_items, start=1):
        ranks[orig_idx] = rank_idx
        
    return ranks


# ---------------------------------------------------------------------------
# 4. Null-Centered Paired Cluster Bootstrap
# ---------------------------------------------------------------------------

def run_null_centered_cluster_bootstrap(
    df: pd.DataFrame,
    metric_name: str,
    metric_func: Any,
    score_col_fuzzy: str,
    score_col_base: str,
    target_col: str = "is_annotated_gold",
    claim_id_col: str = "claim_id",
    B: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Performs an exact Null-Centered Paired Claim-Level Cluster Bootstrap Test.
    Resamples claims with replacement (preserving multiplicity) and keeps all
    three associated evidence records together.
    """
    claims = np.array(sorted(df[claim_id_col].unique()))
    M = len(claims)
    
    # Pre-index data by claim for high-performance resampling
    claim_dict = {cid: group for cid, group in df.groupby(claim_id_col)}

    # Point estimates on full sample
    y_full = df[target_col].values
    s_fuzzy_full = df[score_col_fuzzy].values
    s_base_full = df[score_col_base].values

    val_fuzzy = float(metric_func(y_full, s_fuzzy_full))
    val_base = float(metric_func(y_full, s_base_full))
    delta_hat = float(val_fuzzy - val_base)

    # Initialize deterministic random generator
    rng = np.random.Generator(np.random.PCG64(seed))

    bootstrap_deltas = []
    degenerate_count = 0

    for b in range(B):
        # Draw M claim indices with replacement
        sampled_cids = rng.choice(claims, size=M, replace=True)
        
        # Concatenate rows preserving multiplicity
        sampled_rows = pd.concat([claim_dict[cid] for cid in sampled_cids], ignore_index=True)
        
        y_samp = sampled_rows[target_col].values
        pos_count = np.sum(y_samp)
        
        # Check for degenerate replicate
        if pos_count == 0 or pos_count == len(y_samp):
            degenerate_count += 1
            continue

        s_fuzzy_samp = sampled_rows[score_col_fuzzy].values
        s_base_samp = sampled_rows[score_col_base].values

        val_fuzzy_b = metric_func(y_samp, s_fuzzy_samp)
        val_base_b = metric_func(y_samp, s_base_samp)
        delta_b = float(val_fuzzy_b - val_base_b)

        bootstrap_deltas.append(delta_b)

    B_valid = len(bootstrap_deltas)
    assert B_valid > 0, "All bootstrap replicates were degenerate!"

    deltas_star = np.array(bootstrap_deltas)

    # Shift under null: delta_tilde = delta_star - delta_hat
    deltas_tilde = deltas_star - delta_hat

    # Two-sided empirical p-value with finite-replicate correction
    numerator = 1.0 + np.sum(np.abs(deltas_tilde) >= np.abs(delta_hat))
    p_boot = float(numerator / (B_valid + 1.0))
    p_boot = min(1.0, max(0.0, p_boot))

    # Marginal 95% Percentile Confidence Interval
    ci_95_lower = float(np.percentile(deltas_star, 2.5, method="linear"))
    ci_95_upper = float(np.percentile(deltas_star, 97.5, method="linear"))

    # Bonferroni-Adjusted 97.5% Percentile Confidence Interval (for m=2 co-primary metrics)
    ci_975_lower = float(np.percentile(deltas_star, 1.25, method="linear"))
    ci_975_upper = float(np.percentile(deltas_star, 98.75, method="linear"))

    return {
        "metric_name": metric_name,
        "point_estimate_fuzzy": val_fuzzy,
        "point_estimate_base": val_base,
        "delta_hat": delta_hat,
        "p_value_raw": p_boot,
        "ci_95_marginal": [ci_95_lower, ci_95_upper],
        "ci_975_bonferroni": [ci_975_lower, ci_975_upper],
        "replicates_requested": B,
        "replicates_valid": B_valid,
        "replicates_degenerate": degenerate_count,
        "bootstrap_deltas": deltas_star.tolist()
    }


# ---------------------------------------------------------------------------
# 5. Holm-Bonferroni Multiple Comparisons Adjustment
# ---------------------------------------------------------------------------

def apply_holm_bonferroni(
    primary_results: Dict[str, Dict[str, Any]], 
    alpha: float = 0.05
) -> Dict[str, Dict[str, Any]]:
    """
    Applies the Holm-Bonferroni step-down procedure across co-primary hypotheses
    to guarantee Family-Wise Error Rate (FWER) control at alpha = 0.05.
    """
    keys = list(primary_results.keys())
    # Sort hypotheses by raw p-value ascending
    sorted_keys = sorted(keys, key=lambda k: primary_results[k]["p_value_raw"])
    m = len(sorted_keys)

    adjusted_results = {}
    prev_adj_p = 0.0

    for rank, key in enumerate(sorted_keys, start=1):
        res = primary_results[key]
        p_raw = res["p_value_raw"]
        step_alpha = alpha / (m - rank + 1)
        
        # Holm adjusted p-value
        p_adj = min(1.0, p_raw * (m - rank + 1))
        p_adj = max(p_adj, prev_adj_p)  # enforce monotonicity
        prev_adj_p = p_adj

        rejected = (p_adj < alpha)

        adjusted_results[key] = {
            **res,
            "hypothesis_id": key,
            "rank_order": rank,
            "step_alpha": step_alpha,
            "p_value_holm_adjusted": p_adj,
            "statistically_significant_at_fwer_005": rejected
        }

    return adjusted_results


# ---------------------------------------------------------------------------
# 6. Primary Execution Function
# ---------------------------------------------------------------------------

def run_formal_experiment_2():
    print("=" * 70)
    print("FORMAL EXPERIMENT 2: FUZZY REASONING VS. LINEAR BASELINE")
    print("=" * 70)
    print("Timestamp: ", datetime.datetime.now().isoformat())
    print("Python:    ", sys.version)

    output_dir = "outputs/phase11_formal"
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # 1. Config and Data Loading
    cfg = Config(top_k=3, shap_num_samples=25, lime_num_samples=25)
    records = load_records("data/fever_dev.jsonl")[:cfg.max_claims]
    df_exp1 = pd.read_csv("outputs/phase10_formal/results.csv")

    retriever = EvidenceRetriever(config=cfg)

    # 2. Run Pre-Run Integrity Checks
    validation_info, retrieved_map = validate_pre_run_integrity(
        df_exp1, records, retriever, cfg
    )

    # 3. Compute Dynamic Signals (Conflict and Stability) with Caching
    dynamic_signals = compute_dynamic_signals(
        records, retrieved_map, cfg, cache_path=os.path.join(output_dir, ".signals_cache.json")
    )

    # 4. Initialize Inference Engines
    fuzzy_engine = FuzzyEvidenceEngine(config=cfg)

    print("\n" + "=" * 70)
    print("STEP 3: DUAL-CONDITION EVIDENCE SCORING")
    print("=" * 70)

    # Group CSV records by claim
    exp1_claim_groups = {cid: group for cid, group in df_exp1.groupby("claim_id")}

    records_evaluated = []
    
    for rec in records:
        cid = str(rec.claim_id)
        claim_text = rec.claim_text
        c_rows = exp1_claim_groups[cid].to_dict(orient="records")
        evs = retrieved_map[cid]

        # Extract dynamic signals for this claim
        c_deg_dynamic = dynamic_signals[cid]["conflict_score"]
        s_stab_dynamic = dynamic_signals[cid]["stability_score"]

        claim_evaluations = []

        for i, ev in enumerate(evs):
            row = c_rows[i]
            assert ev.evidence_id == row["evidence_id"], f"Evidence ID mismatch at claim {cid}, idx {i}"

            # Base features
            ret_score = float(ev.retrieval_score)
            ret_rank = int(ev.retrieval_rank)
            shap_raw = float(row["SHAP_score_raw"])
            lime_raw = float(row["LIME_score_raw"])
            attr_strength = float(compute_attribution_strength(shap_raw, lime_raw, cfg.fuzzy_shap_weight, cfg.fuzzy_lime_weight))
            interv_infl = float(row["intervention_influence"])
            gold_ids = {ep.evidence_id for ep in rec.evidence}
            is_gold = 1 if ev.evidence_id in gold_ids else 0
            src_rel = 0.75  # Wikipedia default

            # Condition 1: Primary Condition (Dynamic Inputs)
            inp_primary = FuzzyInput(
                relevance=ret_score,
                attribution_shap=shap_raw,
                attribution_lime=lime_raw,
                intervention_influence=interv_infl,
                contradiction_degree=c_deg_dynamic,
                stability_score=s_stab_dynamic,
                source_reliability=src_rel
            )
            fuz_out_primary = fuzzy_engine.evaluate(inp_primary)
            base_out_primary = compute_baseline_score(inp_primary, config=cfg)

            # Condition 2: Secondary Ablation (Static Defaults)
            inp_ablation = FuzzyInput(
                relevance=ret_score,
                attribution_shap=shap_raw,
                attribution_lime=lime_raw,
                intervention_influence=interv_infl,
                contradiction_degree=0.0,
                stability_score=1.0,
                source_reliability=src_rel
            )
            fuz_out_ablation = fuzzy_engine.evaluate(inp_ablation)
            base_out_ablation = compute_baseline_score(inp_ablation, config=cfg)

            # Attribution-gap flags
            in_std_gap = bool((attr_strength >= 0.60) and (interv_infl <= 0.40))
            in_ext_gap = bool((attr_strength >= 0.80) and (interv_infl <= 0.10))
            # Rank-based divergence flag (Attribution rank 1, intervention rank 3)
            in_rank_div_gap = bool((int(row["SHAP_magnitude_rank"]) == 1 or int(row["LIME_magnitude_rank"]) == 1) and (int(row["intervention_rank"]) == 3))

            claim_evaluations.append({
                "claim_id": cid,
                "claim_text": claim_text,
                "evidence_id": ev.evidence_id,
                "evidence_text": ev.text,
                "doc_title": ev.doc_title,
                "sentence_idx": ev.sentence_idx,
                "retrieval_score": ret_score,
                "retrieval_rank": ret_rank,
                "is_annotated_gold": is_gold,
                "SHAP_score_raw": shap_raw,
                "LIME_score_raw": lime_raw,
                "SHAP_magnitude": float(row["SHAP_magnitude"]),
                "LIME_magnitude": float(row["LIME_magnitude"]),
                "attribution_strength": attr_strength,
                "intervention_influence": interv_infl,
                "source_reliability": src_rel,
                "conflict_score_dynamic": c_deg_dynamic,
                "stability_score_dynamic": s_stab_dynamic,
                "conflict_score_static": 0.0,
                "stability_score_static": 1.0,
                
                # Primary condition scores
                "fuzzy_evidence_reliability": float(fuz_out_primary.evidence_reliability),
                "fuzzy_explanation_confidence": float(fuz_out_primary.explanation_confidence),
                "baseline_adjusted_score": float(base_out_primary.adjusted_baseline_score),
                "baseline_raw_score": float(base_out_primary.baseline_score),
                
                # Secondary ablation scores
                "fuzzy_evidence_reliability_static": float(fuz_out_ablation.evidence_reliability),
                "fuzzy_explanation_confidence_static": float(fuz_out_ablation.explanation_confidence),
                "baseline_adjusted_score_static": float(base_out_ablation.adjusted_baseline_score),
                "baseline_raw_score_static": float(base_out_ablation.baseline_score),
                
                # Attribution-gap membership indicators
                "in_standard_gap": in_std_gap,
                "in_extreme_gap": in_ext_gap,
                "in_rank_divergence_gap": in_rank_div_gap
            })

        # Compute deterministic within-claim ranks for both conditions
        # Primary condition ranks
        f_ranks_prim = compute_within_claim_ranks(claim_evaluations, "fuzzy_evidence_reliability")
        b_ranks_prim = compute_within_claim_ranks(claim_evaluations, "baseline_adjusted_score")

        # Static ablation ranks
        f_ranks_stat = compute_within_claim_ranks(claim_evaluations, "fuzzy_evidence_reliability_static")
        b_ranks_stat = compute_within_claim_ranks(claim_evaluations, "baseline_adjusted_score_static")

        for idx in range(len(claim_evaluations)):
            claim_evaluations[idx]["fuzzy_rank"] = f_ranks_prim[idx]
            claim_evaluations[idx]["baseline_rank"] = b_ranks_prim[idx]
            claim_evaluations[idx]["fuzzy_top1"] = (f_ranks_prim[idx] == 1)
            claim_evaluations[idx]["baseline_top1"] = (b_ranks_prim[idx] == 1)

            claim_evaluations[idx]["fuzzy_rank_static"] = f_ranks_stat[idx]
            claim_evaluations[idx]["baseline_rank_static"] = b_ranks_stat[idx]
            claim_evaluations[idx]["fuzzy_top1_static"] = (f_ranks_stat[idx] == 1)
            claim_evaluations[idx]["baseline_top1_static"] = (b_ranks_stat[idx] == 1)

        records_evaluated.extend(claim_evaluations)

    # Convert to DataFrame
    df_results = pd.DataFrame(records_evaluated)
    print(f"[PASS] Successfully scored all {len(df_results)} evidence records across both conditions.")

    # Save full raw CSV output
    results_csv_path = os.path.join(output_dir, "results.csv")
    df_results.to_csv(results_csv_path, index=False)
    print(f"Saved complete evaluation results to: {results_csv_path}")

    # -----------------------------------------------------------------------
    # 5. Compute Primary Discrimination Metrics & Bootstrap Inference
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: CO-PRIMARY METRIC INFERENCE (NULL-CENTERED CLUSTER BOOTSTRAP)")
    print("=" * 70)

    # Condition 1 (Primary: Dynamic)
    boot_auroc_dynamic = run_null_centered_cluster_bootstrap(
        df_results,
        metric_name="AUROC (Dynamic Full Pipeline)",
        metric_func=roc_auc_score,
        score_col_fuzzy="fuzzy_evidence_reliability",
        score_col_base="baseline_adjusted_score",
        B=1000,
        seed=42
    )

    boot_ap_dynamic = run_null_centered_cluster_bootstrap(
        df_results,
        metric_name="Average Precision (Dynamic Full Pipeline)",
        metric_func=average_precision_score,
        score_col_fuzzy="fuzzy_evidence_reliability",
        score_col_base="baseline_adjusted_score",
        B=1000,
        seed=42
    )

    # Apply Holm-Bonferroni correction to co-primary pair
    co_primary_raw = {
        "H1_AUROC": boot_auroc_dynamic,
        "H2_Average_Precision": boot_ap_dynamic
    }
    co_primary_adjusted = apply_holm_bonferroni(co_primary_raw, alpha=0.05)

    print("\n--- Co-Primary Confirmatory Results (Primary Dynamic Condition) ---")
    for hyp_id, res in co_primary_adjusted.items():
        print(f"\n[{hyp_id}] {res['metric_name']}:")
        print(f"  Fuzzy Point Estimate:      {res['point_estimate_fuzzy']:.4f}")
        print(f"  Baseline Point Estimate:   {res['point_estimate_base']:.4f}")
        print(f"  Delta (Fuzzy - Baseline):  {res['delta_hat']:+.4f}")
        print(f"  95% Marginal CI:           [{res['ci_95_marginal'][0]:+.4f}, {res['ci_95_marginal'][1]:+.4f}]")
        print(f"  97.5% Bonferroni CI:       [{res['ci_975_bonferroni'][0]:+.4f}, {res['ci_975_bonferroni'][1]:+.4f}]")
        print(f"  Raw Two-Sided p-value:     {res['p_value_raw']:.4f}")
        print(f"  Holm-Adjusted p-value:     {res['p_value_holm_adjusted']:.4f} (Step alpha: {res['step_alpha']:.4f})")
        print(f"  Statistically Significant: {res['statistically_significant_at_fwer_005']}")

    # Condition 2 (Secondary Ablation: Static Defaults)
    boot_auroc_static = run_null_centered_cluster_bootstrap(
        df_results,
        metric_name="AUROC (Static Default Ablation)",
        metric_func=roc_auc_score,
        score_col_fuzzy="fuzzy_evidence_reliability_static",
        score_col_base="baseline_adjusted_score_static",
        B=1000,
        seed=42
    )

    boot_ap_static = run_null_centered_cluster_bootstrap(
        df_results,
        metric_name="Average Precision (Static Default Ablation)",
        metric_func=average_precision_score,
        score_col_fuzzy="fuzzy_evidence_reliability_static",
        score_col_base="baseline_adjusted_score_static",
        B=1000,
        seed=42
    )

    # -----------------------------------------------------------------------
    # 6. Secondary Metrics Computation
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: SECONDARY METRICS (TOP-1 HIT RATES & AGREEMENT)")
    print("=" * 70)

    def compute_top1_metrics(df_sub: pd.DataFrame, prefix: str = "") -> Dict[str, Any]:
        suf = f"_{prefix}" if prefix else ""
        fuzzy_top1_col = f"fuzzy_top1{suf}"
        base_top1_col = f"baseline_top1{suf}"

        # Metric A: Top-1 Gold Hit Rate over all 500 claims
        top1_fuzzy_rows = df_sub[df_sub[fuzzy_top1_col] == True]
        top1_base_rows = df_sub[df_sub[base_top1_col] == True]

        hit_a_fuzzy = int(top1_fuzzy_rows["is_annotated_gold"].sum())
        hit_a_base = int(top1_base_rows["is_annotated_gold"].sum())
        rate_a_fuzzy = hit_a_fuzzy / 500.0
        rate_a_base = hit_a_base / 500.0

        # Metric B: Top-1 Gold Hit Rate among the 430 claims with >= 1 gold in Top-3
        claims_with_gold_set = set(df_sub[df_sub["is_annotated_gold"] == 1]["claim_id"].unique())
        assert len(claims_with_gold_set) == 430, f"Expected 430 gold claims, got {len(claims_with_gold_set)}"

        df_gold_claims = df_sub[df_sub["claim_id"].isin(claims_with_gold_set)]
        top1_fuzzy_gold_claims = df_gold_claims[df_gold_claims[fuzzy_top1_col] == True]
        top1_base_gold_claims = df_gold_claims[df_gold_claims[base_top1_col] == True]

        hit_b_fuzzy = int(top1_fuzzy_gold_claims["is_annotated_gold"].sum())
        hit_b_base = int(top1_base_gold_claims["is_annotated_gold"].sum())
        rate_b_fuzzy = hit_b_fuzzy / 430.0
        rate_b_base = hit_b_base / 430.0

        # Metric C: Direct Top-1 Agreement Rate between Fuzzy and Baseline
        # Count claims where fuzzy and baseline pick the exact same evidence_id as Rank 1
        agreement_count = 0
        for cid, group in df_sub.groupby("claim_id"):
            f_top1_id = group[group[fuzzy_top1_col] == True]["evidence_id"].iloc[0]
            b_top1_id = group[group[base_top1_col] == True]["evidence_id"].iloc[0]
            if f_top1_id == b_top1_id:
                agreement_count += 1
        rate_c_agreement = agreement_count / 500.0

        return {
            "metric_a_all_claims": {
                "denominator": 500,
                "fuzzy_hits": hit_a_fuzzy,
                "fuzzy_rate": rate_a_fuzzy,
                "baseline_hits": hit_a_base,
                "baseline_rate": rate_a_base,
                "difference": rate_a_fuzzy - rate_a_base
            },
            "metric_b_gold_present_claims": {
                "denominator": 430,
                "fuzzy_hits": hit_b_fuzzy,
                "fuzzy_rate": rate_b_fuzzy,
                "baseline_hits": hit_b_base,
                "baseline_rate": rate_b_base,
                "difference": rate_b_fuzzy - rate_b_base
            },
            "metric_c_direct_agreement": {
                "denominator": 500,
                "agreed_claims": agreement_count,
                "agreement_rate": rate_c_agreement
            }
        }

    top1_dynamic = compute_top1_metrics(df_results, prefix="")
    top1_static = compute_top1_metrics(df_results, prefix="static")

    print("\n--- Top-1 Evidence Alignment (Primary Dynamic Condition) ---")
    print(f"Metric A (All 500 Claims):")
    print(f"  Fuzzy Top-1 Gold Hit Rate:      {top1_dynamic['metric_a_all_claims']['fuzzy_hits']}/500 ({top1_dynamic['metric_a_all_claims']['fuzzy_rate']*100:.2f}%)")
    print(f"  Baseline Top-1 Gold Hit Rate:   {top1_dynamic['metric_a_all_claims']['baseline_hits']}/500 ({top1_dynamic['metric_a_all_claims']['baseline_rate']*100:.2f}%)")
    print(f"  Difference:                     {top1_dynamic['metric_a_all_claims']['difference']*100:+.2f}%")

    print(f"\nMetric B (Claims with >= 1 Gold in Top-3, N=430):")
    print(f"  Fuzzy Top-1 Gold Hit Rate:      {top1_dynamic['metric_b_gold_present_claims']['fuzzy_hits']}/430 ({top1_dynamic['metric_b_gold_present_claims']['fuzzy_rate']*100:.2f}%)")
    print(f"  Baseline Top-1 Gold Hit Rate:   {top1_dynamic['metric_b_gold_present_claims']['baseline_hits']}/430 ({top1_dynamic['metric_b_gold_present_claims']['baseline_rate']*100:.2f}%)")
    print(f"  Difference:                     {top1_dynamic['metric_b_gold_present_claims']['difference']*100:+.2f}%")

    print(f"\nMetric C (Direct Top-1 Agreement Rate):")
    print(f"  Agreed Claims:                  {top1_dynamic['metric_c_direct_agreement']['agreed_claims']}/500 ({top1_dynamic['metric_c_direct_agreement']['agreement_rate']*100:.2f}%)")

    # -----------------------------------------------------------------------
    # 7. Score Distribution Diagnostics
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 6: SCORE DISTRIBUTION DIAGNOSTICS")
    print("=" * 70)

    def compute_distribution_stats(series: pd.Series) -> Dict[str, float]:
        vals = series.values
        q25, q75 = np.percentile(vals, [25, 75])
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
            "iqr": float(q75 - q25),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "saturation_count_0": int(np.sum(vals == 0.0)),
            "saturation_count_1": int(np.sum(vals == 1.0)),
            "saturation_rate": float(np.mean((vals == 0.0) | (vals == 1.0)))
        }

    dist_stats = {
        "fuzzy_evidence_reliability": compute_distribution_stats(df_results["fuzzy_evidence_reliability"]),
        "baseline_adjusted_score": compute_distribution_stats(df_results["baseline_adjusted_score"]),
        "fuzzy_explanation_confidence": compute_distribution_stats(df_results["fuzzy_explanation_confidence"]),
        "baseline_raw_score": compute_distribution_stats(df_results["baseline_raw_score"])
    }

    for col, st in dist_stats.items():
        print(f"  {col:30s} | Mean: {st['mean']:.4f} | Std: {st['std']:.4f} | Median: {st['median']:.4f} | IQR: {st['iqr']:.4f} | Saturation: {st['saturation_rate']*100:.2f}%")

    # -----------------------------------------------------------------------
    # 8. Attribution-Gap Analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 7: ATTRIBUTION-GAP SUBGROUP EVALUATION")
    print("=" * 70)

    # Membership-based subgroups
    std_gap_df = df_results[df_results["in_standard_gap"] == True]
    ext_gap_df = df_results[df_results["in_extreme_gap"] == True]

    print(f"Membership Standard Gap (a >= 0.60, i <= 0.40): N = {len(std_gap_df)} instances")
    print(f"Membership Extreme Gap  (a >= 0.80, i <= 0.10): N = {len(ext_gap_df)} instances")
    print("  Note: As pre-specified in the protocol, because N = 0 in the evaluated dataset,")
    print("        no empirical effect can be estimated for the membership-based gap subgroups.")

    # Rank-based divergence subgroup (Exploratory Only)
    rank_gap_df = df_results[df_results["in_rank_divergence_gap"] == True]
    n_rank_gap = len(rank_gap_df)
    print(f"\nExploratory Rank-Divergence Gap (Attr Rank 1, Interv Rank 3): N = {n_rank_gap} instances ({n_rank_gap/1500*100:.2f}%)")

    rank_gap_summary = {}
    if n_rank_gap > 0:
        mean_conf = float(rank_gap_df["fuzzy_explanation_confidence"].mean())
        median_conf = float(rank_gap_df["fuzzy_explanation_confidence"].median())
        mean_base = float(rank_gap_df["baseline_raw_score"].mean())
        median_base = float(rank_gap_df["baseline_raw_score"].median())

        mean_fuz_rank = float(rank_gap_df["fuzzy_rank"].mean())
        mean_base_rank = float(rank_gap_df["baseline_rank"].mean())

        downweighted_count = int((rank_gap_df["fuzzy_rank"] > rank_gap_df["baseline_rank"]).sum())
        downweighted_rate = float(downweighted_count / n_rank_gap)

        rank_gap_summary = {
            "status": "EXPLORATORY_DESCRIPTIVE_ONLY",
            "sample_size": n_rank_gap,
            "sample_percentage": float(n_rank_gap / 1500.0),
            "explanation_confidence_mean": mean_conf,
            "explanation_confidence_median": median_conf,
            "baseline_raw_score_mean": mean_base,
            "baseline_raw_score_median": median_base,
            "fuzzy_mean_within_claim_rank": mean_fuz_rank,
            "baseline_mean_within_claim_rank": mean_base_rank,
            "fuzzy_rank_downweighted_count": downweighted_count,
            "fuzzy_rank_downweighted_rate": downweighted_rate
        }

        print(f"  Explanation Confidence (Target): Mean={mean_conf:.4f}, Median={median_conf:.4f}")
        print(f"  Baseline Diagnostic Score:      Mean={mean_base:.4f}, Median={median_base:.4f}")
        print(f"  Mean Within-Claim Rank:          Fuzzy={mean_fuz_rank:.2f} vs Baseline={mean_base_rank:.2f}")
        print(f"  Fuzzy Downweighting Frequency:   {downweighted_count}/{n_rank_gap} ({downweighted_rate*100:.2f}%)")
        print("  [EXPLORATORY NOTICE] Confirmatory testing is deferred to an independent validation set.")

    # -----------------------------------------------------------------------
    # 9. Generate Publication-Quality Plots
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 8: GENERATING VISUALIZATION ARTIFACTS")
    print("=" * 70)

    # Plot 1: ROC and PR Curves (Primary Dynamic Condition)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ROC Curves
    fpr_f, tpr_f, _ = roc_curve(df_results["is_annotated_gold"], df_results["fuzzy_evidence_reliability"])
    fpr_b, tpr_b, _ = roc_curve(df_results["is_annotated_gold"], df_results["baseline_adjusted_score"])
    
    ax1.plot(fpr_f, tpr_f, label=f"Fuzzy Reliability (AUC = {boot_auroc_dynamic['point_estimate_fuzzy']:.4f})", color="#1f77b4", lw=2)
    ax1.plot(fpr_b, tpr_b, label=f"Adjusted Baseline (AUC = {boot_auroc_dynamic['point_estimate_base']:.4f})", color="#ff7f0e", lw=2, linestyle="--")
    ax1.plot([0, 1], [0, 1], color="gray", linestyle=":", lw=1.5, label="Random Chance (AUC = 0.50)")
    ax1.set_xlabel("False Positive Rate", fontsize=11)
    ax1.set_ylabel("True Positive Rate", fontsize=11)
    ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # PR Curves
    prec_f, rec_f, _ = precision_recall_curve(df_results["is_annotated_gold"], df_results["fuzzy_evidence_reliability"])
    prec_b, rec_b, _ = precision_recall_curve(df_results["is_annotated_gold"], df_results["baseline_adjusted_score"])
    base_rate = df_results["is_annotated_gold"].mean()

    ax2.plot(rec_f, prec_f, label=f"Fuzzy Reliability (AP = {boot_ap_dynamic['point_estimate_fuzzy']:.4f})", color="#1f77b4", lw=2)
    ax2.plot(rec_b, prec_b, label=f"Adjusted Baseline (AP = {boot_ap_dynamic['point_estimate_base']:.4f})", color="#ff7f0e", lw=2, linestyle="--")
    ax2.axhline(base_rate, color="gray", linestyle=":", lw=1.5, label=f"Prevalence Baseline ({base_rate:.3f})")
    ax2.set_xlabel("Recall", fontsize=11)
    ax2.set_ylabel("Precision", fontsize=11)
    ax2.set_title("Precision-Recall (PR) Curve", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Experiment 2: Gold Evidence Proxy Discrimination (Dynamic Pipeline)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    roc_plot_path = os.path.join(plot_dir, "gold_discrimination_roc.png")
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    print(f"Saved ROC/PR curves plot to: {roc_plot_path}")

    # Plot 2: Score Distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.hist(df_results["fuzzy_evidence_reliability"], bins=30, alpha=0.6, color="#1f77b4", label="Fuzzy Reliability", density=True)
    ax1.hist(df_results["baseline_adjusted_score"], bins=30, alpha=0.6, color="#ff7f0e", label="Adjusted Baseline", density=True)
    ax1.set_xlabel("Score Value", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title("Primary Scores: Fuzzy vs. Baseline", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.hist(df_results["fuzzy_explanation_confidence"], bins=30, alpha=0.6, color="#2ca02c", label="Explanation Confidence", density=True)
    ax2.hist(df_results["baseline_raw_score"], bins=30, alpha=0.6, color="#d62728", label="Unadjusted Baseline Score", density=True)
    ax2.set_xlabel("Score Value", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title("Secondary Diagnostic Scores", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Experiment 2: Score Distributions & Centroid Compression", fontsize=14, fontweight="bold")
    plt.tight_layout()
    dist_plot_path = os.path.join(plot_dir, "score_distribution_comparison.png")
    plt.savefig(dist_plot_path, dpi=300)
    plt.close()
    print(f"Saved score distribution plot to: {dist_plot_path}")

    # Plot 3: Top-1 Hit Rate Comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    categories = ["All Claims (N=500)", "Gold-Present Claims (N=430)"]
    fuzzy_rates = [top1_dynamic['metric_a_all_claims']['fuzzy_rate']*100, top1_dynamic['metric_b_gold_present_claims']['fuzzy_rate']*100]
    base_rates = [top1_dynamic['metric_a_all_claims']['baseline_rate']*100, top1_dynamic['metric_b_gold_present_claims']['baseline_rate']*100]

    x = np.arange(len(categories))
    width = 0.35

    rects1 = ax.bar(x - width/2, fuzzy_rates, width, label='Fuzzy Reliability', color='#1f77b4', alpha=0.85)
    rects2 = ax.bar(x + width/2, base_rates, width, label='Adjusted Baseline', color='#ff7f0e', alpha=0.85)

    ax.set_ylabel('Top-1 Gold Hit Rate (%)', fontsize=11)
    ax.set_title('Top-1 Gold-Evidence Hit Rate Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(frameon=True)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    # Add bar values
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    top1_plot_path = os.path.join(plot_dir, "top1_hit_rate_comparison.png")
    plt.savefig(top1_plot_path, dpi=300)
    plt.close()
    print(f"Saved Top-1 hit rate plot to: {top1_plot_path}")

    # Plot 4: Attribution-Gap Diagnostics (Exploratory Subgroup)
    if n_rank_gap > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Diagnostic score comparison in gap subset
        gap_scores = [rank_gap_df["fuzzy_explanation_confidence"].values, rank_gap_df["baseline_raw_score"].values]
        bplot = ax1.boxplot(gap_scores, patch_artist=True, labels=["Explanation\nConfidence", "Unadjusted\nBaseline"])
        colors = ['#2ca02c', '#d62728']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax1.set_ylabel("Score Value", fontsize=11)
        ax1.set_title("Diagnostic Scores in Rank-Gap Subset (N=87)", fontsize=12, fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Within-claim rank distribution
        f_ranks = rank_gap_df["fuzzy_rank"].value_counts().sort_index()
        b_ranks = rank_gap_df["baseline_rank"].value_counts().sort_index()
        x_ranks = np.array([1, 2, 3])
        w = 0.35
        ax2.bar(x_ranks - w/2, [f_ranks.get(r, 0) for r in x_ranks], width=w, label="Fuzzy Reliability Rank", color="#1f77b4", alpha=0.8)
        ax2.bar(x_ranks + w/2, [b_ranks.get(r, 0) for r in x_ranks], width=w, label="Baseline Rank", color="#ff7f0e", alpha=0.8)
        ax2.set_xlabel("Assigned Within-Claim Rank (1 = Top priority)", fontsize=11)
        ax2.set_ylabel("Count of Evidence Instances", fontsize=11)
        ax2.set_title("Rank Shift on Attribution-Divergent Evidence", fontsize=12, fontweight="bold")
        ax2.set_xticks(x_ranks)
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

        plt.suptitle("Exploratory Attribution-Divergence Subgroup Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout()
        gap_plot_path = os.path.join(plot_dir, "attribution_gap_behavior.png")
        plt.savefig(gap_plot_path, dpi=300)
        plt.close()
        print(f"Saved attribution-gap plot to: {gap_plot_path}")

    # -----------------------------------------------------------------------
    # 10. Assemble and Save Summary JSON & Config JSON
    # -----------------------------------------------------------------------
    summary_data = {
        "experiment_name": "Experiment 2 — Comparative Evaluation of Fuzzy Evidence Reasoning vs. Linear Baseline",
        "phase": 11,
        "run_timestamp": datetime.datetime.now().isoformat(),
        "integrity_validation": validation_info,
        "sample_sizes": {
            "claims_evaluated": 500,
            "evidence_evaluations_total": 1500,
            "gold_instances": 443,
            "distractor_instances": 1057,
            "claims_with_gold_in_top3": 430,
            "claims_with_zero_gold_in_top3": 70
        },
        "co_primary_results_dynamic_pipeline": {
            "H1_AUROC": {
                "point_estimate_fuzzy": co_primary_adjusted["H1_AUROC"]["point_estimate_fuzzy"],
                "point_estimate_base": co_primary_adjusted["H1_AUROC"]["point_estimate_base"],
                "delta_hat": co_primary_adjusted["H1_AUROC"]["delta_hat"],
                "ci_95_marginal": co_primary_adjusted["H1_AUROC"]["ci_95_marginal"],
                "ci_975_bonferroni": co_primary_adjusted["H1_AUROC"]["ci_975_bonferroni"],
                "p_value_raw": co_primary_adjusted["H1_AUROC"]["p_value_raw"],
                "p_value_holm_adjusted": co_primary_adjusted["H1_AUROC"]["p_value_holm_adjusted"],
                "step_alpha": co_primary_adjusted["H1_AUROC"]["step_alpha"],
                "statistically_significant": co_primary_adjusted["H1_AUROC"]["statistically_significant_at_fwer_005"],
                "replicates_valid": co_primary_adjusted["H1_AUROC"]["replicates_valid"],
                "replicates_degenerate": co_primary_adjusted["H1_AUROC"]["replicates_degenerate"]
            },
            "H2_Average_Precision": {
                "point_estimate_fuzzy": co_primary_adjusted["H2_Average_Precision"]["point_estimate_fuzzy"],
                "point_estimate_base": co_primary_adjusted["H2_Average_Precision"]["point_estimate_base"],
                "delta_hat": co_primary_adjusted["H2_Average_Precision"]["delta_hat"],
                "ci_95_marginal": co_primary_adjusted["H2_Average_Precision"]["ci_95_marginal"],
                "ci_975_bonferroni": co_primary_adjusted["H2_Average_Precision"]["ci_975_bonferroni"],
                "p_value_raw": co_primary_adjusted["H2_Average_Precision"]["p_value_raw"],
                "p_value_holm_adjusted": co_primary_adjusted["H2_Average_Precision"]["p_value_holm_adjusted"],
                "step_alpha": co_primary_adjusted["H2_Average_Precision"]["step_alpha"],
                "statistically_significant": co_primary_adjusted["H2_Average_Precision"]["statistically_significant_at_fwer_005"],
                "replicates_valid": co_primary_adjusted["H2_Average_Precision"]["replicates_valid"],
                "replicates_degenerate": co_primary_adjusted["H2_Average_Precision"]["replicates_degenerate"]
            }
        },
        "secondary_ablation_static_defaults": {
            "AUROC": {
                "point_estimate_fuzzy": boot_auroc_static["point_estimate_fuzzy"],
                "point_estimate_base": boot_auroc_static["point_estimate_base"],
                "delta_hat": boot_auroc_static["delta_hat"],
                "ci_95_marginal": boot_auroc_static["ci_95_marginal"],
                "p_value_raw": boot_auroc_static["p_value_raw"]
            },
            "Average_Precision": {
                "point_estimate_fuzzy": boot_ap_static["point_estimate_fuzzy"],
                "point_estimate_base": boot_ap_static["point_estimate_base"],
                "delta_hat": boot_ap_static["delta_hat"],
                "ci_95_marginal": boot_ap_static["ci_95_marginal"],
                "p_value_raw": boot_ap_static["p_value_raw"]
            }
        },
        "top1_alignment_metrics": {
            "dynamic_pipeline": top1_dynamic,
            "static_ablation": top1_static
        },
        "score_distribution_diagnostics": dist_stats,
        "attribution_gap_subgroups": {
            "membership_standard_gap_count": len(std_gap_df),
            "membership_extreme_gap_count": len(ext_gap_df),
            "rank_divergence_gap_exploratory": rank_gap_summary
        }
    }

    summary_json_path = os.path.join(output_dir, "summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)
    print(f"Saved experiment summary to: {summary_json_path}")

    config_json_path = os.path.join(output_dir, "config.json")
    config_data = {
        "experiment": "Experiment 2 Formal Evaluation",
        "dataset": "FEVER dev split (first 500 claims)",
        "retrieval_mode": cfg.retrieval_mode,
        "top_k": cfg.top_k,
        "candidate_pool_size": len(retriever.corpus),
        "embedding_model": cfg.embedding_model,
        "verifier_model": cfg.verifier_model,
        "fuzzy_weights": {
            "shap_weight": cfg.fuzzy_shap_weight,
            "lime_weight": cfg.fuzzy_lime_weight
        },
        "baseline_weights": cfg.fuzzy_baseline_weights,
        "bootstrap": {
            "replicates": 1000,
            "seed": 42,
            "resampling_unit": "claim_level_cluster",
            "preserves_multiplicity": True,
            "null_centering": True
        },
        "co_primary_alpha": 0.05,
        "multiple_testing_correction": "Holm-Bonferroni"
    }
    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4)
    print(f"Saved configuration metadata to: {config_json_path}")

    print("\n" + "=" * 70)
    print("FORMAL EXPERIMENT 2 COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_formal_experiment_2()
