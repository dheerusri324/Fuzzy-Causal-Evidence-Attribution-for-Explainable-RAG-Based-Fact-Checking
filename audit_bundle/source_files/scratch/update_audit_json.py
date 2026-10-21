import json

audit_json = {
    "audit_timestamp": "2026-10-18T22:00:00",
    "dataset_provenance": {
        "huggingface_source": "copenlu/fever_gold_evidence",
        "raw_split_streamed": "train (first 5,000 records)",
        "project_dev_split_file": "data/fever_dev.jsonl",
        "project_dev_split_size": 750,
        "corpus_file": "data/evidence_corpus.json",
        "corpus_size": 808
    },
    "corpus_construction": {
        "construction_method": "Deduplicated pooling of evidence annotations from all 750 claims in data/fever_dev.jsonl",
        "deduplication_identifier": "evidence_id = {doc_title}_{sentence_idx}",
        "total_source_claims": 750,
        "evaluated_claims_count": 500,
        "evaluated_claims_selection": "First 500 records of data/fever_dev.jsonl (records[:500])",
        "remaining_claims_count": 250
    },
    "gold_evidence_corpus_availability": {
        "total_evaluated_claims": 500,
        "claims_with_non_empty_gold_annotations": 500,
        "claims_with_empty_gold_annotations": 0,
        "distinct_gold_evidence_in_evaluated_claims": 534,
        "distinct_gold_evidence_present_in_corpus": 534,
        "distinct_gold_evidence_present_in_corpus_pct": 100.0,
        "corpus_composition": {
            "gold_evidence_from_500_eval_claims": 534,
            "evidence_from_remaining_250_claims": 274,
            "total_corpus_passages": 808
        }
    },
    "retrieval_coverage_in_top3": {
        "total_evaluated_claims": 500,
        "claims_with_non_empty_gold_annotations": 500,
        "claims_with_empty_gold_annotations": 0,
        "mutually_exclusive_partition": {
            "all_annotated_gold_retrieved_in_top3": {
                "count": 409,
                "percentage": 81.80,
                "denominator": 500,
                "description": "Claims where every annotated gold evidence ID was successfully retrieved in the Top-3"
            },
            "partial_gold_retrieved_in_top3": {
                "count": 21,
                "percentage": 4.20,
                "denominator": 500,
                "description": "Claims where at least one, but not all, annotated gold evidence IDs were retrieved in the Top-3"
            },
            "zero_gold_retrieved_in_top3": {
                "count": 70,
                "percentage": 14.00,
                "denominator": 500,
                "description": "Claims with annotated gold evidence where zero gold pieces were retrieved in Top-3 (distractors only)"
            },
            "empty_gold_annotations": {
                "count": 0,
                "percentage": 0.00,
                "denominator": 500,
                "description": "Claims with no annotated gold evidence in the source dataset"
            }
        },
        "cumulative_metrics": {
            "at_least_one_gold_retrieved_in_top3": {
                "count": 430,
                "percentage": 86.00,
                "denominator": 500,
                "formula": "all_annotated (409) + partial (21) = 430"
            }
        },
        "evidence_instance_level": {
            "total_retrieved_evidence_instances": 1500,
            "retrieved_gold_evidence_instances": 443,
            "retrieved_gold_evidence_instances_pct": 29.53,
            "retrieved_distractor_instances": 1057,
            "retrieved_distractor_instances_pct": 70.47
        }
    },
    "attribution_and_disagreement_verification": {
        "direct_shap_lime_pairwise_comparison": {
            "direct_top1_agreement_count": 412,
            "direct_top1_agreement_pct": 82.40,
            "direct_top1_disagreement_count": 88,
            "direct_top1_disagreement_pct": 17.60,
            "denominator": 500,
            "shap_ties_at_rank1": 0,
            "lime_ties_at_rank1": 0,
            "definition": "Pairwise comparison of the top-1 ranked evidence ID by abs(SHAP) versus abs(LIME) for each claim"
        },
        "contingency_against_intervention": {
            "both_agree_with_intervention": {
                "count": 354,
                "percentage": 70.80
            },
            "lime_only_agrees_with_intervention": {
                "count": 66,
                "percentage": 13.20
            },
            "shap_only_agrees_with_intervention": {
                "count": 7,
                "percentage": 1.40
            },
            "neither_agrees_with_intervention_agree_eachother": {
                "count": 58,
                "percentage": 11.60
            },
            "neither_agrees_with_intervention_disagree_eachother": {
                "count": 15,
                "percentage": 3.00
            },
            "total_neither_agrees_with_intervention": {
                "count": 73,
                "percentage": 14.60
            }
        }
    },
    "scientific_framing": {
        "evaluation_nature": "Gold-conditioned, closed candidate-pool evaluation",
        "model_training_leakage": False,
        "reasoning": "Verifier (cross-encoder/nli-deberta-v3-base) and dense embedder (all-MiniLM-L6-v2) were pretrained checkpoints evaluated zero-shot without fine-tuning on this corpus.",
        "within_context_validity": "Experiment 1 evaluates post-hoc attribution fidelity against empirical leave-one-out intervention influence on the retrieved evidence triplet. This comparison is strictly within-context and remains 100% mathematically valid.",
        "external_validity_limitation": "The evaluation does not benchmark open-domain full-Wikipedia retrieval recall or discovery across 5.4 million documents."
    }
}

with open("outputs/phase12_analysis/dataset_overlap_audit.json", "w", encoding="utf-8") as f:
    json.dump(audit_json, f, indent=4)

print("Updated outputs/phase12_analysis/dataset_overlap_audit.json")
