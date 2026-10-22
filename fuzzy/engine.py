"""
Fuzzy Evidence Reasoning Engine using scikit-fuzzy.
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from config.settings import Config
from fuzzy.schema import FuzzyInput, FuzzyOutput
from fuzzy.normalization import compute_attribution_strength


class FuzzyEvidenceEngine:
    def __init__(self, config: Config = None, version: str = "v2"):
        self.config = config or Config()
        self.version = str(version).lower()
        self._build_system()

    def _build_system(self):
        """Construct the fuzzy control system (variables, MFs, and rules)."""
        
        # 1. Define Antecedents (Inputs)
        universe = np.arange(0, 1.01, 0.01)
        
        relevance = ctrl.Antecedent(universe, 'relevance')
        attribution = ctrl.Antecedent(universe, 'attribution')
        intervention = ctrl.Antecedent(universe, 'intervention')
        contradiction = ctrl.Antecedent(universe, 'contradiction')
        source_reliability = ctrl.Antecedent(universe, 'source_reliability')
        stability = ctrl.Antecedent(universe, 'stability')
        
        # 2. Define Consequents (Outputs)
        evidence_relevance = ctrl.Consequent(universe, 'evidence_relevance')
        evidence_reliability = ctrl.Consequent(universe, 'evidence_reliability')
        support_degree = ctrl.Consequent(universe, 'support_degree')
        contradiction_output = ctrl.Consequent(universe, 'contradiction_output')
        explanation_confidence = ctrl.Consequent(universe, 'explanation_confidence')

        # 3. Define Membership Functions
        for var in [relevance, attribution, intervention, contradiction, source_reliability, stability,
                    evidence_relevance, evidence_reliability, support_degree, contradiction_output, explanation_confidence]:
            var['LOW'] = fuzz.trimf(var.universe, [0, 0, 0.4])
            var['MEDIUM'] = fuzz.trimf(var.universe, [0.2, 0.5, 0.8])
            var['HIGH'] = fuzz.trapmf(var.universe, [0.6, 0.8, 1.0, 1.0])

        # 4. Define Fuzzy Rules based on version
        if self.version == "v1":
            # Exact historical 11-rule baseline
            rule1 = ctrl.Rule(
                relevance['HIGH'] & attribution['HIGH'] & intervention['HIGH'] &
                contradiction['LOW'] & source_reliability['HIGH'],
                evidence_reliability['HIGH']
            )
            rule2 = ctrl.Rule(
                relevance['HIGH'] & attribution['HIGH'] & intervention['LOW'],
                explanation_confidence['LOW']
            )
            rule3 = ctrl.Rule(
                relevance['LOW'] & attribution['LOW'] & intervention['LOW'],
                evidence_reliability['LOW']
            )
            rule4 = ctrl.Rule(
                relevance['HIGH'] & contradiction['HIGH'],
                contradiction_output['HIGH']
            )
            rule5 = ctrl.Rule(
                relevance['HIGH'] & intervention['HIGH'] & contradiction['LOW'] & stability['HIGH'],
                support_degree['HIGH']
            )
            rule6 = ctrl.Rule(
                attribution['HIGH'] & intervention['HIGH'] & stability['HIGH'],
                explanation_confidence['HIGH']
            )
            rule7 = ctrl.Rule(
                attribution['HIGH'] & stability['LOW'],
                explanation_confidence['LOW']
            )
            rule8 = ctrl.Rule(
                relevance['MEDIUM'] & attribution['MEDIUM'] & intervention['MEDIUM'],
                evidence_reliability['MEDIUM']
            )
            rule9 = ctrl.Rule(relevance['HIGH'], evidence_relevance['HIGH'])
            rule10 = ctrl.Rule(relevance['LOW'], evidence_relevance['LOW'])
            rule11 = ctrl.Rule(contradiction['LOW'], contradiction_output['LOW'])

            rules = [
                rule1, rule2, rule3, rule4, rule5,
                rule6, rule7, rule8, rule9, rule10, rule11
            ]
        else:
            # V2 Remediated comprehensive rule base with 100% input-space coverage
            rules = [
                # Evidence Relevance (full partition)
                ctrl.Rule(relevance['HIGH'], evidence_relevance['HIGH'], label='R_rel_high'),
                ctrl.Rule(relevance['MEDIUM'], evidence_relevance['MEDIUM'], label='R_rel_med'),
                ctrl.Rule(relevance['LOW'], evidence_relevance['LOW'], label='R_rel_low'),

                # Contradiction Output (full partition, distractor-aware)
                ctrl.Rule(contradiction['HIGH'] & relevance['HIGH'], contradiction_output['HIGH'], label='R_cnt_high_rel'),
                ctrl.Rule(contradiction['HIGH'] & relevance['MEDIUM'], contradiction_output['MEDIUM'], label='R_cnt_med_rel'),
                ctrl.Rule(contradiction['HIGH'] & relevance['LOW'], contradiction_output['LOW'], label='R_cnt_low_distractor'),
                ctrl.Rule(contradiction['MEDIUM'], contradiction_output['MEDIUM'], label='R_cnt_med'),
                ctrl.Rule(contradiction['LOW'], contradiction_output['LOW'], label='R_cnt_low'),

                # Support Degree (full partition)
                ctrl.Rule(contradiction['HIGH'], support_degree['LOW'], label='R_sup_cnt_high'),
                ctrl.Rule(relevance['LOW'] | intervention['LOW'], support_degree['LOW'], label='R_sup_low_rel_or_int'),
                ctrl.Rule(contradiction['LOW'] & relevance['HIGH'] & intervention['HIGH'], support_degree['HIGH'], label='R_sup_strong'),
                ctrl.Rule(contradiction['LOW'] & (relevance['MEDIUM'] | intervention['MEDIUM']), support_degree['MEDIUM'], label='R_sup_med'),
                ctrl.Rule(contradiction['MEDIUM'] & relevance['HIGH'] & intervention['HIGH'], support_degree['MEDIUM'], label='R_sup_cnt_med_strong'),
                ctrl.Rule(contradiction['MEDIUM'] & (relevance['MEDIUM'] | intervention['MEDIUM']), support_degree['LOW'], label='R_sup_cnt_med_mod'),
                ctrl.Rule(contradiction['LOW'] & relevance['LOW'] & intervention['LOW'], support_degree['LOW'], label='R_sup_low_all'),

                # Evidence Reliability (full partition)
                ctrl.Rule(relevance['HIGH'] & (attribution['HIGH'] | intervention['HIGH']) & contradiction['LOW'] & ~source_reliability['LOW'], evidence_reliability['HIGH'], label='R_relb_high'),
                ctrl.Rule(relevance['HIGH'] & attribution['HIGH'] & intervention['HIGH'] & source_reliability['HIGH'], evidence_reliability['HIGH'], label='R_relb_strong'),
                ctrl.Rule(relevance['MEDIUM'] & contradiction['LOW'], evidence_reliability['MEDIUM'], label='R_relb_med_rel'),
                ctrl.Rule(attribution['MEDIUM'] | intervention['MEDIUM'], evidence_reliability['MEDIUM'], label='R_relb_med_attr_int'),
                ctrl.Rule(contradiction['MEDIUM'], evidence_reliability['MEDIUM'], label='R_relb_med_cnt'),
                ctrl.Rule(source_reliability['MEDIUM'], evidence_reliability['MEDIUM'], label='R_relb_med_src'),
                ctrl.Rule(relevance['LOW'], evidence_reliability['LOW'], label='R_relb_low_rel'),
                ctrl.Rule(attribution['LOW'] & intervention['LOW'], evidence_reliability['LOW'], label='R_relb_low_attr_int'),
                ctrl.Rule(contradiction['HIGH'], evidence_reliability['LOW'], label='R_relb_high_contra'),
                ctrl.Rule(source_reliability['LOW'], evidence_reliability['LOW'], label='R_relb_low_src'),

                # Explanation Confidence (full partition)
                ctrl.Rule(attribution['HIGH'] & intervention['HIGH'] & stability['HIGH'], explanation_confidence['HIGH'], label='R_conf_high'),
                ctrl.Rule(attribution['HIGH'] & intervention['HIGH'] & stability['MEDIUM'], explanation_confidence['HIGH'], label='R_conf_high_med_stab'),
                ctrl.Rule(attribution['MEDIUM'] | intervention['MEDIUM'], explanation_confidence['MEDIUM'], label='R_conf_med_attr_int'),
                ctrl.Rule(stability['MEDIUM'], explanation_confidence['MEDIUM'], label='R_conf_med_stab'),
                ctrl.Rule(attribution['LOW'] & intervention['HIGH'], explanation_confidence['MEDIUM'], label='R_conf_low_attr_high_int'),
                ctrl.Rule(attribution['HIGH'] & intervention['LOW'], explanation_confidence['LOW'], label='R_conf_low_int'),
                ctrl.Rule(stability['LOW'], explanation_confidence['LOW'], label='R_conf_low_stab'),
                ctrl.Rule(attribution['LOW'] & intervention['LOW'], explanation_confidence['LOW'], label='R_conf_low_attr_int')
            ]

        # 5. Build Control System
        self.ctrl_sys = ctrl.ControlSystem(rules)
        self.sim = ctrl.ControlSystemSimulation(self.ctrl_sys)

    def evaluate(self, inputs: FuzzyInput) -> FuzzyOutput:
        """Evaluate the fuzzy system for a given set of inputs with field-level status tracking."""
        
        # Compute combined attribution strength
        attr_strength = compute_attribution_strength(
            inputs.attribution_shap, 
            inputs.attribution_lime,
            self.config.fuzzy_shap_weight,
            self.config.fuzzy_lime_weight
        )
        
        # Load inputs into simulation
        self.sim.input['relevance'] = max(0.0, min(1.0, inputs.relevance))
        self.sim.input['attribution'] = max(0.0, min(1.0, attr_strength))
        self.sim.input['intervention'] = max(0.0, min(1.0, inputs.intervention_influence))
        self.sim.input['contradiction'] = max(0.0, min(1.0, inputs.contradiction_degree))
        self.sim.input['source_reliability'] = max(0.0, min(1.0, inputs.source_reliability))
        self.sim.input['stability'] = max(0.0, min(1.0, inputs.stability_score))
        
        # Compute!
        self.sim.compute()
        
        # Extract outputs while explicitly distinguishing defuzzified vs fallback values
        output_keys = {
            "evidence_relevance": 0.5,
            "evidence_reliability": 0.5,
            "support_degree": 0.0,
            "contradiction_output": 0.0,
            "explanation_confidence": 0.5
        }

        field_status = {}
        missing_fields = []
        extracted_vals = {}

        for k, default_val in output_keys.items():
            if k in self.sim.output:
                extracted_vals[k] = float(self.sim.output[k])
                field_status[k] = "COMPUTED"
            else:
                extracted_vals[k] = default_val
                field_status[k] = "FALLBACK_EMPTY_RULES"
                missing_fields.append(k)

        # Field-level status for contradiction_degree maps to contradiction_output
        field_status["contradiction_degree"] = field_status["contradiction_output"]

        # Track activated rules
        activated_rules = []
        for i, r in enumerate(self.ctrl_sys.rules):
            try:
                if r.aggregate_firing[self.sim] > 0.0:
                    activated_rules.append(i + 1)
            except Exception:
                pass

        activated_rule_count = len(activated_rules)
        fallback_used = len(missing_fields) > 0
        if len(missing_fields) == 0:
            inference_status = "SUCCESS"
        elif len(missing_fields) == len(output_keys):
            inference_status = "TOTAL_FALLBACK"
        else:
            inference_status = "PARTIAL_FALLBACK"

        return FuzzyOutput(
            evidence_relevance=extracted_vals['evidence_relevance'],
            evidence_reliability=extracted_vals['evidence_reliability'],
            support_degree=extracted_vals['support_degree'],
            contradiction_degree=extracted_vals['contradiction_output'],
            explanation_confidence=extracted_vals['explanation_confidence'],
            field_status=field_status,
            inference_status=inference_status,
            fallback_used=fallback_used,
            activated_rule_count=activated_rule_count,
            activated_rules=activated_rules,
            missing_fields=missing_fields
        )
