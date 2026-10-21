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
    def __init__(self, config: Config = None):
        self.config = config or Config()
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
        # Following the prompt's suggested overlapping triangles/trapezoids
        
        for var in [relevance, attribution, intervention, contradiction, source_reliability, stability,
                    evidence_relevance, evidence_reliability, support_degree, contradiction_output, explanation_confidence]:
            var['LOW'] = fuzz.trimf(var.universe, [0, 0, 0.4])
            var['MEDIUM'] = fuzz.trimf(var.universe, [0.2, 0.5, 0.8])
            var['HIGH'] = fuzz.trapmf(var.universe, [0.6, 0.8, 1.0, 1.0])

        # 4. Define Fuzzy Rules
        # RULE 1: Strong reliable evidence
        rule1 = ctrl.Rule(
            relevance['HIGH'] & attribution['HIGH'] & intervention['HIGH'] & 
            contradiction['LOW'] & source_reliability['HIGH'],
            evidence_reliability['HIGH']
        )
        
        # RULE 2: Strong relevance but weak intervention -> low confidence
        rule2 = ctrl.Rule(
            relevance['HIGH'] & attribution['HIGH'] & intervention['LOW'],
            explanation_confidence['LOW']
        )
        
        # RULE 3: Weak evidence
        rule3 = ctrl.Rule(
            relevance['LOW'] & attribution['LOW'] & intervention['LOW'],
            evidence_reliability['LOW']
        )
        
        # RULE 4: Strongly conflicting evidence
        rule4 = ctrl.Rule(
            relevance['HIGH'] & contradiction['HIGH'],
            contradiction_output['HIGH']
        )
        
        # RULE 5: Strong support
        rule5 = ctrl.Rule(
            relevance['HIGH'] & intervention['HIGH'] & contradiction['LOW'] & stability['HIGH'],
            support_degree['HIGH']
        )
        
        # RULE 6: Stable explanation
        rule6 = ctrl.Rule(
            attribution['HIGH'] & intervention['HIGH'] & stability['HIGH'],
            explanation_confidence['HIGH']
        )
        
        # RULE 7: Unstable explanation
        rule7 = ctrl.Rule(
            attribution['HIGH'] & stability['LOW'],
            explanation_confidence['LOW']
        )
        
        # RULE 8: Moderate evidence
        rule8 = ctrl.Rule(
            relevance['MEDIUM'] & attribution['MEDIUM'] & intervention['MEDIUM'],
            evidence_reliability['MEDIUM']
        )
        
        # Additional baseline passthrough rules to ensure outputs always have a defined value
        rule9 = ctrl.Rule(relevance['HIGH'], evidence_relevance['HIGH'])
        rule10 = ctrl.Rule(relevance['LOW'], evidence_relevance['LOW'])
        rule11 = ctrl.Rule(contradiction['LOW'], contradiction_output['LOW'])

        # 5. Build Control System
        self.ctrl_sys = ctrl.ControlSystem([
            rule1, rule2, rule3, rule4, rule5, 
            rule6, rule7, rule8, rule9, rule10, rule11
        ])
        self.sim = ctrl.ControlSystemSimulation(self.ctrl_sys)

    def evaluate(self, inputs: FuzzyInput) -> FuzzyOutput:
        """Evaluate the fuzzy system for a given set of inputs."""
        
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
        
        # Extract outputs with safe defaults in case rules don't activate
        def safe_extract(key: str, default: float = 0.5) -> float:
            try:
                return float(self.sim.output[key])
            except KeyError:
                return default

        return FuzzyOutput(
            evidence_relevance=safe_extract('evidence_relevance'),
            evidence_reliability=safe_extract('evidence_reliability'),
            support_degree=safe_extract('support_degree', 0.0),
            contradiction_degree=safe_extract('contradiction_output', 0.0),
            explanation_confidence=safe_extract('explanation_confidence')
        )
