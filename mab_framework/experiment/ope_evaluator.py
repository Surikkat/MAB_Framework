import numpy as np
from obp.ope import (
    OffPolicyEvaluation,
    InverseProbabilityWeighting,
    SelfNormalizedInverseProbabilityWeighting,
)


ESTIMATOR_MAP = {
    "ips": InverseProbabilityWeighting,
    "snips": SelfNormalizedInverseProbabilityWeighting,
}


class OPEEvaluator:
    def __init__(self, estimator_names):
        self.estimators = []
        for name in estimator_names:
            key = name.lower()
            if key not in ESTIMATOR_MAP:
                raise ValueError(
                    f"Unknown OPE estimator '{name}'. "
                    f"Available: {list(ESTIMATOR_MAP.keys())}"
                )
            self.estimators.append(ESTIMATOR_MAP[key]())

    def evaluate(self, bandit_feedback, action_dist):
        ope = OffPolicyEvaluation(
            bandit_feedback=bandit_feedback,
            ope_estimators=self.estimators,
        )
        return ope.estimate_policy_values(action_dist=action_dist)
