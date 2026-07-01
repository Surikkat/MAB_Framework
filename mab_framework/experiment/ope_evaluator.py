import numpy as np
from obp.ope import (
    OffPolicyEvaluation,
    InverseProbabilityWeighting,
    SelfNormalizedInverseProbabilityWeighting,
    DirectMethod,
    DoublyRobust,
)

ESTIMATOR_MAP = {
    "ips": InverseProbabilityWeighting,
    "snips": SelfNormalizedInverseProbabilityWeighting,
    "dm": DirectMethod,
    "dr": DoublyRobust,
}

REWARD_MODEL_ESTIMATORS = {"dm", "dr"}


class OPEEvaluator:
    def __init__(self, estimator_names):
        self.estimator_names = [name.lower() for name in estimator_names]
        self.estimators = []
        for key in self.estimator_names:
            if key not in ESTIMATOR_MAP:
                raise ValueError(
                    f"Unknown OPE estimator '{key}'. "
                    f"Available: {list(ESTIMATOR_MAP.keys())}"
                )
            self.estimators.append(ESTIMATOR_MAP[key]())
        self.needs_reward_model = bool(set(self.estimator_names) & REWARD_MODEL_ESTIMATORS)

    def evaluate(self, bandit_feedback, action_dist, estimated_rewards_by_reg_model=None):
        ope = OffPolicyEvaluation(
            bandit_feedback=bandit_feedback,
            ope_estimators=self.estimators,
        )
        return ope.estimate_policy_values(
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
