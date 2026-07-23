import numpy as np

try:
    from obp.ope import (
        OffPolicyEvaluation as OBPOffPolicyEvaluation,
        InverseProbabilityWeighting as OBPIPS,
        SelfNormalizedInverseProbabilityWeighting as OBPSNIPS,
        DirectMethod as OBPDM,
        DoublyRobust as OBPDR,
    )
    OBP_AVAILABLE = True
except ImportError:
    OBP_AVAILABLE = False


class OPEEvaluator:
    """
    Нативный NumPy-эвалуатор для Off-Policy Evaluation (OPE).
    Поддерживает IPS, SNIPS, DM, DR с возможностью клиппинга весов и расчетом ESS.
    Также поддерживает fallback на библиотеку obp при необходимости.
    """
    def __init__(self, estimator_names, clipping_value=None, use_obp=False):
        self.estimator_names = [name.lower() for name in estimator_names]
        self.clipping_value = clipping_value
        self.use_obp = use_obp and OBP_AVAILABLE
        self.needs_reward_model = bool(set(self.estimator_names) & {"dm", "dr"})

        valid_names = {"ips", "snips", "dm", "dr"}
        for name in self.estimator_names:
            if name not in valid_names:
                raise ValueError(f"Unknown OPE estimator '{name}'. Available: {list(valid_names)}")

        if self.use_obp:
            self.obp_estimators = []
            map_obp = {
                "ips": OBPIPS(),
                "snips": OBPSNIPS(),
                "dm": OBPDM(),
                "dr": OBPDR(),
            }
            for name in self.estimator_names:
                self.obp_estimators.append(map_obp[name])

    def evaluate(self, bandit_feedback, action_dist, estimated_rewards_by_reg_model=None):
        if self.use_obp:
            ope = OBPOffPolicyEvaluation(
                bandit_feedback=bandit_feedback,
                ope_estimators=self.obp_estimators,
            )
            return ope.estimate_policy_values(
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            )

        # Нативная NumPy-реализация
        actions = np.asarray(bandit_feedback["action"]).reshape(-1)
        rewards = np.asarray(bandit_feedback["reward"]).reshape(-1)
        pscore = np.asarray(bandit_feedback["pscore"]).reshape(-1)
        n_rounds = len(actions)

        # Приводим action_dist к нужной форме
        if action_dist.ndim == 3:
            pi_e_matrix = action_dist[:, :, 0]
            pi_e_logged = pi_e_matrix[np.arange(n_rounds), actions]
        elif action_dist.ndim == 2:
            pi_e_matrix = action_dist
            pi_e_logged = pi_e_matrix[np.arange(n_rounds), actions]
        else:
            # 1D array — уже вероятности для залогированных действий
            pi_e_matrix = None
            pi_e_logged = np.asarray(action_dist).reshape(-1)

        # Веса важности
        pscore_clipped = np.clip(pscore, 1e-6, 1.0)
        weights = np.where(pscore_clipped > 0, pi_e_logged / pscore_clipped, 0.0)
        
        if self.clipping_value is not None and self.clipping_value > 0:
            weights_eval = np.clip(weights, 0.0, self.clipping_value)
        else:
            weights_eval = weights

        results = {}

        # 1. Direct Method (DM)
        dm_score = 0.0
        r_hat_matrix = None
        r_hat_logged = None
        if self.needs_reward_model and estimated_rewards_by_reg_model is not None:
            if estimated_rewards_by_reg_model.ndim == 3:
                r_hat_matrix = estimated_rewards_by_reg_model[:, :, 0]
                r_hat_logged = r_hat_matrix[np.arange(n_rounds), actions]
                if pi_e_matrix is not None:
                    dm_score = float(np.mean(np.sum(pi_e_matrix * r_hat_matrix, axis=1)))
                else:
                    dm_score = float(np.mean(pi_e_logged * r_hat_logged))
            elif estimated_rewards_by_reg_model.ndim == 2:
                r_hat_matrix = estimated_rewards_by_reg_model
                r_hat_logged = r_hat_matrix[np.arange(n_rounds), actions]
                if pi_e_matrix is not None:
                    dm_score = float(np.mean(np.sum(pi_e_matrix * r_hat_matrix, axis=1)))
                else:
                    dm_score = float(np.mean(pi_e_logged * r_hat_logged))
            else:
                r_hat_logged = np.asarray(estimated_rewards_by_reg_model).reshape(-1)
                dm_score = float(np.mean(pi_e_logged * r_hat_logged))

        for name in self.estimator_names:
            if name == "ips":
                results["ips"] = float(np.mean(weights_eval * rewards))
            elif name == "snips":
                w_sum = np.sum(weights_eval)
                results["snips"] = float(np.sum(weights_eval * rewards) / w_sum) if w_sum > 0 else 0.0
            elif name == "dm":
                results["dm"] = dm_score
            elif name == "dr":
                if r_hat_logged is not None:
                    dr_correction = float(np.mean(weights_eval * (rewards - r_hat_logged)))
                    results["dr"] = dm_score + dr_correction
                else:
                    results["dr"] = float(np.mean(weights_eval * rewards))  # fallback to IPS

        # Диагностические метрики (ESS и статистика весов)
        w_sum_raw = np.sum(weights)
        w_sq_sum_raw = np.sum(weights ** 2)
        results["ess"] = float(w_sum_raw ** 2 / w_sq_sum_raw) if w_sq_sum_raw > 0 else 0.0
        results["max_weight"] = float(np.max(weights))
        results["mean_weight"] = float(np.mean(weights))

        return results
