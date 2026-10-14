import numpy as np
from typing import List, Dict, Any

from ..base import BaseAlgorithm


class NeuralUCBAlgorithm(BaseAlgorithm):
    """
    Neural Contextual Bandits with UCB-based Exploration.

    Per-arm variant: each arm has its own neural model and its own
    exploration matrix Z. The exploration bonus is computed using the
    gradient features of the neural network (see Zhou, Li, Gu, 2020).

    References
    ----------
    Zhou, D., Li, L., & Gu, Q. (2020). "Neural Contextual Bandits with
    UCB-based Exploration." ICML.
    """

    def __init__(
        self,
        n_arms: int,
        model,
        T: int = 10000,
        lambd: float = 1.0,
        nu: float = 1.0,
        delta: float = 0.01,
        S: float = 1.0,
        m: int = 64,
        L: int = 2,
        C1: float = 1.0,
        C2: float = 1.0,
        C3: float = 1.0,
        use_adaptive_gamma: bool = True,
        **kwargs,
    ):
        super().__init__(n_arms=n_arms, model=model)

        self.T = T
        self.lambd = lambd
        self.nu = nu
        self.delta = delta
        self.S = S
        self.m = m
        self.L = L
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.use_adaptive_gamma = use_adaptive_gamma

        # Reference model (for shape / attribute queries)
        m_ref = self.model[0]

        # Pull eta / J from the model
        self.eta = float(getattr(m_ref, "eta", 1e-3))
        self.J = int(getattr(m_ref, "J", 20))

        # Feature dimension
        if hasattr(m_ref, "p"):
            self.p = int(m_ref.p)
        else:
            self.p = int(m_ref.theta0_vec.size)

        # One Z per arm
        self.Z = [
            (lambd * np.eye(self.p)).astype(np.float64)
            for _ in range(n_arms)
        ]
        self._Z_inv_cache = [None] * n_arms
        self._Z_inv_valid = [False] * n_arms

        self._eps = 1e-12
        self.t_round = 0

    # ------------------------------------------------------------------ #
    # Exploration scale
    # ------------------------------------------------------------------ #
    def compute_gamma(self, t: int) -> float:
        t = max(2, int(t))

        C1, C2, C3 = self.C1, self.C2, self.C3
        m, L, lambd, nu, delta, S = (
            self.m, self.L, self.lambd,
            self.nu, self.delta, self.S,
        )
        eta, J = self.eta, self.J

        # Determinant of Z (use arm 0 as reference for the bound)
        signZ, logdetZ = np.linalg.slogdet(
            self.Z[0] + self._eps * np.eye(self.p)
        )
        logdetLam = self.p * np.log(lambd)
        log_det_ratio = logdetZ - logdetLam
        if log_det_ratio < 0:
            log_det_ratio = max(log_det_ratio, -1e6)

        log_term1 = np.log(max(2.0, m * (L ** 4) * t))
        log_term2 = np.log(max(2.0, m * (L ** (7 / 2)) * t))

        A = C1 * m ** (-1.0 / 6.0) * np.sqrt(
            log_term1 * (t ** (7.0 / 6.0)) * (lambd ** (-7.0 / 6.0))
        )

        B = (
            nu * np.sqrt(max(0.0, log_det_ratio))
            + C2 * m ** (-1.0 / 6.0)
              * np.sqrt(log_term1 * (t ** (5.0 / 3.0))
                        * (lambd ** (-1.0 / 6.0)))
            - 2.0 * np.log(delta)
            + np.sqrt(lambd) * S
        )

        first_block = np.sqrt(1.0 + A * B)

        decay_term = (1.0 - eta * m * lambd)
        if decay_term < 0:
            decay_term = 0.0

        sgd_term = (lambd + C3 * t * L) * (
            (decay_term ** (J / 2.0)) * np.sqrt(t / max(1e-12, lambd))
            + m ** (-1.0 / 6.0)
              * np.sqrt(log_term2 * (t ** (5.0 / 3.0))
                        * (lambd ** (-5.0 / 3.0)))
              * (1.0 + np.sqrt(t / max(1e-12, lambd)))
        )

        return float(first_block + sgd_term)

    # ------------------------------------------------------------------ #
    # Z inverse caching
    # ------------------------------------------------------------------ #
    def _ensure_Z_inv(self, arm: int) -> np.ndarray:
        if not self._Z_inv_valid[arm]:
            try:
                self._Z_inv_cache[arm] = np.linalg.inv(
                    self.Z[arm] + self._eps * np.eye(self.p)
                )
            except np.linalg.LinAlgError:
                self._Z_inv_cache[arm] = np.linalg.pinv(
                    self.Z[arm] + self._eps * np.eye(self.p)
                )
            self._Z_inv_valid[arm] = True
        return self._Z_inv_cache[arm]

    # ------------------------------------------------------------------ #
    # Action selection
    # ------------------------------------------------------------------ #
    def select_arm(self, context: np.ndarray) -> int:
        if context.ndim == 1:
            context = context.reshape(1, -1)

        t = max(1, self.t_round + 1)
        gamma_t = self.compute_gamma(t)

        K = context.shape[0]
        utilities = []
        for i in range(K):
            x = context[i]
            Z_inv = self._ensure_Z_inv(i)

            mu, _ = self.model[i].predict(x)
            f_val = float(np.asarray(mu).ravel()[0])

            g_vec = self.model[i].get_grad_features(x)
            g_vec = np.asarray(g_vec, dtype=np.float64).ravel()

            q = float(g_vec.dot(Z_inv.dot(g_vec)))
            u = f_val + gamma_t * np.sqrt(max(0.0, q) / float(self.m))
            utilities.append(float(u))

        return int(np.argmax(utilities))

    # ------------------------------------------------------------------ #
    # Update
    # ------------------------------------------------------------------ #
    def update(self, feedbacks: List[Dict[str, Any]]) -> None:
        for fb in feedbacks:
            action = int(fb["action"])
            reward = float(fb["reward"])
            context = fb["context"]

            if context.ndim > 1:
                x_chosen = context[action]
            else:
                x_chosen = context

            self.t_round += 1
            self.model[action].fit(x_chosen, reward)

            g_vec = self.model[action].get_grad_features(x_chosen)
            g_vec = np.asarray(g_vec, dtype=np.float64).ravel()
            g_vec_m = g_vec / np.sqrt(float(self.m))

            # Sherman–Morrison update of Z and its inverse
            self.Z[action] += np.outer(g_vec_m, g_vec_m)

            Z_inv = self._ensure_Z_inv(action)
            Z_inv_g = Z_inv.dot(g_vec_m)
            denom = 1.0 + float(g_vec_m.dot(Z_inv_g))
            self._Z_inv_cache[action] = (
                Z_inv - np.outer(Z_inv_g, Z_inv_g) / denom
            )
            self._Z_inv_valid[action] = True

        if feedbacks:
            for m in self.model:
                m.finalize_update()