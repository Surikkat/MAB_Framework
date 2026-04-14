"""NeuralUCB — exact reproduction of NN_AGP/bandits/neural_ucb.py.

Full gradient-based NeuralUCB with Z-matrix and gamma_t computation.
This has been decoupled to interact solely with BaseAlgorithms.
"""
import numpy as np
from typing import Optional, List, Dict, Any
from .base import BaseAlgorithm

class NeuralUCBAlgorithm(BaseAlgorithm):
    """Exact port of NN_AGP/bandits/neural_ucb.py NeuralUCB class decoupled from PyTorch."""
    def __init__(self,
                 n_arms: int,
                 model,
                 T: int,
                 lambd: float = 1.0,
                 nu: float = 1.0,
                 delta: float = 0.01,
                 S: float = 1.0,
                 m: int = 64,
                 L: int = 2,
                 C1: float = 1.0,
                 C2: float = 1.0,
                 C3: float = 1.0):
        super().__init__(n_arms, model)
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

        if hasattr(self.model, "p"):
            self.p = self.model.p
        else:
            self.p = self.model.theta0_vec.size
            
        self.Z = (lambd * np.eye(self.p)).astype(np.float64)

        self._Z_inv_cache = None
        self._Z_inv_valid = False
        self._eps = 1e-12
        self.t_round = 0

    def compute_gamma(self, t: int) -> float:
        t = max(2, int(t))

        C1, C2, C3 = self.C1, self.C2, self.C3
        m, L, lambd, nu, delta, S = self.m, self.L, self.lambd, self.nu, self.delta, self.S
        
        # Pull eta and J from model if available
        eta = getattr(self.model, 'eta', 1e-3)
        J = getattr(self.model, 'J', 20)

        signZ, logdetZ = np.linalg.slogdet(self.Z + self._eps*np.eye(self.p))
        signLam, logdetLam = np.linalg.slogdet(lambd * np.eye(self.p))
        log_det_ratio = logdetZ - logdetLam
        if log_det_ratio < 0:
            log_det_ratio = max(log_det_ratio, -1e6)

        log_term1 = np.log(max(2.0, m * (L**4) * t))
        log_term2 = np.log(max(2.0, m * (L**(7/2)) * t))

        A = C1 * m**(-1.0/6.0) * np.sqrt(log_term1 * (t**(7.0/6.0)) * (lambd**(-7.0/6.0)))

        B = (
            nu * np.sqrt(max(0.0, log_det_ratio))
            + C2 * m**(-1.0/6.0) * np.sqrt(log_term1 * (t**(5.0/3.0)) * (lambd**(-1.0/6.0)))
            - 2.0 * np.log(delta)
            + np.sqrt(lambd) * S
        )

        first_block = np.sqrt(1.0 + A * B)

        decay_term = (1.0 - eta * m * lambd)
        if decay_term < 0:
            decay_term = 0.0

        sgd_term = (lambd + C3 * t * L) * (
            (decay_term**(J/2.0)) * np.sqrt(t / max(1e-12, lambd))
            + m**(-1.0/6.0) * np.sqrt(log_term2 * (t**(5.0/3.0)) * (lambd**(-5.0/3.0))) * (1.0 + np.sqrt(t / max(1e-12, lambd)))
        )

        gamma_t = float(first_block + sgd_term)
        return gamma_t

    def _ensure_Z_inv(self):
        if not self._Z_inv_valid:
            try:
                self._Z_inv_cache = np.linalg.inv(self.Z + self._eps*np.eye(self.p))
            except np.linalg.LinAlgError:
                self._Z_inv_cache = np.linalg.pinv(self.Z + self._eps*np.eye(self.p))
            self._Z_inv_valid = True
        return self._Z_inv_cache

    def select_arm(self, context: np.ndarray) -> int:
        if context.ndim == 1:
            context = context.reshape(1, -1)

        t = max(1, self.t_round + 1)
        gamma_t = self.compute_gamma(t)

        Z_inv = self._ensure_Z_inv()

        utilities = []
        K = context.shape[0]
        for i in range(K):
            x = context[i]
            
            mu, _ = self.model.predict(x)
            f_val = float(mu[0])

            g_vec = self.model.get_grad_features(x)
            q = float(g_vec.dot(Z_inv.dot(g_vec)))
            u = f_val + gamma_t * np.sqrt(max(0.0, q) / float(self.m))
            utilities.append(float(u))

        return int(np.argmax(utilities))

    def update(self, feedbacks: List[Dict[str, Any]]) -> None:
        for fb in feedbacks:
            action = fb["action"]
            reward = fb["reward"]
            context = fb["context"]
            if context.ndim > 1:
                x_chosen = context[action]
            else:
                x_chosen = context

            self.t_round += 1
            self.model.fit(x_chosen, reward)

            g_vec = self.model.get_grad_features(x_chosen)
            self.Z += np.outer(g_vec, g_vec) / float(self.m)
            self._Z_inv_valid = False
            
        if len(feedbacks) > 0:
            self.model.finalize_update()