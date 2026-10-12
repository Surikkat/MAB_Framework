"""
Adaptive m via SVD pruning. Single shared model for all arms.
"""
import math
import numpy as np
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from ..base import BaseAlgorithm
from ...models.nn_agp_model import _NeuralEmbedding, _rbf_kernel_matrix


class NNAGPUCBAdaptiveAlgorithm(BaseAlgorithm):
    """
    NN-AGP Adaptive UCB Algorithm with SVD pruning.
    
    References
    ----------
    Suraveikin, E. (2026). "Uncertainty Quantification and Interpretability 
    for Contextual Bandits through Structured Latent Representations."
    """
    def __init__(
        self,
        n_arms: int,
        theta_dim: int = 0,
        x_dim: int = 5,
        m: int = 5,
        hidden_dim: int = 64,
        init_lengthscale: float = 1.0,
        init_var: float = 1.0,
        init_noise: float = 0.05,
        beta: float = 2.0,
        lr: float = 1e-3,
        mll_steps: int = 60,
        jitter: float = 1e-6,
        T_spec: int = 5,
        eps_prune: float = 1e-2,
        window_size: int = None,
        model=None
    ):
        super().__init__(n_arms, model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.theta_dim = theta_dim
        self.x_dim = x_dim
        self.m_max = int(m)
        self.m = self.m_max
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.jitter = jitter
        self.mll_steps = int(mll_steps)
        self.T_spec = T_spec
        self.eps_prune = eps_prune
        self.window_size = window_size
        self.step_count = 0
        self.lr = lr

        self.encoder = _NeuralEmbedding(theta_dim, self.m_max, hidden_dim).to(self.device)
        self.B = nn.Parameter(torch.randn(self.m_max, self.m_max, device=self.device) * 0.1)

        self.log_lengthscale = nn.Parameter(torch.log(torch.tensor(init_lengthscale, device=self.device)))
        self.log_variance = nn.Parameter(torch.log(torch.tensor(init_var, device=self.device)))
        self.log_noise = nn.Parameter(torch.log(torch.tensor(init_noise, device=self.device)))

        self._build_optimizer(lr)

        self.thetas = []
        self.Xs = []
        self.ys = []
        self.m_history = []
        self.V_proj = None

    def _build_optimizer(self, lr: float):
        params = list(self.encoder.parameters()) + [self.B, self.log_lengthscale, self.log_variance, self.log_noise]
        self.optimizer = optim.Adam(params, lr=lr)

    @property
    def lengthscale(self):
        return torch.exp(self.log_lengthscale)

    @property
    def variance(self):
        return torch.exp(self.log_variance)

    @property
    def sigma_eps(self):
        return torch.exp(self.log_noise)

    def _split_context(self, x: np.ndarray):
        theta = x[:self.theta_dim]
        arm_x = x[self.theta_dim:]
        return theta, arm_x

    def _assemble(self):
        if len(self.Xs) == 0:
            return None
        thetas_stack = torch.stack(self.thetas, dim=0).to(self.device)
        X = torch.stack(self.Xs, dim=0).to(self.device)
        y = torch.stack(self.ys, dim=0).view(-1, 1).to(self.device)
        G = self.encoder(thetas_stack)
        Sigma_p = self.B @ self.B.t()
        k_x = _rbf_kernel_matrix(X, X, self.lengthscale, self.variance)
        GSGT = (G @ Sigma_p) @ G.t()
        K_tilde = k_x * GSGT
        return {"G": G, "X": X, "y": y, "Sigma_p": Sigma_p, "K_tilde": K_tilde}

    def _neg_log_marginal_likelihood(self):
        mats = self._assemble()
        if mats is None:
            return None
        n = mats["K_tilde"].shape[0]
        K = mats["K_tilde"] + (self.sigma_eps ** 2 + self.jitter) * torch.eye(n, device=self.device)
        y = mats["y"]
        try:
            L = torch.linalg.cholesky(K)
        except RuntimeError:
            K = K + 1e-6 * torch.eye(n, device=self.device)
            L = torch.linalg.cholesky(K)
        v = torch.linalg.solve(L, y)
        alpha = torch.linalg.solve(L.t(), v)
        logdet = 2.0 * torch.sum(torch.log(torch.diag(L) + 1e-12))
        mll = 0.5 * (y.t() @ alpha) + 0.5 * logdet + 0.5 * n * math.log(2.0 * math.pi)
        return mll.squeeze()

    def _fit_mll(self, steps: int = 10):
        if len(self.Xs) == 0:
            return
        for _ in range(steps):
            self.optimizer.zero_grad()
            loss = self._neg_log_marginal_likelihood()
            if loss is None:
                break
            if not loss.requires_grad:
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + [self.B, self.log_lengthscale, self.log_variance, self.log_noise],
                5.0
            )
            self.optimizer.step()

    def posterior(self, x: np.ndarray, theta: np.ndarray):
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
            theta_t = torch.tensor(theta, dtype=torch.float32, device=self.device).unsqueeze(0)
            g_t = self.encoder(theta_t).view(self.m_max, 1)
            Sigma_p = self.B @ self.B.t()
            k_x_x = self.variance

            if self.V_proj is not None:
                V = self.V_proj
                g_proj = V @ g_t
                Sigma_proj = V @ Sigma_p @ V.t()
            else:
                g_proj = g_t
                Sigma_proj = Sigma_p

            if len(self.Xs) == 0:
                prior_var = (g_proj.t() @ Sigma_proj @ g_proj).squeeze() * k_x_x
                return 0.0, float(prior_var.cpu().item())

            mats = self._assemble()
            G = mats["G"]
            X = mats["X"]
            y = mats["y"]
            n = X.shape[0]

            if self.V_proj is not None:
                G_proj = G @ self.V_proj.t()
            else:
                G_proj = G

            GS = G_proj @ Sigma_proj
            GSGT = GS @ G_proj.t()
            k_x = _rbf_kernel_matrix(X, X, self.lengthscale, self.variance)
            K = k_x * GSGT + (self.sigma_eps ** 2 + self.jitter) * torch.eye(n, device=self.device)

            try:
                L = torch.linalg.cholesky(K)
            except RuntimeError:
                K = K + 1e-6 * torch.eye(n, device=self.device)
                L = torch.linalg.cholesky(K)

            v = torch.linalg.solve(L, y)
            alpha = torch.linalg.solve(L.t(), v)

            k_x_vec = _rbf_kernel_matrix(x_t, X, self.lengthscale, self.variance).view(-1)
            gts = (GS @ g_proj).view(-1)
            k_tilde = k_x_vec * gts
            mu = (k_tilde.view(1, -1) @ alpha).squeeze()

            rhs = k_tilde.view(-1, 1)
            v2 = torch.linalg.solve(L, rhs)
            prior_var = (g_proj.t() @ Sigma_proj @ g_proj).squeeze() * k_x_x
            sigma2 = prior_var - (v2.t() @ v2).squeeze()
            sigma2 = torch.clamp(sigma2, min=1e-12)
            return float(mu.cpu().item()), float(sigma2.cpu().item())

    def prior_variance(self, x: np.ndarray, theta: np.ndarray) -> float:
        with torch.no_grad():
            theta_t = torch.tensor(theta, dtype=torch.float32, device=self.device).unsqueeze(0)
            g_t = self.encoder(theta_t).view(self.m_max, 1)
            Sigma_p = self.B @ self.B.t()
            k_x_x = self.variance

            if self.V_proj is not None:
                V = self.V_proj
                g_proj = V @ g_t
                Sigma_proj = V @ Sigma_p @ V.t()
            else:
                g_proj = g_t
                Sigma_proj = Sigma_p

            prior_var = (g_proj.t() @ Sigma_proj @ g_proj).squeeze() * k_x_x
            return float(prior_var.cpu().item())

    def select_arm(self, context: np.ndarray) -> int:
        if context.ndim == 1:
            context = context.reshape(1, -1)
        mus, sig2s = [], []
        for a in range(context.shape[0]):
            theta, arm_x = self._split_context(context[a])
            mu, s2 = self.posterior(arm_x, theta)
            mus.append(mu)
            sig2s.append(s2)
        mus = np.array(mus)
        sig2s = np.array(sig2s)
        ucbs = mus + math.sqrt(self.beta) * np.sqrt(sig2s)
        return int(np.argmax(ucbs))

    def update(self, feedbacks: List[Dict[str, Any]]) -> None:
        for fb in feedbacks:
            action = fb["action"]
            reward = fb["reward"]
            context = fb["context"]
            if context.ndim > 1:
                x_full = context[action]
            else:
                x_full = context
            theta, arm_x = self._split_context(x_full)

            theta_t = torch.tensor(theta, dtype=torch.float32, device=self.device).view(-1)
            x_t = torch.tensor(arm_x, dtype=torch.float32, device=self.device).view(-1)
            y_t = torch.tensor([reward], dtype=torch.float32, device=self.device).view(-1)
            self.thetas.append(theta_t)
            self.Xs.append(x_t)
            self.ys.append(y_t)
            self.step_count += 1
            
        if len(feedbacks) == 0:
            return

        self._fit_mll(steps=10)

        # Spectral update
        if (self.step_count % self.T_spec == 0) and (len(self.thetas) >= 2):
            if self.window_size is not None and len(self.thetas) > self.window_size:
                spec_thetas = self.thetas[-self.window_size:]
            else:
                spec_thetas = self.thetas

            G_stack = torch.stack(spec_thetas, dim=0).to(self.device)
            with torch.no_grad():
                G_emb = self.encoder(G_stack)
                U, S, Vh = torch.linalg.svd(G_emb, full_matrices=False)
                if S.numel() > 0 and S.max() > 0:
                    m_star = int(torch.sum(S > self.eps_prune * S.max()).item())
                    m_star = max(1, min(m_star, self.m_max))
                    self.m = m_star
                    if m_star < self.m_max:
                        self.V_proj = Vh[:m_star, :].to(self.device)
                    else:
                        self.V_proj = None

        self.m_history.append(self.m)

