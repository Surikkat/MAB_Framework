import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .base import BaseModel


class _NeuralEmbedding(nn.Module):
    def __init__(self, theta_dim: int, m: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(theta_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, m)
        )

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        return self.net(theta)


def _rbf_kernel_matrix(X1, X2, lengthscale, variance):
    x1 = X1 / lengthscale
    x2 = X2 / lengthscale
    x1_sq = (x1 ** 2).sum(dim=1).unsqueeze(1)
    x2_sq = (x2 ** 2).sum(dim=1).unsqueeze(0)
    sqdist = x1_sq + x2_sq - 2.0 * (x1 @ x2.t())
    return variance * torch.exp(-0.5 * sqdist)


class NNAGPModel(BaseModel):
    """NN-AGP model — returns (mu, sigma²) from predict(), matching original posterior()."""
    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        m: int = 5,
        hidden_dim: int = 64,
        init_lengthscale: float = 1.0,
        init_var: float = 1.0,
        init_noise: float = 0.05,
        lr: float = 1e-3,
        mll_steps: int = 60,
        jitter: float = 1e-6
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.theta_dim = theta_dim
        self.x_dim = x_dim
        self.m = int(m)
        self.jitter = float(jitter)
        self.mll_steps = int(mll_steps)

        self.encoder = _NeuralEmbedding(theta_dim, self.m, hidden_dim).to(self.device)
        self.B = nn.Parameter(torch.randn(self.m, self.m, device=self.device) * 0.1)
        self.log_lengthscale = nn.Parameter(torch.log(torch.tensor(init_lengthscale, device=self.device)))
        self.log_variance = nn.Parameter(torch.log(torch.tensor(init_var, device=self.device)))
        self.log_noise = nn.Parameter(torch.log(torch.tensor(init_noise, device=self.device)))

        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + [self.B, self.log_lengthscale, self.log_variance, self.log_noise],
            lr=lr
        )

        self.thetas_hist = []
        self.Xs_hist = []
        self.ys_hist = []

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
        if len(self.Xs_hist) == 0:
            return None
        thetas_stack = torch.stack(self.thetas_hist, dim=0).to(self.device)
        X = torch.stack(self.Xs_hist, dim=0).to(self.device)
        y = torch.stack(self.ys_hist, dim=0).view(-1, 1).to(self.device)
        G = self.encoder(thetas_stack)
        Sigma_p = self.B @ self.B.t()
        k_x = _rbf_kernel_matrix(X, X, self.lengthscale, self.variance)
        GS = G @ Sigma_p
        GSGT = GS @ G.t()
        K_tilde = k_x * GSGT
        return {"G": G, "X": X, "y": y, "Sigma_p": Sigma_p, "k_x": k_x, "GSGT": GSGT, "K_tilde": K_tilde}

    def _safe_cholesky(self, K_base, n):
        jitter = self.jitter
        for _ in range(6):
            try:
                K = K_base + (self.sigma_eps ** 2 + jitter) * torch.eye(n, device=self.device)
                K = 0.5 * (K + K.t())
                L = torch.linalg.cholesky(K)
                return L, K
            except RuntimeError:
                jitter *= 10.0
        K_sym = 0.5 * (K_base + K_base.t())
        eigvals, eigvecs = torch.linalg.eigh(K_sym)
        min_eig = max(jitter, float(torch.max(-eigvals.min(), torch.tensor(1e-12, device=eigvals.device))))
        rel_eps = 1e-8
        min_eig = max(min_eig, float(eigvals.max()) * rel_eps)
        eigvals_clipped = torch.clamp(eigvals, min=min_eig)
        K = (eigvecs * eigvals_clipped.unsqueeze(0)) @ eigvecs.t()
        K = K + (self.sigma_eps ** 2 + jitter) * torch.eye(n, device=self.device)
        K = 0.5 * (K + K.t())
        L = torch.linalg.cholesky(K)
        return L, K

    def _neg_log_marginal_likelihood(self):
        mats = self._assemble()
        if mats is None:
            return None
        n = mats["K_tilde"].shape[0]
        y = mats["y"]
        L, K = self._safe_cholesky(mats["K_tilde"], n)
        v = torch.linalg.solve(L, y)
        alpha = torch.linalg.solve(L.t(), v)
        logdet = 2.0 * torch.sum(torch.log(torch.diag(L) + 1e-12))
        return (0.5 * (y.t() @ alpha) + 0.5 * logdet + 0.5 * n * math.log(2.0 * math.pi)).squeeze()

    def _fit_mll(self, steps=None):
        if len(self.Xs_hist) == 0:
            return
        steps = steps or self.mll_steps
        for _ in range(steps):
            self.optimizer.zero_grad()
            loss = self._neg_log_marginal_likelihood()
            if loss is None:
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + [self.B, self.log_lengthscale, self.log_variance, self.log_noise], 5.0
            )
            self.optimizer.step()

    def fit(self, x: np.ndarray, y: float, delay_fit: bool = False):
        theta, arm_x = self._split_context(x)
        self.thetas_hist.append(torch.tensor(theta, dtype=torch.float32, device=self.device))
        self.Xs_hist.append(torch.tensor(arm_x, dtype=torch.float32, device=self.device))
        self.ys_hist.append(torch.tensor([y], dtype=torch.float32, device=self.device))
        if not delay_fit:
            self._fit_mll()

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (mu, sigma²) — variance, not std — matching original posterior()."""
        theta, arm_x = self._split_context(x)
        x_t = torch.tensor(arm_x, dtype=torch.float32, device=self.device).unsqueeze(0)
        theta_t = torch.tensor(theta, dtype=torch.float32, device=self.device).unsqueeze(0)
        g_t = self.encoder(theta_t).view(self.m, 1)
        Sigma_p = self.B @ self.B.t()
        k_x_x = self.variance

        if len(self.Xs_hist) == 0:
            prior_var = (g_t.t() @ Sigma_p @ g_t).squeeze() * k_x_x
            return np.array([0.0]), np.array([float(prior_var.detach().cpu().item())])

        with torch.no_grad():
            mats = self._assemble()
            G = mats["G"]
            X = mats["X"]
            y = mats["y"]
            n = X.shape[0]
            L, K = self._safe_cholesky(mats["K_tilde"], n)
            v = torch.linalg.solve(L, y)
            alpha = torch.linalg.solve(L.t(), v)
            k_x_vec = _rbf_kernel_matrix(x_t, X, self.lengthscale, self.variance).view(-1)
            GS = G @ Sigma_p
            gts = (GS @ g_t).view(-1)
            k_tilde = k_x_vec * gts
            mu = (k_tilde.view(1, -1) @ alpha).squeeze()
            rhs = k_tilde.view(-1, 1)
            v2 = torch.linalg.solve(L, rhs)
            prior_var = (g_t.t() @ Sigma_p @ g_t).squeeze() * k_x_x
            sigma2 = prior_var - (v2.t() @ v2).squeeze()
            sigma2 = torch.clamp(sigma2, min=1e-12)
            return np.array([float(mu.cpu().item())]), np.array([float(sigma2.cpu().item())])

    def sample(self, x: np.ndarray) -> np.ndarray:
        mu, sigma2 = self.predict(x)
        return mu + np.random.randn() * np.sqrt(sigma2)
