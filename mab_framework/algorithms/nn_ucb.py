"""NN_UCB — exact reproduction of NN_AGP/bandits/nn_ucb.py.

Diagonal approximation NN-UCB with gradient features.
"""
import numpy as np
from typing import List, Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from .base import BaseAlgorithm


class _SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_width=128, hidden_layers=2, output_dim=1):
        super().__init__()
        layers = []
        dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(dim, hidden_width))
            layers.append(nn.ReLU())
            dim = hidden_width
        layers.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class NNUCBAlgorithm(BaseAlgorithm):
    """Exact port of NN_AGP/bandits/nn_ucb.py NN_UCB class.

    Uses diagonal approximation for confidence (A_diag instead of full matrix).
    """
    def __init__(self,
                 n_arms: int,
                 context_dim: int,
                 hidden_width: int = 256,
                 hidden_layers: int = 2,
                 lambda_: float = 1.0,
                 beta: float = 1.0,
                 J: int = 10,
                 lr: float = 1e-3,
                 device=None,
                 seed: int = 0,
                 model=None):
        super().__init__(n_arms, model)
        self.context_dim = context_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.net = _SimpleMLP(input_dim=context_dim,
                              hidden_width=hidden_width,
                              hidden_layers=hidden_layers).to(self.device)

        self.param_shapes = [p.shape for p in self.net.parameters()]
        self.num_params = sum(p.numel() for p in self.net.parameters())

        self.lambda_ = float(lambda_)
        self.A_diag = (self.lambda_) * np.ones(self.num_params, dtype=float)

        self.buffer_X = []
        self.buffer_y = []

        self.J = int(J)
        self.lr = float(lr)
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr)

        self.beta = float(beta)

        self.eps = 1e-8

    def _params_to_vector(self):
        return torch.cat([p.detach().reshape(-1) for p in self.net.parameters()], dim=0)

    def _grad_feature(self, x):
        self.net.zero_grad()
        if not isinstance(x, torch.Tensor):
            x_t = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            x_t = x.to(self.device).unsqueeze(0)

        out = self.net(x_t)
        out.backward()
        grads = []
        for p in self.net.parameters():
            g = p.grad
            if g is None:
                grads.append(torch.zeros(p.numel(), device=self.device))
            else:
                grads.append(g.detach().reshape(-1))
        phi = torch.cat(grads, dim=0).cpu().numpy()
        return phi

    def select_arm(self, context: np.ndarray) -> int:
        if isinstance(context, np.ndarray):
            ctxs = context
        else:
            ctxs = np.array(context)

        if ctxs.ndim == 1:
            ctxs = ctxs.reshape(1, -1)

        k = ctxs.shape[0]
        ucbs = []
        self.net.eval()

        for i in range(k):
            x_np = ctxs[i]
            with torch.no_grad():
                x_t = torch.tensor(x_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                mu = self.net(x_t).item()
            phi = self._grad_feature(x_np)
            denom = self.A_diag
            v = float(np.sqrt(np.sum((phi ** 2) / (denom + self.eps))))
            ucb = mu + self.beta * v
            ucbs.append(ucb)
        return int(np.argmax(ucbs))

    def update(self, feedbacks: List[Dict[str, Any]]) -> None:
        new_ctxs = []
        for fb in feedbacks:
            action = fb["action"]
            reward = fb["reward"]
            context = fb["context"]
            if context.ndim > 1:
                ctx = context[action]
            else:
                ctx = context

            if isinstance(ctx, torch.Tensor):
                ctx = ctx.cpu().numpy()
            else:
                ctx = np.array(ctx, dtype=np.float32)

            self.buffer_X.append(ctx)
            self.buffer_y.append(float(reward))
            new_ctxs.append(ctx)

        if len(self.buffer_X) == 0 or len(feedbacks) == 0:
            return

        self.net.train()
        X = torch.tensor(np.stack([x if isinstance(x, np.ndarray) else x.numpy() for x in self.buffer_X]),
                         dtype=torch.float32, device=self.device)
        y = torch.tensor(self.buffer_y, dtype=torch.float32, device=self.device)
        loss_fn = nn.MSELoss()

        for _ in range(self.J):
            self.optimizer.zero_grad()
            preds = self.net(X)
            loss = loss_fn(preds, y)
            loss.backward()
            self.optimizer.step()

        for ctx in new_ctxs:
            phi = self._grad_feature(ctx)
            self.A_diag += (phi ** 2)

