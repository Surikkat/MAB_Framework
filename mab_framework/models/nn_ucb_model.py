import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .base import BaseModel

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

class NNUCBModel(BaseModel):
    """
    Decoupled NN-UCB Model responsible for tracking histories and training _SimpleMLP.
    Matches the exact training mathematics from the original NN_UCB algorithm.
    """
    def __init__(self, feature_dim: int, hidden_width: int = 256,
                 hidden_layers: int = 2, J: int = 10, lr: float = 1e-3, 
                 buffer_size: int = None, device: str = "cpu"):
        self.device = device
        self.context_dim = feature_dim
        
        self.net = _SimpleMLP(input_dim=self.context_dim,
                              hidden_width=hidden_width,
                              hidden_layers=hidden_layers).to(self.device)

        self.param_shapes = [p.shape for p in self.net.parameters()]
        self.num_params = sum(p.numel() for p in self.net.parameters())

        self.buffer_X = []
        self.buffer_y = []

        self.J = int(J)
        self.lr = float(lr)
        self.buffer_size = buffer_size
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr)

    def fit(self, x: np.ndarray, y: float) -> None:
        self.buffer_X.append(x)
        self.buffer_y.append(float(y))
        
        if self.buffer_size is not None and len(self.buffer_X) > self.buffer_size:
            self.buffer_X.pop(0)
            self.buffer_y.pop(0)

    def finalize_update(self) -> None:
        if len(self.buffer_X) == 0:
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

    def get_grad_features(self, x: np.ndarray) -> np.ndarray:
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

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.net.eval()
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
            mu = self.net(x_t).item()
        return np.array([mu]), np.array([0.0]) # Note: Variance computation is handled independently by UCB diag

    def sample(self, x: np.ndarray) -> np.ndarray:
        mu, _ = self.predict(x)
        return mu
