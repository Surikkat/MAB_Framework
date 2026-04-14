import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
from .base import BaseModel

torch.set_default_dtype(torch.float32)

class _MLP(nn.Module):
    def __init__(self, input_dim: int, m: int, L: int):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(L):
            layers.append(nn.Linear(in_dim, m))
            layers.append(nn.ReLU())
            in_dim = m
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

class NeuralUCBModel(BaseModel):
    """
    Decoupled NeuralUCB Model responsible for tracking histories and training _MLP via SGD.
    Matches the exact training mathematics from the original NeuralUCB algorithm.
    """
    def __init__(self, feature_dim: int, m: int = 64, L: int = 2, 
                 lambd: float = 1.0, eta: float = 1e-3, J: int = 20, device: str = "cpu"):
        self.device = device
        self.input_dim = feature_dim
        self.m = m
        self.L = L
        self.lambd = lambd
        self.eta = eta
        self.J = J
        
        self._model = _MLP(self.input_dim, m, L).to(self.device)
        self.theta0_vec = self._get_param_vector(self._model).copy()
        self.p = self.theta0_vec.size
        
        self.history_X = []
        self.history_y = []

    def _get_param_vector(self, model: nn.Module) -> np.ndarray:
        with torch.no_grad():
            vecs = []
            for p in model.parameters():
                vecs.append(p.view(-1).cpu().numpy())
            if vecs:
                return np.concatenate(vecs).astype(np.float64)
            else:
                return np.zeros(0, dtype=np.float64)

    def _set_param_vector(self, model: nn.Module, vec: np.ndarray):
        vec = vec.astype(np.float32)
        pointer = 0
        with torch.no_grad():
            for p in model.parameters():
                numel = p.numel()
                chunk = vec[pointer:pointer+numel]
                p.data.copy_(torch.from_numpy(chunk.reshape(p.shape)).to(p.device))
                pointer += numel
        assert pointer == vec.size

    def get_grad_features(self, x: np.ndarray) -> np.ndarray:
        self._model.zero_grad()
        x_t = torch.from_numpy(x.astype(np.float32)).to(self.device).unsqueeze(0)
        out = self._model(x_t)
        grads = torch.autograd.grad(out, [p for p in self._model.parameters()], retain_graph=False, create_graph=False)
        vecs = [g.contiguous().view(-1).cpu().numpy().astype(np.float64) for g in grads]
        return np.concatenate(vecs)

    def fit(self, x: np.ndarray, y: float) -> None:
        self.history_X.append(x.astype(np.float32))
        self.history_y.append(float(y))

    def finalize_update(self) -> None:
        if len(self.history_y) == 0:
            return

        optimizer = optim.SGD(self._model.parameters(), lr=self.eta, momentum=0.0)
        X = torch.from_numpy(np.stack(self.history_X).astype(np.float32)).to(self.device)
        y = torch.from_numpy(np.array(self.history_y, dtype=np.float32)).to(self.device)

        theta0 = torch.from_numpy(self.theta0_vec.astype(np.float32)).to(self.device)

        for _ in range(self.J):
            optimizer.zero_grad()
            preds = self._model(X)
            mse = 0.5 * torch.mean((preds - y)**2) * X.shape[0]
            cur_vec = torch.cat([p.view(-1) for p in self._model.parameters()])
            reg = 0.5 * float(self.m * self.lambd) * torch.sum((cur_vec - theta0)**2)
            loss = mse + reg
            loss.backward()
            optimizer.step()

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            xt = torch.from_numpy(x.astype(np.float32)).to(self.device).unsqueeze(0)
            f_val = float(self._model(xt).cpu().numpy().squeeze())
        return np.array([f_val]), np.array([0.0]) # Note: Variance computation is handled independently by UCB Z-matrix

    def sample(self, x: np.ndarray) -> np.ndarray:
        mu, _ = self.predict(x)
        return mu
