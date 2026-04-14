import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .base import BaseModel

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.net(x)
        val = self.out(features)
        return val, features

class NeuralLinearModel(BaseModel):
    def __init__(self, feature_dim: int, hidden_dim: int = 32, lr: float = 0.01, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = SimpleMLP(feature_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.hidden_dim = hidden_dim
        self.A = np.eye(self.hidden_dim)

    def fit(self, x: np.ndarray, y: float):
        x_t = torch.FloatTensor(x).unsqueeze(0).to(self.device)
        y_t = torch.FloatTensor([y]).unsqueeze(0).to(self.device)
        self.model.train()
        self.optimizer.zero_grad()
        pred, features = self.model(x_t)
        loss = self.criterion(pred, y_t)
        loss.backward()
        self.optimizer.step()
        z = features.detach().cpu().numpy().reshape(-1, 1)
        self.A += z @ z.T

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        with torch.no_grad():
            x_t = torch.FloatTensor(x).unsqueeze(0).to(self.device)
            pred, features = self.model(x_t)
        expected_reward = pred.item()
        z = features.cpu().numpy().reshape(-1, 1)
        try:
            L = np.linalg.cholesky(self.A)
            v = np.linalg.solve(L, z)
            uncertainty = np.sqrt(v.T @ v).item()
        except np.linalg.LinAlgError:
            uncertainty = np.sqrt(z.T @ np.linalg.solve(self.A, z)).item()
            
        return np.array([expected_reward]), np.array([uncertainty])

    def sample(self, x: np.ndarray) -> np.ndarray:
        mu, sigma = self.predict(x)
        return mu + np.random.randn() * sigma

