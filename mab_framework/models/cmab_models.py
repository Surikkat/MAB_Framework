import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class LinearNormalModel:
    """
    Linear Normal Model for Contextual Bandits.

    References
    ----------
    Suraveikin, E., Omirzak, D., Sultimov, R., & Maximov, Y. (2026). 
    "Efficient Contextual Bandit Learning via Reward-Space Sampling 
    and Online Optimization." AAAI.
    """
    def __init__(self, input_dim=None, n_arms=2, lr=0.01, fixed_std=0.1, dtype=np.float64, feature_dim=None, **kwargs):
        input_dim = input_dim or feature_dim or 10
        self.n_arms = n_arms
        self.lr = lr
        self.fixed_std = fixed_std
        self.weights = np.zeros((n_arms, input_dim), dtype=dtype)

    def predict(self, context):
        if context.ndim == 1:
            means = self.weights @ context
        else:
            if context.shape[0] == self.weights.shape[0]:
                means = np.einsum('ij,ij->i', self.weights, context)
            else:
                means = context @ self.weights[0]
        stds = np.full_like(means, self.fixed_std)
        return np.stack([means, stds], axis=1)

    def partial_fit(self, context, action, reward):
        act = action if action < self.weights.shape[0] else 0
        pred = self.weights[act] @ context
        error = pred - reward
        grad = error * context
        self.weights[act] -= self.lr * grad

    def fit(self, context, reward, action=0):
        self.partial_fit(context, action, reward)

    def sample(self, context):
        pred = self.predict(context)
        if context.ndim == 1 or pred.shape[0] == 1:
            mu, std = pred[0, 0], pred[0, 1]
            return float(np.random.normal(mu, std))
        return np.random.normal(pred[:, 0], pred[:, 1])


class GLMNormalModel(LinearNormalModel):
    """
    Generalized Linear Model (GLM) Normal Model alias for LinearNormalModel.
    """
    pass


class NeuralNormalModel:
    """
    Neural Normal Model for Contextual Bandits.

    References
    ----------
    Suraveikin, E., Omirzak, D., Sultimov, R., & Maximov, Y. (2026). 
    "Efficient Contextual Bandit Learning via Reward-Space Sampling 
    and Online Optimization." AAAI.
    """
    def __init__(self, input_dim, n_arms, hidden_dim=64, lr=1e-3, fixed_std=0.1):
        self.n_arms = n_arms
        self.fixed_std = fixed_std

        self.model = nn.Sequential(
            nn.Linear(input_dim + n_arms, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = self._nll_loss_fixed_std

    def _nll_loss_fixed_std(self, pred, target):
        std = self.fixed_std
        var = std ** 2
        loss = 0.5 * torch.log(torch.tensor(2 * np.pi * var)) + 0.5 * ((target - pred) ** 2) / var
        return loss.mean()

    def _prepare_input(self, context_vector, arm):
        one_hot_arm = np.zeros(self.n_arms)
        one_hot_arm[arm] = 1
        return np.concatenate([context_vector, one_hot_arm])

    def predict(self, context_matrix):
        inputs = [self._prepare_input(context_matrix[arm], arm) for arm in range(self.n_arms)]
        inputs = torch.tensor(np.array(inputs), dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(inputs).squeeze().numpy()
        return [(mu, self.fixed_std) for mu in outputs]

    def partial_fit(self, context_vector, arm, reward):
        x = self._prepare_input(context_vector, arm)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor([reward], dtype=torch.float32)

        pred = self.model(x)
        loss = self.loss_fn(pred.squeeze(), y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def fit(self, context_vector, reward, arm=0):
        self.partial_fit(context_vector, arm, reward)

    def sample(self, context):
        if context.ndim == 1:
            x = self._prepare_input(context, 0)
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                mu = float(self.model(x).squeeze().item())
            return float(np.random.normal(mu, self.fixed_std))
        else:
            preds = self.predict(context)
            return np.array([np.random.normal(mu, sigma) for mu, sigma in preds])
