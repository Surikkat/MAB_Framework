import numpy as np
from typing import List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import cvxpy as cp
from scipy.stats import invgamma

from ..base import BaseAlgorithm

import warnings
warnings.filterwarnings("ignore")


class _DNN(nn.Module):
    def __init__(self, d, g, n_arms):
        super(_DNN, self).__init__()
        self.fc1 = nn.Linear(d, 2*d)
        self.fc2 = nn.Linear(2*d, g)
        self.fc3 = nn.Linear(g, n_arms)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)

    def get_phi(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class NeuralBanditWithLimitedMemory_5(BaseAlgorithm):
    def __init__(self,
                 buffer_size,
                 min_buffer_size,
                 n_arms,
                 g,
                 L,
                 input_dim,
                 epsilon=1e-6,
                 lambda_prior=0.1,
                 a_0=6.0, b_0=6.0,
                 P=400,
                 batch_size=1, lr=1e-3, model=None):
        """
        buffer_size - size of the buffer that stores the results of the last n iterations
        n_arms - number of actions
        g - size of last layer activation
        L - learning frequency
        epsilon - accuracy
        lambda_prior - initial prior
        a_0, b_0 - initial invgamma params
        P - learning iterations
        dnn - DNN
        batch_size - for DNN
        lr - learning rate
        """
        super().__init__(n_arms, model)

        self.buffer_size = buffer_size
        self.g = g
        self.L = L
        self.epsilon = epsilon
        self.lambda_prior = lambda_prior
        self.a_0 = a_0
        self.b_0 = b_0
        self.P = P

        self.E = []

        self.context_dim = input_dim

        self.dnn = _DNN(d=self.context_dim, g=self.g, n_arms=self.n_arms)
        self.batch_size = batch_size
        self.lr = lr

        self.precision = [lambda_prior * torch.eye(g) for _ in range(n_arms)]
        self.precision_prior = [lambda_prior * torch.eye(g) for _ in range(n_arms)]

        self.cov = [(1.0 / lambda_prior) * torch.eye(g) for _ in range(n_arms)]

        self.mu = [torch.zeros(g) for _ in range(n_arms)]
        self.mu_prior = torch.zeros((g, n_arms))
        self.f = [torch.zeros(g) for _ in range(n_arms)]

        self.a = torch.full((n_arms,), fill_value=float(a_0))
        self.b = torch.full((n_arms,), fill_value=float(b_0))

        self.yy = torch.zeros(n_arms)

        self.regret = None

        self.it = 0
        self.min_buffer_size = min_buffer_size

    @staticmethod
    def make_positive_definite(matrix, min_eigenval=1e-6):
        if torch.is_tensor(matrix):
            matrix = matrix.numpy()

        matrix = (matrix + matrix.T) / 2

        eigenvals, eigenvecs = np.linalg.eigh(matrix)
        eigenvals = np.maximum(eigenvals, min_eigenval)

        result = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        return torch.tensor(result, dtype=torch.float32)

    @staticmethod
    def solve_sdp(new_contexts, old_contexts, old_precision, epsilon):
        if len(new_contexts) == 0 or len(old_contexts) == 0:
            return old_precision

        if torch.is_tensor(new_contexts):
            new_contexts = new_contexts.detach().cpu().numpy()
        if torch.is_tensor(old_contexts):
            old_contexts = old_contexts.detach().cpu().numpy()
        if torch.is_tensor(old_precision):
            old_precision = old_precision.detach().cpu().numpy()

        n, m = new_contexts.shape

        old_cov = np.linalg.inv(old_precision + epsilon * np.eye(m))
        d = []
        for c in old_contexts:
            confidence_score = np.dot(np.dot(c, old_cov), c.T)
            d.append(confidence_score)

        phi = []
        for c in new_contexts:
            phi.append(np.outer(c, c))

        X = cp.Variable((m, m), PSD=True)

        obj_terms = []
        for i in range(len(d)):
            trace_term = cp.trace(X @ phi[i])
            obj_terms.append((trace_term - d[i]) ** 2)

        objective = cp.Minimize(cp.sum(obj_terms))
        problem = cp.Problem(objective)

        try:
            problem.solve(solver=cp.SCS, eps=1e-6, max_iters=10000, verbose=False)

            if X.value is None or problem.status not in ["optimal", "optimal_inaccurate"]:
                return torch.tensor(old_precision + epsilon * np.eye(m), dtype=torch.float32)

            new_cov = X.value + epsilon * np.eye(m)
            return torch.tensor(new_cov, dtype=torch.float32)

        except Exception as e:
            print(f"[SDP] Solver failed: {e}")
            return torch.tensor(old_precision + epsilon * np.eye(m), dtype=torch.float32)

    def select_arm(self, context_matrix):
        """
        context_matrix: shape (n_arms, d)
        """
        context_matrix = torch.tensor(context_matrix, dtype=torch.float32)
        
        if self.it < self.n_arms * 10:
            a_t = self.it % self.n_arms
        else:
            r_hat = []
            
            phi_values = []
            for arm in range(self.n_arms):
                phi_t = self.dnn.get_phi(context_matrix[arm])
                phi_values.append(phi_t)

            for arm in range(self.n_arms):
                phi_t = phi_values[arm]
                sigma2_sample = self.b[arm].item() * invgamma.rvs(self.a[arm].item())

                try:
                    cov_scaled = sigma2_sample * self.cov[arm]
                    mu_sample = np.random.multivariate_normal(
                        self.mu[arm].detach().numpy(), 
                        cov_scaled.detach().numpy()
                    )
                    mu_sample = torch.tensor(mu_sample, dtype=torch.float32)
                except np.linalg.LinAlgError:
                    mu_sample = torch.zeros(self.g)
                
                r_hat.append((phi_t @ mu_sample).item())

            a_t = torch.argmax(torch.tensor(r_hat)).item()
        
        self.it += 1
        return int(a_t)

    def _update_buffer(self, context, arm, reward):
        if len(self.E) == self.buffer_size:
            self.E.pop(0)
        self.E.append((torch.tensor(context), arm, reward))

    def update(self, feedbacks: List[Dict[str, Any]]) -> None:
        trigger_training = False
        for fb in feedbacks:
            action = fb["action"]
            reward = fb["reward"]
            context = fb["context"]

            if context.ndim > 1:
                context = context[action]

            self._update_buffer(context, action, reward)

            context_tensor = torch.tensor(context, dtype=torch.float32)
            phi_t = self.dnn.get_phi(context_tensor)
            phi_t_flat = phi_t.squeeze()

            self.precision[action] += torch.outer(phi_t_flat, phi_t_flat)
            self.cov[action] = torch.linalg.inv(self.precision[action])

            self.f[action] += phi_t_flat * reward

            prior_contribution = torch.mv(self.precision_prior[action], self.mu_prior[:, action])
            self.mu[action] = torch.mv(self.cov[action], self.f[action] + prior_contribution)

            self.yy[action] += reward ** 2
            self.a[action] += 0.5

            b_update = 0.5 * self.yy[action]
            b_update += 0.5 * torch.dot(self.mu_prior[:, action], torch.mv(self.precision_prior[action], self.mu_prior[:, action]))
            b_update -= 0.5 * torch.dot(self.mu[action], torch.mv(self.precision[action], self.mu[action]))
            self.b[action] = self.b_0 + b_update

            if self.it % self.L == 0 and len(self.E) > self.min_buffer_size:
                trigger_training = True

        if trigger_training:
            old_features = [[] for _ in range(self.n_arms)]
            rewards_by_arm = [[] for _ in range(self.n_arms)]

            with torch.no_grad():
                for ctx, arm_idx, rew in self.E:
                    old_phi = self.dnn.get_phi(ctx).squeeze()
                    old_features[arm_idx].append(old_phi)
                    rewards_by_arm[arm_idx].append(rew)

            for arm_idx in range(self.n_arms):
                if len(old_features[arm_idx]) > 0:
                    old_features[arm_idx] = torch.stack(old_features[arm_idx])
                    rewards_by_arm[arm_idx] = torch.tensor(rewards_by_arm[arm_idx])

            self.update_NN()

            new_features = [[] for _ in range(self.n_arms)]

            with torch.no_grad():
                for ctx, arm_idx, __ in self.E:
                    new_phi = self.dnn.get_phi(ctx).squeeze()
                    new_features[arm_idx].append(new_phi)

            for arm_idx in range(self.n_arms):
                if len(new_features[arm_idx]) > 0:
                    new_features[arm_idx] = torch.stack(new_features[arm_idx])

            for arm_idx in range(self.n_arms):
                if (len(old_features[arm_idx]) > 0 and len(new_features[arm_idx]) > 0 and
                    old_features[arm_idx].shape[0] > 1 and new_features[arm_idx].shape[0] > 1):

                    old_precision = torch.linalg.inv(self.cov[arm_idx])
                    new_cov = self.solve_sdp(
                        new_features[arm_idx], old_features[arm_idx], old_precision, self.epsilon
                    )

                    new_cov = self.make_positive_definite(new_cov)
                    self.cov[arm_idx] = new_cov
                    self.precision_prior[arm_idx] = torch.linalg.inv(new_cov)

                    self.mu_prior[:, arm_idx] = self.dnn.fc3.weight[arm_idx].detach()

                    self.precision[arm_idx] = self.precision_prior[arm_idx].clone()
                    self.f[arm_idx] = torch.zeros(self.g)

                    for i in range(len(new_features[arm_idx])):
                        phi_i = new_features[arm_idx][i]
                        r_i = rewards_by_arm[arm_idx][i]
                        self.precision[arm_idx] += torch.outer(phi_i, phi_i)
                        self.f[arm_idx] += phi_i * r_i

                    self.cov[arm_idx] = torch.linalg.inv(self.precision[arm_idx])
                    prior_contrib = torch.mv(self.precision_prior[arm_idx], self.mu_prior[:, arm_idx])
                    self.mu[arm_idx] = torch.mv(self.cov[arm_idx], self.f[arm_idx] + prior_contrib)

    def update_NN(self):
        self.dnn.train()
        opt = Adam(self.dnn.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        input_data = torch.stack([e[0].squeeze() for e in self.E])
        rewards = torch.tensor([e[2] for e in self.E], dtype=torch.float32)
        arms = torch.tensor([e[1] for e in self.E], dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(input_data, arms, rewards)

        for _ in range(self.P):
            if self.batch_size > 0 and self.batch_size < len(dataset):
                indices = torch.randperm(len(dataset))[:self.batch_size]
                batch_input, batch_arms, batch_rewards = dataset[indices]
            else:
                batch_input, batch_arms, batch_rewards = dataset[:]

            opt.zero_grad()
            output = self.dnn(batch_input)
            filtered_output = output[torch.arange(len(output)), batch_arms]
            loss = criterion(filtered_output, batch_rewards)
            loss.backward()
            opt.step()
