"""NN_TS_B — exact reproduction of cMAB_bandits/bandits/nn_ts_b.py.

Neural Thompson Sampling with per-arm networks.
"""
import numpy as np
from typing import List, Dict, Any
import torch
import torch.nn as nn
from itertools import chain
from torch.optim import Adam
from .base import BaseAlgorithm


class NNTSBAlgorithm(BaseAlgorithm):
    """Exact port of cMAB_bandits/bandits/nn_ts_b.py NN_TS_B class."""
    def __init__(self, n_arms: int, d: int, v: float = 0.1, width: int = 10, depth: int = 2,
                 reg: float = 1, e: float = 0.01, lr: float = 0.01,
                 stop_rounds: int = 1000, max_steps: int = 100, model=None):
        super().__init__(n_arms, model)
        self.max_steps = max_steps
        self.current_round = 0
        self.stop_rounds = stop_rounds
        self.lr = lr
        self.arms = n_arms
        self.v = v
        self.e = e
        self.width = width
        assert depth >= 2, 'depth needs to be >= 2'
        self.depth = depth
        assert reg > 0, 'reg must be positive'
        self.reg = reg
        self.d = d
        self.NNs = [nn.Sequential(nn.Linear(d, width), nn.ReLU(),
                     *list(chain(*zip([nn.Linear(width, width) for i in range(depth-2)],[nn.ReLU() for i in range(depth-2)]))),
                     nn.Linear(width, 1)) for j in range(n_arms)]
        self.contexts = [[] for arm in range(n_arms)]
        self.rewards = [[] for arm in range(n_arms)]
        self.opts = [Adam(self.NNs[i].parameters(), lr=self.lr) for i in range(self.arms)]
        self.init_thetas = [torch.cat([params.data.clone().reshape(-1) for params in self.NNs[i].parameters()], dim=0) for i in range(n_arms)]
        self.parameters_d = sum(torch.prod(torch.tensor(params.data.shape)) for params in self.NNs[0].parameters())
        self.U = [reg*torch.eye(self.parameters_d, dtype=torch.float32) for i in range(n_arms)]

    def __count_grad(self, outputs):
        for output in outputs:
            output.backward()

    def __L(self, num_of_net, reward):
        loss = torch.zeros(1, requires_grad=True)
        init_theta = self.init_thetas[num_of_net]
        for context, reward in zip(self.contexts[num_of_net], self.rewards[num_of_net]):
            loss = loss + (self.NNs[num_of_net](context.unsqueeze(0)).reshape(-1)*self.width**(0.5)-reward)**2/2/len(self.contexts[num_of_net])
        for i, params in enumerate(self.NNs[num_of_net].parameters()):
            loss = loss + torch.linalg.norm(params - init_theta[i])**2*self.width*self.reg/2
        return loss

    def select_arm(self, context: np.ndarray) -> int:
        context_t = torch.FloatTensor(context)
        outputs = [self.NNs[i](context_t[i].unsqueeze(0))*self.width**(0.5) for i in range(self.arms)]
        rewards = torch.zeros(self.arms)
        if self.current_round < self.stop_rounds:
            sigmas = []
            self.__count_grad(outputs)
            parameters_grad_tensor = []
            for i in range(self.arms):
                parameters_grad_tensor_one_net = []
                for parames in self.NNs[i].parameters():
                    parameters_grad_tensor_one_net.append(parames.grad.reshape(-1))
                parameters_grad_tensor_one_net = torch.cat(parameters_grad_tensor_one_net, dim=0)
                parameters_grad_tensor.append(parameters_grad_tensor_one_net)
            for arm in range(self.arms):
                sigma = self.reg*parameters_grad_tensor[arm]*((torch.diag(self.U[arm]))**(-1))@parameters_grad_tensor[arm]/self.width
                sigmas.append(sigma)
                rewards[arm] = torch.randn(1)*abs(sigma*self.v)+outputs[arm]
            for arm in range(self.arms):
                self.opts[arm].zero_grad()
        else:
            for arm in range(self.arms):
                rewards[arm] = outputs[arm]
        arm = rewards.argmax()
        return int(arm)

    def update(self, feedbacks: List[Dict[str, Any]]) -> None:
        new_feedbacks_by_action = {i: [] for i in range(self.arms)}
        
        for fb in feedbacks:
            action = fb["action"]
            reward = fb["reward"]
            context = fb["context"]
            self.current_round += 1
            if self.current_round > self.stop_rounds:
                continue

            if context.ndim > 1:
                ctx = context[action]
            else:
                ctx = context

            ctx_t = torch.FloatTensor(ctx)
            self.rewards[action].append(reward)
            self.contexts[action].append(ctx_t)
            new_feedbacks_by_action[action].append(ctx_t)

        for action, new_ctxs in new_feedbacks_by_action.items():
            if len(new_ctxs) == 0:
                continue

            previous_loss = float('inf')
            step = 0
            while True:
                step += 1
                self.opts[action].zero_grad()
                loss = self.__L(action, 0.0)
                loss.backward()
                self.opts[action].step()
                if 1 - loss / previous_loss < self.e or step > self.max_steps:
                    break
                else:
                    previous_loss = loss

            for ctx_t in new_ctxs:
                self.opts[action].zero_grad()
                (self.NNs[action](ctx_t.unsqueeze(0))*self.width**(0.5)).reshape(-1).backward()
                parameters_grad_tensor_one_net = []
                for parames in self.NNs[action].parameters():
                    parameters_grad_tensor_one_net.append(parames.grad.reshape(-1))
                parameters_grad_tensor_one_net = torch.cat(parameters_grad_tensor_one_net, dim=0)
                self.U[action] += parameters_grad_tensor_one_net.reshape(-1, 1)@parameters_grad_tensor_one_net.reshape(1, -1)/self.width

