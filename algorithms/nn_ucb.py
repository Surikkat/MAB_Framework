"""NN_UCB — exact reproduction of NN_AGP/bandits/nn_ucb.py.

Diagonal approximation NN-UCB with gradient features.
This has been decoupled to interact solely with BaseAlgorithms.
"""
import numpy as np
from typing import List, Dict, Any
from .base import BaseAlgorithm


class NNUCBAlgorithm(BaseAlgorithm):
    """Exact port of NN_AGP/bandits/nn_ucb.py NN_UCB class decoupled from PyTorch.

    Uses diagonal approximation for confidence (A_diag instead of full matrix).
    """
    def __init__(self,
                 n_arms: int,
                 model,
                 lambda_: float = 1.0,
                 beta: float = 1.0):
        super().__init__(n_arms, model)
        self.lambda_ = float(lambda_)
        
        if hasattr(self.model, "num_params"):
            num_params = self.model.num_params
        else:
            raise ValueError("Provided model to NNUCBAlgorithm must expose 'num_params' for A_diag sizing.")
            
        self.A_diag = (self.lambda_) * np.ones(num_params, dtype=float)
        self.beta = float(beta)
        self.eps = 1e-8

    def select_arm(self, context: np.ndarray) -> int:
        if isinstance(context, np.ndarray):
            ctxs = context
        else:
            ctxs = np.array(context)

        if ctxs.ndim == 1:
            ctxs = ctxs.reshape(1, -1)

        k = ctxs.shape[0]
        ucbs = []

        for i in range(k):
            x_np = ctxs[i]
            
            mu, _ = self.model.predict(x_np)
            f_val = float(mu[0])
            
            phi = self.model.get_grad_features(x_np)
            denom = self.A_diag
            v = float(np.sqrt(np.sum((phi ** 2) / (denom + self.eps))))
            
            ucb = f_val + self.beta * v
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

            ctx = np.array(ctx, dtype=np.float32)
            
            self.model.fit(ctx, float(reward))
            new_ctxs.append(ctx)

        if len(feedbacks) == 0:
            return

        self.model.finalize_update()

        for ctx in new_ctxs:
            phi = self.model.get_grad_features(ctx)
            self.A_diag += (phi ** 2)
