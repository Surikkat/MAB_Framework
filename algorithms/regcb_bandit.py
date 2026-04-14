from coba.learners import VowpalRegcbLearner
from .base import BaseAlgorithm
import numpy as np
from typing import List, Dict, Any


class RegcbBanit(BaseAlgorithm):
    def __init__(self,
                 n_arms: int,
                 context_dim: int,
                 model=None
                 ):
        super().__init__(n_arms, model)
        self.actions = list(range(n_arms))
        self.action_prob = None
        self.action_probs = None
        self.d = context_dim
        self.bandit = VowpalRegcbLearner()
    
    def select_arm(self, context: np.ndarray) -> int:
        actions_chosen = []
        action_probs = []
        for action_idx in self.actions:
            action, action_prob, kwargs = self.bandit.predict(context=context[action_idx].tolist(),
                                                            actions=self.actions)
            actions_chosen.append(action)
            action_probs.append(action_prob)
        self.action_probs = action_probs
        return int(action)
    
    def update(self, feedbacks: List[Dict[str, Any]]) -> None:
        for fb in feedbacks:
            action = fb["action"]
            reward = fb["reward"]
            context = fb["context"]
            arm = action

            if int(arm) < len(self.action_probs) and self.action_probs[int(arm)] is not None:
                context = np.asarray(context, dtype=float).flatten().tolist()
                self.bandit.learn(context=context,
                                action=int(arm),
                                reward=float(reward),
                                probability=float(self.action_probs[int(arm)]))
            else:
                raise NotImplementedError("Right now the not None action prob is required! Set it firstly")

