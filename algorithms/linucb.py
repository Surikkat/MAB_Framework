"""LinUCB Algorithm.

LinUCB is mathematically identical to the generic UCB algorithm
when paired with OnlineRidgeRegression as the model. Both compute
UCB = mu + alpha * sigma. This module re-exports UCBAlgorithm
for backward compatibility.
"""
from .ucb import UCBAlgorithm

LinUCBAlgorithm = UCBAlgorithm