"""
Linear Upper Confidence Bound (LinUCB) Algorithm.

This module re-exports UCBAlgorithm for backward compatibility 
when paired with OnlineRidgeRegression.
"""
from .ucb import UCBAlgorithm

LinUCBAlgorithm = UCBAlgorithm