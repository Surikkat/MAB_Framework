"""FGTS Algorithm.

FGTS is mathematically identical to Thompson Sampling: both select
the arm with the highest sampled value from the posterior. The difference
lies in the *model* (FGTSModel with feature gating), not the algorithm.
This module re-exports ThompsonSampling for backward compatibility.
"""
from .thompson_sampling import ThompsonSampling

FGTSAlgorithm = ThompsonSampling