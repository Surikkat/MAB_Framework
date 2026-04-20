from .base import BaseAlgorithm
from .epsilon_greedy import EpsilonGreedy
from .ucb import UCBAlgorithm
from .thompson_sampling import ThompsonSampling
from .nn_agp_ucb import NNAGPUCBAlgorithm

# Added from Batch 3
from .neural_ucb import NeuralUCBAlgorithm
from .nn_ucb import NNUCBAlgorithm
from .gp_ucb_multikernel import GPUCBKernelFlexibleAlgorithm
from .nn_agp_adaptive import NNAGPUCBAdaptiveAlgorithm

# Added from Batch 4
from .nn_ts_b import NNTSBAlgorithm
from .custom_ts_bandit import CustomTSBandit
from .bootstrap_ts_bandit import BootstrapTSBandit
from .sgd_ts_bandit import SGDTSBandit
from .noncontextual_ts_bandit import NonContextualTSBandit
# from .regcb_bandit import RegcbBanit
from .nn_bandit_limited_memory import NeuralBanditWithLimitedMemory_5
from .gpts_bandit import GPTSBandit

# Aliases: these algorithms are mathematically identical to UCBAlgorithm
# (all compute mu + alpha * sigma). Kept for backward compatibility and
# semantic clarity in experiment configs.
LinUCBAlgorithm = UCBAlgorithm

# FGTSAlgorithm is identical to ThompsonSampling (both call model.sample()).
FGTSAlgorithm = ThompsonSampling