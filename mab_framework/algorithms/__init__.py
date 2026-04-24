from .base import BaseAlgorithm

from .stochastic import (
    EpsilonGreedy,
    UCBAlgorithm,
    ThompsonSampling,
    NonContextualTSBandit,
    BootstrapTSBandit,
    LinUCBAlgorithm,
    FGTSAlgorithm,
)

from .contextual import (
    CustomTSBandit,
    SGDTSBandit,
    GPTSBandit,
    GPUCBKernelFlexibleAlgorithm,
)

# try:
#     from .contextual import RegcbBanit
# except ImportError:
#     pass

try:
    from .neural import (
        NeuralUCBAlgorithm,
        NNUCBAlgorithm,
        NNTSBAlgorithm,
        NNAGPUCBAlgorithm,
        NNAGPUCBAdaptiveAlgorithm,
        NeuralBanditWithLimitedMemory_5,
    )
except ImportError:
    pass