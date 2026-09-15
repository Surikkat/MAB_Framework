from .base import BaseAlgorithm

from .non_contextual import (
    EpsilonGreedy,
    UCBAlgorithm,
    ThompsonSampling,
    NonContextualTSBandit,
    BootstrapTSBandit,
    FGTSAlgorithm,
    JoulaniDelayedUCB,
    PatientBandits,
    VernadeDelayedUCB,
    DelayedThompsonSampling,
)

from .linear import (
    LinUCBAlgorithm,
    CustomTSBandit,
    SGDTSBandit,
)

from .gaussian_process import (
    GPTSBandit,
    GPUCBKernelFlexibleAlgorithm,
)

# try:
#     from .glm import RegCBBandit
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
        PFNTSAlgorithm,
    )
except ImportError:
    pass