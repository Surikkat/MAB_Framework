"""BanditLab — a modular framework for multi-armed bandit research."""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("banditlab")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

from .algorithms import (
    BaseAlgorithm, EpsilonGreedy, UCBAlgorithm, ThompsonSampling,
    NonContextualTSBandit, BootstrapTSBandit, FGTSAlgorithm,
    DelayedThompsonSampling, JoulaniDelayedUCB, PatientBandits,
    VernadeDelayedUCB, LinUCBAlgorithm, CustomTSBandit, SGDTSBandit,
    GPTSBandit, GPUCBKernelFlexibleAlgorithm, RegCBBandit,
)
from .environments import (
    BaseEnvironment, DatasetEnvironment, DatasetEnvironmentFactory,
    NPZDatasetEnv, CSVDatasetEnv, JSONDatasetEnv, FolderDatasetEnv,
    SyntheticLinearEnv, SyntheticGLMEnv, SyntheticNeuralEnv,
    SyntheticNonContextualEnv, LogDataEnvironment,
)
from .models import (
    BaseModel, OnlineRidgeRegression, FGTSModel, FGTSLassoModel,
    GPRFFModel, KernelUCBModel, ExactGPModel, SGDModel,
    BootstrapEnsembleModel,
)

# Neural algorithms (require [deep] / torch)
try:
    from .algorithms import (
        NeuralUCBAlgorithm,
        NNUCBAlgorithm,
        NNTSBAlgorithm,
        NNAGPUCBAlgorithm,
        NNAGPUCBAdaptiveAlgorithm,
        NeuralBanditWithLimitedMemory_5,
    )
except ImportError:
    pass

# Optional models (neural / cvxpy-dependent)
try:
    from .models import (
        NeuralLinearModel, NeuralUCBModel, NNUCBModel, NNAGPModel,
        GLMLaplaceModel, LinearNormalModel, GLMNormalModel, NeuralNormalModel,
    )
except ImportError:
    pass