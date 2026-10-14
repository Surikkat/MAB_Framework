from .base import BaseModel
from .linear_model import OnlineRidgeRegression
from .fgts_model import FGTSModel
from .fgts_lasso_model import FGTSLassoModel
from .gp_rff_model import GPRFFModel
from .kernel_ucb_model import KernelUCBModel
from .exact_gp_model import ExactGPModel
from .sgd_model import SGDModel
from .bootstrap_model import BootstrapEnsembleModel

# Optional: cvxpy
try:
    from .glm_laplace_model import GLMLaplaceModel
except ImportError:
    pass

# Optional: torch
try:
    from .neural_network import NeuralLinearModel
    from .neural_ucb_model import NeuralUCBModel
    from .nn_ucb_model import NNUCBModel
    from .nn_agp_model import NNAGPModel
    from .cmab_models import NeuralNormalModel
except ImportError:
    pass

# Core CMAB models (без torch — но файл тянет torch, поэтому в try/except)
try:
    from .cmab_models import LinearNormalModel, GLMNormalModel
except ImportError:
    pass
