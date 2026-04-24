from .custom_ts_bandit import CustomTSBandit
from .sgd_ts_bandit import SGDTSBandit
from .gpts_bandit import GPTSBandit
from .gp_ucb_multikernel import GPUCBKernelFlexibleAlgorithm

try:
    from .regcb_bandit import RegcbBanit
except ImportError:
    pass
