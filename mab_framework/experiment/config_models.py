from pydantic import BaseModel, Field, root_validator
from typing import List, Dict, Any, Optional, Union, Literal

class ModelConfig(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)
    one_model_per_arm: bool = True

class AlgorithmConfig(BaseModel):
    name: str
    display_name: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    model: Optional[ModelConfig] = None
    one_model_per_arm: Optional[bool] = None

class DelayConfig(BaseModel):
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)

class EnvironmentConfig(BaseModel):
    type: Optional[str] = None
    name: Optional[str] = None
    n_arms: Optional[int] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    delay: Optional[DelayConfig] = None

    @root_validator(pre=True)
    def check_name_or_type(cls, values):
        if not values.get('name') and not values.get('type'):
            raise ValueError("Environment 'name' or 'type' must be specified in the config")
        if not values.get('name'):
            values['name'] = values.get('type')
        return values

class OutputConfig(BaseModel):
    save_path: str = "./results/experiment"

class GlobalExperimentConfig(BaseModel):
    name: str = "experiment"
    steps: int = 100
    n_runs: int = 1
    seed: Optional[int] = None
    device: str = "cpu"

class ExperimentConfig(BaseModel):
    experiment: GlobalExperimentConfig = Field(default_factory=GlobalExperimentConfig)
    environment: EnvironmentConfig
    algorithms: List[AlgorithmConfig]
    metrics: List[str] = ["cumulative_regret"]
    output: OutputConfig = Field(default_factory=OutputConfig)
