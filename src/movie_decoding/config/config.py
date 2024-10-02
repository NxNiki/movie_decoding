from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, Field


class BaseConfig(BaseModel):
    alias: Dict[str, str] = {}
    param: Dict[str, Any] = {}

    def __getitem__(self, key: str) -> Any:
        if key in self.param:
            return self.param[key]
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        if key in self.model_fields:
            setattr(self, key, value)
        else:
            self.param[key] = value

    def __getattr__(self, name):
        """Handles alias access and custom parameters."""
        if name in self.alias:
            return getattr(self, self.alias[name])
        if name in self.param:
            return self.param[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        """Handles alias assignment, field setting, or adding to _param."""
        if name in self.alias:
            name = self.alias[name]

        # Check if it's a field defined in the model
        if name in self.model_fields:
            super().__setattr__(name, value)
        else:
            # Otherwise, treat it as a custom parameter
            self.param[name] = value

    def __contains__(self, key: str) -> bool:
        return key in self.param or hasattr(self, key)


class ExperimentConfig(BaseConfig):
    """
    configurations regarding the experiment
    """

    name: Optional[str] = None
    patient: Optional[Union[List[int], int]] = None


class ModelConfig(BaseConfig):
    name: Optional[str] = None
    learning_rate: Optional[float] = Field(1e-4, alias="lr")
    learning_rate_drop: Optional[int] = Field(50, alias="lr_drop")
    batch_size: Optional[int] = 128
    epochs: Optional[int] = 100
    hidden_size: Optional[int] = 192
    num_hidden_layers: Optional[int] = 4
    num_attention_heads: Optional[int] = 6
    patch_size: Optional[Tuple[int, int]] = None

    alias: Dict[str, str] = {
        "lr": "learning_rate",
        "lr_drop": "learning_rate_drop",
    }


class DataConfig(BaseConfig):
    data_type: Optional[str] = None
    sd: Optional[float] = None
    root_path: Optional[Union[str, Path]] = None
    data_path: Optional[Union[str, Path]] = None


class PipelineConfig(BaseModel):
    experiment: Optional[ExperimentConfig] = ExperimentConfig()
    model: Optional[ModelConfig] = ModelConfig()
    data: Optional[DataConfig] = DataConfig()

    # class Config:
    #     arbitrary_types_allowed = True

    @classmethod
    def read_config(cls, config_file: Union[str, Path]) -> "PipelineConfig":
        """Reads a YAML configuration file and returns an instance of PipelineConfig."""
        with open(config_file, "r") as file:
            config_dict = yaml.safe_load(file)
        return cls(**config_dict)

    def export_config(self, output_file: Union[str, Path] = "config.yaml") -> None:
        """Exports current properties to a YAML configuration file."""
        if isinstance(output_file, str):
            output_file = Path(output_file)

        if not output_file.suffix:
            output_file = output_file / "config.yaml"

        # Create new path with the suffix added before the extension
        output_file = output_file.with_name(f"{output_file.stem}{self._file_tag}{output_file.suffix}")

        dir_path = output_file.parent
        dir_path.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as file:
            yaml.safe_dump(self.model_dump(), file)

    @property
    def _file_tag(self) -> str:
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d-%H:%M:%S")
        return f"_{self.experiment.name}-{self.model.name}-{self.data.data_type}_{formatted_time}"


if __name__ == "__main__":
    pipeline_config = PipelineConfig()
    pipeline_config.model.name = "vit"
    pipeline_config.model.learning_rate = 0.001
    pipeline_config.experiment.name = "movie-decoding"

    # Access and print properties
    print(f"Experiment Name: {pipeline_config.experiment.name}")
    print(f"Patient ID: {pipeline_config.experiment.patient}")
    print(f"Model Name: {pipeline_config.model.name}")
    print(f"Learning Rate: {pipeline_config.model.learning_rate}")
    print(f"Batch Size: {pipeline_config.model.batch_size}")

    # Access using aliases
    print(f"Learning Rate (alias 'lr'): {pipeline_config.model['lr']}")
    print(f"Learning Rate (alias 'lr'): {pipeline_config.model.lr}")

    # Set new custom parameters
    pipeline_config.model["new_param"] = "custom_value"
    print(f"Custom Parameter 'new_param': {pipeline_config.model['new_param']}")
    pipeline_config.model.new_param2 = "custom_value"
    print(f"Custom Parameter 'new_param2': {pipeline_config.model.new_param2}")

    # Try to access a non-existent field (will raise AttributeError)
    try:
        print(pipeline_config.model.some_non_existent_field)
    except AttributeError as e:
        print(e)

    # Export config:
    pipeline_config.export_config()
