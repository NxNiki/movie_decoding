import argparse
import datetime
import os
import random
import string
import time
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
import yaml
from trainer import Trainer
from utils.initializer import initialize_dataloaders, initialize_evaluator, initialize_model

import wandb
from brain_decoding.config.config import PipelineConfig
from brain_decoding.config.file_path import CONFIG_FILE_PATH
from brain_decoding.param.base_param import device

# torch.autograd.set_detect_anomaly(True)
# torch.backends.cuda.matmul.allow_tf32=True
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
# torch.use_deterministic_algorithms(True)

# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True


def set_config(
    config_file: Union[str, Path],
    patient_id: int,
    spike_data_sd: Union[List[int], int] = 3.5,
    spike_data_sd_inference: int = 3.5,
) -> PipelineConfig:
    """
    set parameters based on config file.
    :param config_file:
    :param patient_id:
    :param spike_data_sd:
    :param spike_data_sd_inference:
    :return:
    """

    if isinstance(spike_data_sd, int):
        spike_data_sd = [spike_data_sd]

    config = PipelineConfig.read_config(config_file)

    config.experiment["patient"] = patient_id
    config.experiment.name = "8concepts"
    config.data.spike_data_sd = spike_data_sd
    config.data.spike_data_sd_inference = spike_data_sd_inference

    output_folder = f"{patient_id}_{config.data.data_type}_{config.model.architecture}_test53_optimalX_CARX"
    output_path = os.path.join(config.data.result_path, config.experiment.name, output_folder)
    config.data.train_save_path = os.path.join(output_path, "train")
    config.data.valid_save_path = os.path.join(output_path, "valid")
    config.data.test_save_path = os.path.join(output_path, "test")
    config.data.memory_save_path = os.path.join(output_path, "memory")

    return config


def random_string(length: int = 3) -> str:
    letters = string.ascii_lowercase
    res = "".join(random.choice(letters) for i in range(length))
    return res


def pipeline(config: PipelineConfig) -> Trainer:
    torch.manual_seed(config.experiment["seed"])
    torch.cuda.manual_seed(config.experiment["seed"]) if torch.cuda.is_available() else None
    np.random.seed(config.experiment["seed"])
    random.seed(config.experiment["seed"])

    dataloaders = initialize_dataloaders(config)
    model = initialize_model(config)
    # model = torch.compile(model)
    model = model.to(device)

    wandb.config.update(config)  # type: ignore
    # print(config)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.model["lr"], weight_decay=config.model["weight_decay"])  # type: ignore
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.model["lr_drop"])
    evaluator = initialize_evaluator(config, 1)

    # label_weights = dataset.label_weights
    trainer = Trainer(model, evaluator, optimizer, lr_scheduler, dataloaders, config)

    return trainer


if __name__ == "__main__":
    patient = 562
    config_file = CONFIG_FILE_PATH / "config_test-None-None_2024-10-02-17:31:47.yaml"

    config = set_config(
        config_file,
        patient,
    )

    print("start: ", patient)

    os.environ["WANDB_MODE"] = "offline"
    # os.environ['WANDB_API_KEY'] = '5a6051ed615a193c44eb9f655b81703925460851'
    wandb.login()
    run_name = f"LFP Concept level {config.experiment['patient']} MultiEncoder"
    wandb.init(project="24_Concepts", name=run_name, reinit=True, entity="24")

    trainer = pipeline(config)

    print("Start training")
    trainer.train(config.model["epochs"], 1)
    print("done: ", patient)
    print()
