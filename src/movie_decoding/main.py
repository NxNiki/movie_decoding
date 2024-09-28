import argparse
import datetime
import os
import random
import string
import time
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch
import yaml
from trainer import Trainer
from utils.initializer import initialize_configs, initialize_dataloaders, initialize_evaluator, initialize_model

import wandb
from movie_decoding.param.base_param import device

# torch.autograd.set_detect_anomaly(True)
# torch.backends.cuda.matmul.allow_tf32=True
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
# torch.use_deterministic_algorithms(True)

# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True


def set_config(config_file: Union[str, Path], root_path: Union[str, Path]) -> Dict:
    """
    set parameters based on config file.
    :param config_file:
    :param root_path:
    :return:
    """
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    args = initialize_configs(config)
    if config.data_type == "clusterless":
        args["use_spike"] = True
        args["use_lfp"] = False
        args["use_combined"] = False
        model_architecture = "multi-vit"  # 'multi-vit'
    elif config.data_type == "lfp":
        args["use_spike"] = False
        args["use_lfp"] = True
        args["use_combined"] = False
        model_architecture = "multi-vit"
    elif config.data_type == "combined":
        args["use_spike"] = True
        args["use_lfp"] = True
        args["use_combined"] = True
        model_architecture = "multi-crossvit"
    else:
        ValueError(f"undefined data_type: {config.data_type}")

    args["seed"] = 42
    args["patient"] = config.patient
    args["use_spontaneous"] = False
    if config.use_clusterless:
        args["use_shuffle"] = True
    elif config.use_lfp:
        args["use_shuffle"] = False

    args["use_bipolar"] = False
    args["use_sleep"] = False
    args["use_overlap"] = False
    args["model_architecture"] = model_architecture

    args["spike_data_mode"] = data_directory
    args["spike_data_mode_inference"] = data_directory
    args["spike_data_sd"] = [sd]
    args["spike_data_sd_inference"] = sd
    args["use_augment"] = False
    args["use_long_input"] = False
    args["use_shuffle_diagnostic"] = False
    args["model_aggregate_type"] = "sum"

    output_folder = f"{args['patient']}_{config.data_type}_{model_architecture}_{suffix}"
    train_save_path = os.path.join(root_path, f"results/8concepts/{output_folder}/train")
    valid_save_path = os.path.join(root_path, f"results/8concepts/{output_folder}/valid")
    test_save_path = os.path.join(root_path, f"results/8concepts/{output_folder}/test")
    memory_save_path = os.path.join(root_path, f"results/8concepts/{output_folder}/memory")

    os.makedirs(train_save_path, exist_ok=True)
    os.makedirs(valid_save_path, exist_ok=True)
    os.makedirs(test_save_path, exist_ok=True)
    os.makedirs(memory_save_path, exist_ok=True)
    args["train_save_path"] = train_save_path
    args["valid_save_path"] = valid_save_path
    args["test_save_path"] = test_save_path
    args["memory_save_path"] = memory_save_path

    return args


def random_string(length: int = 3) -> str:
    letters = string.ascii_lowercase
    res = "".join(random.choice(letters) for i in range(length))
    return res


def pipeline(config: Dict[str, Any]) -> Trainer:
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"]) if torch.cuda.is_available() else None
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    dataloaders = initialize_dataloaders(config)
    model = initialize_model(config)
    # model = torch.compile(model)
    model = model.to(device)

    wandb.config.update(config)  # type: ignore
    # print(config)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])  # type: ignore
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config["lr_drop"])
    evaluator = initialize_evaluator(config, 1)

    # label_weights = dataset.label_weights
    trainer = Trainer(model, evaluator, optimizer, lr_scheduler, dataloaders, config)

    return trainer


if __name__ == "__main__":
    patient = "562"
    sd = 3.5
    data_directory = "notch CAR-quant-neg"
    early_stop = 75
    root_path = Path(__file__).resolve().parents[2]
    config_file = root_path / "src/movie_decoding/config/config.yaml"

    config = set_params(
        config_file,
    )

    print("start: ", patient)
    for run in range(5, 6):
        suffix = f"test53_optimalX_CARX_{run}"

        os.environ["WANDB_MODE"] = "offline"
        # os.environ['WANDB_API_KEY'] = '5a6051ed615a193c44eb9f655b81703925460851'
        wandb.login()
        if use_lfp:
            run_name = "LFP Concept level {} MultiEncoder".format(args["patient"])
        else:
            run_name = "Clusterless Concept level {} MultiEncoder".format(args["patient"])
        wandb.init(project="24_Concepts", name=run_name, reinit=True, entity="24")

        trainer = pipeline(args)

        print("Start training")
        # start_time = time.time()

        trainer.train(args["epochs"], 1)
    print("done: ", patient)
    print()
