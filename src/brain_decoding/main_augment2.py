import argparse
import datetime
import os
import random
import string
import time

import numpy as np
import torch
from trainer import Trainer
from utils.initializer import *

import wandb

# torch.autograd.set_detect_anomaly(True)
# torch.backends.cuda.matmul.allow_tf32=True
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
# torch.use_deterministic_algorithms(True)

# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True


def pipeline(config):
    device = torch.device(config["device"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"]) if torch.cuda.is_available() else None
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    dataloaders = initialize_dataloaders(config)
    model = initialize_model(config)
    # model = torch.compile(model)
    model = model.to(device)

    wandb.config.update(config)
    # print(config)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config["lr_drop"])
    evaluator = initialize_evaluator(config, 1)

    # label_weights = dataset.label_weights
    trainer = Trainer(model, evaluator, optimizer, lr_scheduler, dataloaders, config)

    return trainer


if __name__ == "__main__":
    # for patient in ['562', '563', '566', 'i728', '567', '572']:
    patient_list = ["i728", "572", "567", "566"]  # , '563', '562']
    # sd_list = [[4, 4.5, 5], [3, 3.5, 4], [4, 4.5, 5], [4.5, 4, 5], [4.5, 3, 5], [4, 3.5, 4.5]]
    sd_list = [[3.5, 4], [3.5, 4], [3.5, 4], [3.5, 4]]  # , [4, 3, 5], [4, 3, 5]]
    # data_list = ['notch CAR4.5', 'notch CAR3.5', 'notch CAR4.5', 'notch CAR4', 'notch CAR3.5', 'notch CAR3.5']
    data_list = [
        "notch CAR-quant-neg",
        "notch CAR-quant-neg",
        "notch CAR-quant-neg",
        "notch CAR-quant-neg",
        "notch CAR-quant-neg",
        "notch CAR-quant-neg",
    ]
    early_stop = [100, 100, 100, 50, 50, 75]
    for patient, sd, dd in zip(patient_list, sd_list, data_list):
        print()
        print("start: ", patient)
        for data_type in ["clusterless"]:
            for run in range(6, 7):
                root_path = os.path.dirname(os.path.abspath(__file__))
                # save the results
                letters = string.ascii_lowercase
                # suffix = ''.join(random.choice(letters) for i in range(3))
                suffix = f"test53_optimalX_CARX_{run}"
                if data_type == "clusterless":
                    use_clusterless = True
                    use_lfp = False
                    use_combined = False
                    model_architecture = "multi-vit"  #'multi-vit'
                elif data_type == "lfp":
                    use_clusterless = False
                    use_lfp = True
                    use_combined = False
                    model_architecture = "multi-vit"
                elif data_type == "combined":
                    use_clusterless = True
                    use_lfp = True
                    use_combined = True
                    model_architecture = "multi-crossvit"
                args = initialize_configs(architecture=model_architecture)
                args["seed"] = 42
                args["device"] = "cuda:1"
                args["patient"] = patient
                args["use_spike"] = use_clusterless
                args["use_lfp"] = use_lfp
                args["use_combined"] = use_combined
                args["use_spontaneous"] = False
                if use_clusterless:
                    args["use_shuffle"] = True
                elif use_lfp:
                    args["use_shuffle"] = False

                args["use_bipolar"] = False
                args["use_sleep"] = False
                args["use_overlap"] = False
                args["model_architecture"] = model_architecture

                args["spike_data_mode"] = dd
                args["spike_data_mode_inference"] = dd
                args["spike_data_sd"] = sd
                args["spike_data_sd_inference"] = sd[0]
                args["model_aggregate_type"] = "mean"
                args["use_augment"] = False
                args["use_long_input"] = False
                args["use_shuffle_diagnostic"] = False

                train_save_path = os.path.join(
                    root_path,
                    "results/8concepts/{}_{}_{}_{}/train".format(
                        args["patient"], data_type, model_architecture, suffix
                    ),
                )
                valid_save_path = os.path.join(
                    root_path,
                    "results/8concepts/{}_{}_{}_{}/valid".format(
                        args["patient"], data_type, model_architecture, suffix
                    ),
                )
                test_save_path = os.path.join(
                    root_path,
                    "results/8concepts/{}_{}_{}_{}/test".format(args["patient"], data_type, model_architecture, suffix),
                )
                memory_save_path = os.path.join(
                    root_path,
                    "results/8concepts/{}_{}_{}_{}/memory".format(
                        args["patient"], data_type, model_architecture, suffix
                    ),
                )
                os.makedirs(train_save_path, exist_ok=True)
                os.makedirs(valid_save_path, exist_ok=True)
                os.makedirs(test_save_path, exist_ok=True)
                os.makedirs(memory_save_path, exist_ok=True)
                args["train_save_path"] = train_save_path
                args["valid_save_path"] = valid_save_path
                args["test_save_path"] = test_save_path
                args["memory_save_path"] = memory_save_path

                os.environ["WANDB_MODE"] = "offline"
                # os.environ['WANDB_API_KEY'] = '5a6051ed615a193c44eb9f655b81703925460851'
                wandb.login()
                if use_lfp:
                    run_name = "LFP Concept level {} MultiEncoder".format(args["patient"])
                else:
                    run_name = "Clusterless Concept level {} MultiEncoder".format(args["patient"])
                wandb.init(
                    project="memory_decoding",
                    name=run_name,
                    reinit=True,
                    entity="uoops",
                )

                trainer = pipeline(args)

                print("Start training")
                # start_time = time.time()

                trainer.train(args["epochs"], 1)
        print("done: ", patient)
        print()
