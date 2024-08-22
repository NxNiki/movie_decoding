import argparse
import datetime
import os
import random
import string
import time

import numpy as np
import torch
import wandb
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from trainer import Trainer
from utils.initializer import *

# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


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

    # wandb.config.update(config)
    print(config)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config["lr_drop"])
    evaluator = initialize_evaluator(config, 1)

    # label_weights = dataset.label_weights
    trainer = Trainer(model, evaluator, optimizer, lr_scheduler, dataloaders, config)

    trainer.train(config["epochs"], 1)


if __name__ == "__main__":
    patient_list = ["566"]
    sd_list = [[4, 3.5]]
    # data_list = ['notch CAR4.5', 'notch CAR3.5', 'notch CAR4.5', 'notch CAR4', 'notch CAR3.5', 'notch CAR3.5']
    data_list = ["notch CAR-quant-neg"]
    for patient, sd, dd in zip(patient_list, sd_list, data_list):
        for data_type in ["clusterless"]:
            root_path = os.path.dirname(os.path.abspath(__file__))
            # save the results
            letters = string.ascii_lowercase
            # suffix = ''.join(random.choice(letters) for i in range(3))
            suffix = f"test53_optimalX_CARX_search"
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
            args["device"] = "cuda:0"
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
                "results/8concepts/{}_{}_{}_{}/test".format(
                    args["patient"], data_type, model_architecture, suffix
                ),
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

            # args['lr'] = tune.uniform(1e-6, 1e-4)
            args["num_hidden_layers"] = tune.grid_search([6, 5, 3])
            # args['hidden_size'] = tune.grid_search([192 * 3, 192 * 2, 192])
            # args['patch_size'] = tune.grid_search([(1, 10)])
            scheduler = ASHAScheduler(
                grace_period=1,
                reduction_factor=2,
            )

            # result = tune.run(
            #     pipeline,
            #     resources_per_trial={"cpu": 8, "gpu": 1},
            #     config=args,
            #     scheduler=scheduler,
            # )

            tuner = tune.Tuner(
                tune.with_resources(
                    tune.with_parameters(pipeline), resources={"cpu": 2, "gpu": 1}
                ),
                tune_config=tune.TuneConfig(
                    metric="recall",
                    mode="max",
                    scheduler=scheduler,
                ),
                param_space=args,
            )
            results = tuner.fit()

            best_result = results.get_best_result("recall", "max")
            print("Best trial config: {}".format(best_result.config))
            print(
                "Best trial final validation loss: {}".format(
                    best_result.metrics["recall"]
                )
            )
            print(
                "Best trial final validation accuracy: {}".format(
                    best_result.metrics["jack"]
                )
            )
