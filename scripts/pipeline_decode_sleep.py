"""
train the model with data while patients are viewing the movie Twilight and 24 and test the model with data during free
recall. The model will be trained to predict movie (Twilight vs 24, model 1) or character in both movies (model 2).
Finally, we will apply the model (with better performance) on data during sleep to examine how reactivation/replay will
impact free recall before and after sleep.
"""

import os
import string
from pathlib import Path

import wandb
from brain_decoding.config.file_path import CONFIG_FILE_PATH
from brain_decoding.main import pipeline, set_config
from brain_decoding.utils.initializer import *

PROJECT_NAME = "sleep_decoding"

# patient_ids = [1728, 572, 567, 566, 563, 562]
patient_ids = [562]
sd_list = [4, 4, 3.5, 4, 4, 3.5]
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

for patient in patient_ids:
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
    wandb.init(project=PROJECT_NAME, name=run_name, reinit=True, entity="24")

    trainer = pipeline(config)

    print("Start training")
    trainer.train(config.model["epochs"], 1)
    print("done: ", patient)
    print()
