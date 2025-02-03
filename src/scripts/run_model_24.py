import datetime
import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch

import wandb
from brain_decoding.config.file_path import CONFIG_FILE_PATH, DATA_PATH, MOVIE24_LABEL_PATH
from brain_decoding.config.save_config import config
from brain_decoding.main import pipeline, set_config
from brain_decoding.param.param_data import MOVIE24_LABELS
from brain_decoding.utils.analysis import concept_frequency

if __name__ == "__main__":
    patient = 567
    phase_train = "movie_24_1"
    phase_test = "sleep_1"
    experiment_name = "movie_24_merged"
    CONFIG_FILE = CONFIG_FILE_PATH / "config_sleep-None-None_2024-10-16-19:17:43.yaml"

    config = set_config(
        # CONFIG_FILE,
        config,
        patient,
        experiment_name,
        phase_train,
        phase_test,
    )

    config.data.movie_label_path = MOVIE24_LABEL_PATH
    config.data.movie_label_sr = 1  # 1 for 24, 4 for twilight

    config.model.labels = MOVIE24_LABELS
    config.model.num_labels = len(config.model.labels)

    config.data.spike_data_mode = "notch"
    config.data.spike_data_mode_inference = "notch"

    # _, concept_weights = concept_frequency(config.data.movie_label_path, config.model.labels)
    # concept_weights = torch.from_numpy(concept_weights.astype(np.float32))
    # concept_weights = concept_weights.to(device)
    # config.model.train_loss = torch.nn.BCEWithLogitsLoss(reduction="none", weight=concept_weights)

    print("start: ", patient)

    os.environ["WANDB_MODE"] = "offline"
    # os.environ['WANDB_API_KEY'] = '5a6051ed615a193c44eb9f655b81703925460851'
    wandb.login()
    run_name = f"LFP Concept level {config.experiment['patient']} MultiEncoder"
    wandb.init(project=experiment_name, name=run_name, reinit=True, entity="24")

    trainer = pipeline(config)

    print("Start training")
    trainer.train(config.model["epochs"], 1)
    print("done: ", patient)
    print()
