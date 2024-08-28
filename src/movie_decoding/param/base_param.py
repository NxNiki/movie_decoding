import os
from pathlib import Path

import numpy as np
import torch

base_path = Path(__file__).resolve().parents[3]
param_dict = {
    "lr": 1e-4,
    "batch_size": 128,
    "weight_decay": 1e-4,
    "epochs": 100,
    "lr_drop": 50,
    "validation_step": 25,
    "num_labels": 8,
    "merge_label": True,
    "img_embedding_size": 192,
    "hidden_size_spike": 192,
    "hidden_size_lfp": 192,
    "num_hidden_layers": 4,
    "num_attention_heads": 6,
    "intermediate_size": 192 * 2,
    "classifier_proj_size": 192,
    # path
    "movie_label_path": str(base_path.joinpath("data/8concepts_merged.npy")),
    "spike_path": str(base_path.joinpath("data")),
    "lfp_path": "/mnt/SSD2/yyding/Datasets/neuron/lfp_data",
    "lfp_data_mode": "sf2000-bipolar-region-clean",
    "spike_data_mode": "notch CAR",
}

if torch.cuda.is_available():
    device = torch.device("cuda:1")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
