import os

import numpy as np

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
    "hidden_size_lfp": 96,
    "num_hidden_layers": 4,
    "num_attention_heads": 6,
    "intermediate_size": 192 * 2,
    # 'classifier_proj_size': 192,
    # path
    # 'movie_label_path': '/mnt/SSD2/yyding/Datasets/12concepts/12concepts_john.npy',
    "movie_label_path": "/mnt/SSD2/yyding/Datasets/12concepts/8concepts_merged.npy",
    "spike_path": "/mnt/SSD2/yyding/Datasets/neuron/spike_data",
    "lfp_path": "/mnt/SSD2/yyding/Datasets/neuron/lfp_data",
    "lfp_data_mode": "sf2000-bipolar-region",
    "spike_data_mode": "notch CAR",
}
