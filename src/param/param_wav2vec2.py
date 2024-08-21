import numpy as np 
import os


param_dict={
    'lr': 1e-4,
    'batch_size': 128,
    'weight_decay': 1e-4,
    'epochs': 100,
    'lr_drop': 50,
    'validation_step': 20,
    'num_labels': 12,
    'img_embedding_size': 64,
    'hidden_size': 64,
    'num_hidden_layers': 3,
    'num_attention_heads': 4,
    'intermediate_size': 128,
    # path
    'movie_label_path': '/mnt/SSD2/yyding/Datasets/12concepts/12concepts_john.npy',
    # 'movie_label_path': '/mnt/SSD5/yyding/Datasets/12concepts/12concepts.npy',
    'spike_path': '/mnt/SSD2/yyding/Datasets/neuron/spike_data',
    'lfp_path': '/mnt/SSD2/yyding/Datasets/neuron/lfp_data',
    'lfp_data_mode': 'sf2000-bipolar-full',
    'spike_data_mode': 'notch CAR'
}