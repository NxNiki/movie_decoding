import numpy as np 
import os


param_dict={
    'lr': 1e-4,
    'batch_size': 128,
    'weight_decay': 0,
    'epochs': 100,
    'lr_drop': 50,
    'validation_step': 100,
    'num_labels': 12,
    'conv_dim': 32,
    'layer_norm_eps': 1e-5,
    'feat_proj_dropout': 0.0,
    'layerdrop': 0.0,
    'activation_dropout': 0.1,
    'attention_dropout': 0.1,
    'hidden_dropout': 0.1,
    'mask_time_prob': 0.0, 
    'mask_feature_prob': 0.0,
    'add_adapter': False,
    'use_weighted_layer_sum': False,
    'num_feat_extract_layers': 1,
    'num_conv_pos_embeddings': 32, 
    'num_conv_pos_embedding_groups': 16,
    'hidden_size': 32,
    'num_hidden_layers': 3,
    'num_attention_heads': 4,
    'intermediate_size': 128,
    'classifier_proj_size': 32,
    # path
    'movie_label_path': '/mnt/SSD2/yyding/Datasets/12concepts/12concepts_merged_more.npy',
    # 'movie_label_path': '/mnt/SSD5/yyding/Datasets/12concepts/12concepts.npy',
    'spike_path': '/mnt/SSD2/yyding/Datasets/neuron/spike_data',
    'lfp_path': '/mnt/SSD2/yyding/Datasets/neuron/lfp_data',
}