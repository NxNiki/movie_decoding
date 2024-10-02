"""
This script is used to define the basic config parameters for a movie decoding project.
Custom parameters can be added to any of the three fields of config (experiment, model, data).
"""

from movie_decoding.config.config import ExperimentConfig, PipelineConfig
from movie_decoding.config.file_path import CONFIG_FILE_PATH, DATA_PATH, RESULT_PATH

if __name__ == "__main__":
    experiment_config = ExperimentConfig(name="MemoryTest", patient=562)

    config = PipelineConfig(experiment=experiment_config)
    config.model.architecture = "multi-vit"
    config.model.learning_rate = 1e-4
    config.model.batch_size = 128
    config.model.weight_decay = 1e-4
    config.model.epochs = 100
    config.model.lr_drop = 50
    config.model.validation_step = 25
    config.model.early_stop = 75
    config.model.num_labels = 8
    config.model.merge_label = True
    config.model.img_embedding_size = 192
    config.model.hidden_size_spike = 192
    config.model.hidden_size_lfp = 192
    config.model.num_hidden_layers = 4
    config.model.num_attention_heads = 6
    config.model.intermediate_size = 192 * 2
    config.model.classifier_proj_size = 192

    config.experiment.seed = 42
    config.experiment.use_spike = True
    config.experiment.use_lfp = False
    config.experiment.use_combined = False
    config.experiment.use_shuffle = True
    config.experiment.use_bipolar = False
    config.experiment.use_sleep = False
    config.experiment.use_overlap = False
    config.experiment.use_spontaneous = False
    config.experiment.use_augment = False
    config.experiment.use_shuffle_diagnostic = True

    config.data.result_path = RESULT_PATH
    config.data.spike_path = "undefined"
    config.data.lfp_path = "undefined"
    config.data.spike_data_mode = "notch CAR-quant-neg"
    config.data.spike_data_mode_inference = "notch CAR-quant-neg"
    config.data.spike_data_sd = [3.5]
    config.data.spike_data_sd_inference = 3.5
    config.data.use_augment = False
    config.data.use_long_input = False
    config.data.use_shuffle_diagnostic = False
    config.data.model_aggregate_type = "sum"
    config.data.movie_label_path = DATA_PATH / "8concepts_merged.npy"

    config.export_config(CONFIG_FILE_PATH)
