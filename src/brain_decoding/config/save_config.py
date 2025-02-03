"""
This script is used to define the basic config parameters for a movie decoding project.
Custom parameters can be added to any of the three fields of config (experiment, model, data).
"""

from torch import nn

from brain_decoding.config.config import ExperimentConfig, PipelineConfig
from brain_decoding.config.file_path import CONFIG_FILE_PATH, DATA_PATH, RESULT_PATH

# if __name__ == "__main__":
experiment_config = ExperimentConfig(name="sleep", patient=562)

config = PipelineConfig(experiment=experiment_config)
config.model.architecture = "multi-vit"
config.model.learning_rate = 1e-4
config.model.batch_size = 128
config.model.weight_decay = 1e-4
config.model.epochs = 40
config.model.lr_drop = 50
config.model.validation_step = 10
config.model.early_stop = 75
config.model.num_labels = 18  # 8 for 24, 18 for twilight
config.model.merge_label = True
config.model.img_embedding_size = 192
config.model.hidden_size = 256
config.model.num_hidden_layers = 6
config.model.num_attention_heads = 8
config.model.patch_size = (1, 5)
config.model.intermediate_size = 192 * 2
config.model.classifier_proj_size = 192
config.model.train_loss = nn.BCEWithLogitsLoss(reduction="none")

config.experiment.seed = 42
config.experiment.use_spike = True
config.experiment.use_lfp = False
config.experiment.use_combined = False
config.experiment.use_shuffle = True
config.experiment.use_bipolar = False
# config.experiment.use_sleep = (
#     True  # set true to use sleep data as inference dataset, otherwise use free recall, is this right?
# )
config.experiment.use_overlap = False
config.experiment.use_long_input = False
config.experiment.use_spontaneous = False
config.experiment.use_augment = False
config.experiment.use_shuffle_diagnostic = True
config.experiment.testing_mode = False  # in testing mode, a maximum of 1e4 clusterless data will be loaded.
config.experiment.model_aggregate_type = "sum"
config.experiment.train_phases = ["twilight_1"]
config.experiment.test_phases = ["sleep_1"]
config.experiment.compute_accuracy = False

config.experiment.ensure_list("train_phases")
config.experiment.ensure_list("test_phases")

config.data.result_path = str(RESULT_PATH)
config.data.spike_path = str(DATA_PATH)
config.data.lfp_path = None
config.data.lfp_data_mode = "sf2000-bipolar-region-clean"
config.data.spike_data_mode = "notch"
config.data.spike_data_mode_inference = "notch"
config.data.spike_data_sd = [3.5]
config.data.spike_data_sd_inference = 3.5
config.data.model_aggregate_type = "sum"
config.data.movie_label_path = str(DATA_PATH / "8concepts_merged.npy")
config.data.movie_label_sr = 1
config.data.movie_sampling_rate = 30
config.data.filter_low_occurrence_samples = True

# TO DO: fix pydantic export json error.
# config.export_config(CONFIG_FILE_PATH)
