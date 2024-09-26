from pathlib import Path

from movie_decoding.config.config import ExperimentConfig, PipelineConfig

CONFIG_FILE_PATH = Path(__file__).resolve().parents[1] / "config"

if __name__ == "__main__":
    experiment_config = ExperimentConfig(name="MemoryTest", patient=562)

    config = PipelineConfig(experiment=experiment_config)
    config.model.learning_rate = 1e-4
    config.model.batch_size = 128
    config.model.weight_decay = 1e-4
    config.model.epochs = 100
    config.model.lr_drop = 50
    config.model.validation_step = 25
    config.model.num_labels = 8
    config.model.merge_label = True
    config.model.img_embedding_size = 192
    config.model.hidden_size_spike = 192
    config.model.hidden_size_lfp = 192
    config.model.num_hidden_layers = 4
    config.model.num_attention_heads = 6
    config.model.intermediate_size = 192 * 2
    config.model.classifier_proj_size = 192

    config.export_config(CONFIG_FILE_PATH)
