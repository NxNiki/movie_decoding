from typing import Dict

from transformers import ViTConfig, Wav2Vec2Config

from brain_decoding.config.config import DataConfig, PipelineConfig
from brain_decoding.dataloader.free_recall import InferenceDataset, create_inference_combined_loaders
from brain_decoding.dataloader.movie import NeuronDataset, create_weighted_loaders
from brain_decoding.models.ensemble import Ensemble
from brain_decoding.models.multichannel_encoder_vit import MultiEncoder as MultiEncoderViT
from brain_decoding.models.multichannel_encoder_vit_sum import MultiEncoder as MultiEncoderViTSum
from brain_decoding.models.multichannel_encoder_wav2vec import MultiEncoder as MultiEncoderWav2Vec2

# from brain_decoding.models.tttt import MultiCCT as MultiViTCCT
from brain_decoding.models.tttt2 import CCT

# from brain_decoding.models.vit_huggingface_3choose1 import ViTForImageClassification
# from brain_decoding.models.vit_huggingface_3in1 import ViTForImageClassification
from brain_decoding.models.vit_huggingface import ViTForImageClassification
from brain_decoding.models.wav2vec_huggingface import Wav2Vec2ForSequenceClassification
from brain_decoding.param import param_crossvit, param_vit, param_vit_cct, param_wav2vec, param_wav2vec2
from brain_decoding.param.param_data import LFP_CHANNEL, LFP_FRAME, SPIKE_CHANNEL, SPIKE_FRAME

# from models.vit_huggingface_archive import ViTForImageClassification
from brain_decoding.utils.evaluator import Evaluator


def set_architecture(config: DataConfig) -> str:
    architecture = ""
    if config.data_type == "clusterless":
        architecture = "multi-vit"  # 'multi-vit'
    elif config.data_type == "lfp":
        architecture = "multi-vit"
    elif config.data_type == "combined":
        architecture = "multi-crossvit"
    else:
        ValueError(f"undefined data_type: {config.data_type}")

    return architecture


def initialize_configs(architecture) -> Dict:
    if architecture == "vit":
        args = param_vit.param_dict
    elif architecture == "multi-vit":
        args = param_vit.param_dict
    elif architecture == "multi-vit-cct":
        args = param_vit_cct.param_dict
    elif architecture == "multi-crossvit":
        args = param_crossvit.param_dict
    elif architecture == "wav2vec":
        args = param_wav2vec.param_dict
    elif architecture == "wav2vec2":
        args = param_wav2vec2.param_dict
    elif architecture == "multi-wav2vec2":
        args = param_wav2vec2.param_dict
    else:
        raise NotImplementedError(f"{architecture} is not implemented")
    return args


def initialize_inference_dataloaders(config: PipelineConfig):
    if config.experiment["use_sleep"]:
        dataset = InferenceDataset(
            config.data["data_path"],
            config.experiment["patient"],
            config.experiment["use_lfp"],
            config.experiment["use_spike"],
            config.experiment["use_bipolar"],
            config.experiment["use_sleep"],
            config.experiment["free_recall_phase"],
            config.experiment["hour"],
        )
    else:
        dataset = InferenceDataset(config)

    LFP_CHANNEL[config["patient"]] = dataset.lfp_channel_by_region
    test_loader = create_inference_combined_loaders(dataset, config, batch_size=config["batch_size"])

    dataloaders = {"train": None, "valid": None, "inference": test_loader}
    return dataloaders


def initialize_dataloaders(config: PipelineConfig):
    transform = False
    dataset = NeuronDataset(config)
    LFP_CHANNEL[str(config.experiment["patient"])] = dataset.lfp_channel_by_region
    train_loader, val_loader, test_loader = create_weighted_loaders(
        dataset,
        config,
        batch_size=config.model["batch_size"],
        shuffle=True,
        p_val=0,
        transform=transform,
    )

    dataloaders = {"train": train_loader, "valid": val_loader, "inference": test_loader}
    return dataloaders


def initialize_evaluator(config, fold):
    evaluator = Evaluator(config, fold)
    return evaluator


def initialize_model(config: PipelineConfig):
    lfp_model = None
    spike_model = None

    if config.experiment["use_combined"]:
        image_height = LFP_CHANNEL[config["patient"]]
        image_height = list(image_height.values())
        image_width = LFP_FRAME[config["patient"]]
        cfg = {
            "img_embedding_size": config["img_embedding_size"],
            "hidden_size": config["hidden_size_lfp"],
            "num_hidden_layers": config["num_hidden_layers"],
            "num_attention_heads": config["num_attention_heads"],
            "intermediate_size": config["intermediate_size"],
            "image_height": image_height,
            "image_width": image_width,
            "patch_size": (1, 25),  # (height ratio, width)
            "input_channels": len(image_height),
            "num_labels": config["num_labels"],
            "num_channels": 1,
            "return_dict": True,
        }
        configuration_lfp = ViTConfig(**cfg)

        image_height = SPIKE_CHANNEL[config["patient"]]
        image_width = SPIKE_FRAME[config["patient"]]
        cfg = {
            "img_embedding_size": config["img_embedding_size"],
            "hidden_size": config["hidden_size_spike"],
            "num_hidden_layers": config["num_hidden_layers"],
            "num_attention_heads": config["num_attention_heads"],
            "intermediate_size": config["intermediate_size"],
            "image_height": 8,  # image_height,
            "image_width": image_width,
            "patch_size": (1, 5),  # (height ratio, width)
            "input_channels": image_height // 8,
            "num_labels": config["num_labels"],
            "num_channels": 1,
            "return_dict": True,
        }
        configuration_spike = ViTConfig(**cfg)

        # branch_model = MultiEncoderCrossViT(configuration_spike, configuration_lfp)
        branch_model = None
        spike_model = MultiEncoderViT(configuration_spike)
        lfp_model = MultiEncoderViT(configuration_lfp)
        model = Ensemble(lfp_model, spike_model, config, branch_model=branch_model)
        return model

    if config.experiment["use_lfp"]:
        image_height = LFP_CHANNEL[config["patient"]]
        image_height = list(image_height.values())
        image_width = LFP_FRAME[config["patient"]]
        cfg = {
            "hidden_size": config["hidden_size"],
            "num_hidden_layers": config["num_hidden_layers"],
            "num_attention_heads": config["num_attention_heads"],
            # "intermediate_size": config['intermediate_size'],
            "image_height": image_height,
            "image_width": image_width,
            "patch_size": (1, 25),  # (height ratio, width)
            "input_channels": len(image_height),
            "num_labels": config["num_labels"],
            "num_channels": 1,
            "return_dict": True,
        }
        configuration = ViTConfig(**cfg)
        lfp_model = MultiEncoderViT(configuration)

    if config.experiment["use_spike"]:
        # config.num_neuron = SPIKE_CHANNEL[config.patient]
        # config.num_frame = SPIKE_FRAME[config.patient]
        # config.return_hidden = True
        # spike_model = Wav2VecForSequenceClassification(config)
        image_height = SPIKE_CHANNEL[str(config.experiment["patient"])]
        image_width = SPIKE_FRAME[str(config.experiment["patient"])]

        if config.experiment["use_overlap"] or config.experiment["use_long_input"]:
            image_width = image_width * 2
        cfg = {
            "hidden_size": config.model["hidden_size"],
            "num_hidden_layers": config.model["num_hidden_layers"],
            "num_attention_heads": config.model["num_attention_heads"],
            # "intermediate_size": config['intermediate_size'],
            "image_height": 8,  # image_height,
            "image_width": image_width,
            "patch_size": config.model["patch_size"],  # (1, 5),  # (height ratio, width)
            "input_channels": image_height // 8,
            "num_labels": config.model["num_labels"],
            "num_channels": 1,
            "return_dict": True,
        }
        configuration = ViTConfig(**cfg)
        # spike_model = ViTForImageClassification(configuration)
        if config.experiment["model_aggregate_type"] == "sum":
            spike_model = MultiEncoderViTSum(configuration)
        elif config.experiment["model_aggregate_type"] == "mean":
            spike_model = MultiEncoderViT(configuration)
        else:
            raise NotImplementedError(f"model aggregate type is not implemented")

    model = Ensemble(lfp_model, spike_model, config)
    return model
