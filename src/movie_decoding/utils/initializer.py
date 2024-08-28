from transformers import ViTConfig, Wav2Vec2Config

from movie_decoding.dataloader.free_recall import InferenceDataset, create_inference_combined_loaders
from movie_decoding.dataloader.movie import *
from movie_decoding.models.ensemble import *
from movie_decoding.models.multichannel_encoder_vit import MultiEncoder as MultiEncoderViT
from movie_decoding.models.multichannel_encoder_vit_sum import MultiEncoder as MultiEncoderViTSum
from movie_decoding.models.multichannel_encoder_wav2vec import MultiEncoder as MultiEncoderWav2Vec2
from movie_decoding.models.tttt import MultiCCT as MultiViTCCT
from movie_decoding.models.tttt2 import CCT

# from movie_decoding.models.vit_huggingface_3choose1 import ViTForImageClassification
# from movie_decoding.models.vit_huggingface_3in1 import ViTForImageClassification
from movie_decoding.models.vit_huggingface import ViTForImageClassification
from movie_decoding.models.wav2vec_huggingface import Wav2Vec2ForSequenceClassification
from movie_decoding.param import param_crossvit, param_vit, param_vit_cct, param_wav2vec, param_wav2vec2
from movie_decoding.param.param_data import *

# from models.vit_huggingface_archive import ViTForImageClassification
from movie_decoding.utils.evaluator import Evaluator


def initialize_configs(architecture=""):
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


def initialize_inference_dataloaders(config):
    if config["use_sleep"]:
        dataset = InferenceDataset(
            config["data_path"],
            config["patient"],
            config["use_lfp"],
            config["use_spike"],
            config["use_bipolar"],
            config["use_sleep"],
            config["free_recall_phase"],
            config["hour"],
        )
    else:
        dataset = InferenceDataset(config)

    LFP_CHANNEL[config["patient"]] = dataset.lfp_channel_by_region
    test_loader = create_inference_combined_loaders(dataset, config, batch_size=config["batch_size"])

    dataloaders = {"train": None, "valid": None, "inference": test_loader}
    return dataloaders


def initialize_dataloaders(config):
    transform = False
    dataset = NeuronDataset(config)
    LFP_CHANNEL[config["patient"]] = dataset.lfp_channel_by_region
    train_loader, val_loader, test_loader = create_weighted_loaders(
        dataset,
        config,
        batch_size=config["batch_size"],
        shuffle=True,
        p_val=0,
        transform=transform,
    )

    dataloaders = {"train": train_loader, "valid": val_loader, "inference": test_loader}
    return dataloaders


def initialize_evaluator(config, fold):
    evaluator = Evaluator(config, fold)
    return evaluator


def initialize_model(config):
    lfp_model = None
    spike_model = None

    if config["use_combined"]:
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

    if config["use_lfp"]:
        if config["model_architecture"] == "vit":
            image_height = LFP_CHANNEL[config["patient"]]
            image_width = LFP_FRAME[config["patient"]]
            cfg = {
                "img_embedding_size": config["img_embedding_size"],
                "hidden_size": config["hidden_size"],
                "num_hidden_layers": config["num_hidden_layers"],
                "num_attention_heads": config["num_attention_heads"],
                "intermediate_size": config["intermediate_size"],
                "image_height": image_height,
                "image_width": image_width,
                "patch_size": (image_height, 25),
                "num_labels": config["num_labels"],
                "num_channels": 1,
                "return_dict": True,
            }
            configuration = ViTConfig(**cfg)
            lfp_model = ViTForImageClassification(configuration)
        elif config["model_architecture"] == "multi-vit":
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
        elif config["model_architecture"] == "wav2vec2":
            cfg = {
                "hidden_size": config["hidden_size"],
                "intermediate_size": config["intermediate_size"],
                "classifier_proj_size": config["hidden_size"],
                "num_feat_extract_layers": 3,
                "conv_dim": [128, 128, 128],
                "conv_kernel": [10, 3, 3],
                "conv_stride": [5, 2, 2],
                "num_hidden_layers": 3,
                "num_attention_heads": 4,
                "num_channels": LFP_CHANNEL[config["patient"]],
                "num_frames": LFP_FRAME[config["patient"]],
                "num_labels": config["num_labels"],
                "return_dict": True,
            }
            configuration = Wav2Vec2Config(**cfg)
            lfp_model = Wav2Vec2ForSequenceClassification(configuration)
        elif config["model_architecture"] == "multi-wav2vec2":
            image_height = LFP_CHANNEL[config["patient"]]
            image_height = list(image_height.values())
            image_width = LFP_FRAME[config["patient"]]
            cfg = {
                "hidden_size": config["hidden_size"],
                "intermediate_size": config["intermediate_size"],
                "classifier_proj_size": config["hidden_size"],
                "num_feat_extract_layers": 3,
                "conv_dim": [128, 128, 128],
                "conv_kernel": [10, 3, 3],
                "conv_stride": [5, 2, 2],
                "num_hidden_layers": config["num_hidden_layers"],
                "num_attention_heads": config["num_attention_heads"],
                "input_channels": len(image_height),
                "num_channels": image_height,
                "num_frames": image_width,
                "num_labels": config["num_labels"],
                "return_dict": True,
            }
            configuration = Wav2Vec2Config(**cfg)
            lfp_model = MultiEncoderWav2Vec2(configuration)
        else:
            raise ValueError(f"Model Architecture {config['model_architecture']} not supported")

    if config["use_spike"]:
        # config.num_neuron = SPIKE_CHANNEL[config.patient]
        # config.num_frame = SPIKE_FRAME[config.patient]
        # config.return_hidden = True
        # spike_model = Wav2VecForSequenceClassification(config)
        if config["model_architecture"] == "multi-vit":
            image_height = SPIKE_CHANNEL[config["patient"]]
            image_width = SPIKE_FRAME[config["patient"]]

            if config["use_overlap"] or config["use_long_input"]:
                image_width = image_width * 2
            cfg = {
                "hidden_size": config["hidden_size"],
                "num_hidden_layers": config["num_hidden_layers"],
                "num_attention_heads": config["num_attention_heads"],
                # "intermediate_size": config['intermediate_size'],
                "image_height": 8,  # image_height,
                "image_width": image_width,
                "patch_size": config["patch_size"],  # (1, 5),  # (height ratio, width)
                "input_channels": image_height // 8,
                "num_labels": config["num_labels"],
                "num_channels": 1,
                "return_dict": True,
            }
            configuration = ViTConfig(**cfg)
            # spike_model = ViTForImageClassification(configuration)
            if config["model_aggregate_type"] == "sum":
                spike_model = MultiEncoderViTSum(configuration)
            elif config["model_aggregate_type"] == "mean":
                spike_model = MultiEncoderViT(configuration)
            else:
                raise NotImplementedError(f"model aggregate type is not implemented")
        elif config["model_architecture"] == "multi-vit-cct":
            image_height = SPIKE_CHANNEL[config["patient"]]
            image_width = SPIKE_FRAME[config["patient"]]
            spike_model = CCT(
                img_size=(image_height, image_width),
                n_input_channels=2,
                embedding_dim=config["hidden_size"],
                n_conv_layers=1,
                kernel_size=3,
                stride=2,
                padding=3,
                pooling_kernel_size=3,
                pooling_stride=2,
                pooling_padding=1,
                num_layers=config["num_hidden_layers"],
                num_heads=config["num_attention_heads"],
                mlp_ratio=2.0,
                num_classes=config["num_labels"],
                positional_embedding="sine",  # ['sine', 'learnable', 'none']
            )

        elif config["model_architecture"] == "wav2vec2":
            cfg = {
                "hidden_size": config["hidden_size"],
                "intermediate_size": config["intermediate_size"],
                "classifier_proj_size": config["hidden_size"],
                "num_feat_extract_layers": 2,
                "conv_dim": [128, 128],
                "conv_kernel": [3, 3],
                "conv_stride": [2, 2],
                "num_hidden_layers": 3,
                "num_attention_heads": 4,
                "num_channels": SPIKE_CHANNEL[config["patient"]],
                "num_frames": SPIKE_FRAME[config["patient"]],
                "num_labels": config["num_labels"],
                "return_dict": True,
            }
            configuration = Wav2Vec2Config(**cfg)
            spike_model = Wav2Vec2ForSequenceClassification(configuration)
        elif config["model_architecture"] == "multi-wav2vec2":
            image_height = SPIKE_CHANNEL[config["patient"]]
            image_width = SPIKE_FRAME[config["patient"]]
            cfg = {
                "hidden_size": config["hidden_size"],
                "intermediate_size": config["intermediate_size"],
                "classifier_proj_size": config["hidden_size"],
                "num_feat_extract_layers": 2,
                "conv_dim": [128, 128],
                "conv_kernel": [3, 3],
                "conv_stride": [2, 2],
                "num_hidden_layers": config["num_hidden_layers"],
                "num_attention_heads": config["num_attention_heads"],
                "input_channels": image_height // 8,
                "num_channels": 8,
                "num_frames": image_width,
                "num_labels": config["num_labels"],
                "return_dict": True,
            }
            configuration = Wav2Vec2Config(**cfg)
            spike_model = MultiEncoderWav2Vec2(configuration)
        else:
            raise ValueError(f"Model Architecture {config['model_architecture']} not supported")

    model = Ensemble(lfp_model, spike_model, config)
    return model
