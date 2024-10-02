"""
train the model with data while patients are viewing the movie Twilight and 24 and test the model with data during free
recall. The model will be trained to predict movie (Twilight vs 24, model 1) or character in both movies (model 2).
Finally, we will apply the model (with better performance) on data during sleep to examine how reactivation/replay will
impact free recall before and after sleep.
"""

import string
from pathlib import Path

import wandb
from brain_decoding.main import pipeline
from brain_decoding.utils.initializer import *

patient_list = ["i728", "572", "567", "566", "563", "562"]
sd_list = [4, 4, 3.5, 4, 4, 3.5]
# data_list = ['notch CAR4.5', 'notch CAR3.5', 'notch CAR4.5', 'notch CAR4', 'notch CAR3.5', 'notch CAR3.5']
data_list = [
    "notch CAR-quant-neg",
    "notch CAR-quant-neg",
    "notch CAR-quant-neg",
    "notch CAR-quant-neg",
    "notch CAR-quant-neg",
    "notch CAR-quant-neg",
]
early_stop = [100, 100, 100, 50, 50, 75]
for patient, sd, dd in zip(patient_list, sd_list, data_list):
    print()
    print("start: ", patient)
    for data_type in ["clusterless"]:
        for run in range(5, 6):
            # root_path = os.path.dirname(os.path.abspath(__file__))
            root_path = Path(__file__).parent.parent
            # save the results
            letters = string.ascii_lowercase
            # suffix = ''.join(random.choice(letters) for i in range(3))
            suffix = f"test53_optimalX_CARX_{run}"
            if data_type == "clusterless":
                use_clusterless = True
                use_lfp = False
                use_combined = False
                model_architecture = "multi-vit"  #'multi-vit'
            elif data_type == "lfp":
                use_clusterless = False
                use_lfp = True
                use_combined = False
                model_architecture = "multi-vit"
            elif data_type == "combined":
                use_clusterless = True
                use_lfp = True
                use_combined = True
                model_architecture = "multi-crossvit"
            else:
                ValueError(f"undefined data_type: {data_type}")

            args = initialize_configs(architecture=model_architecture)
            args["seed"] = 42
            args["device"] = "cuda:1"
            args["patient"] = patient
            args["use_spike"] = use_clusterless
            args["use_lfp"] = use_lfp
            args["use_combined"] = use_combined
            args["use_spontaneous"] = False
            if use_clusterless:
                args["use_shuffle"] = True
            elif use_lfp:
                args["use_shuffle"] = False

            args["use_bipolar"] = False
            args["use_sleep"] = False
            args["use_overlap"] = False
            args["model_architecture"] = model_architecture

            args["spike_data_mode"] = dd
            args["spike_data_mode_inference"] = dd
            args["spike_data_sd"] = [sd]
            args["spike_data_sd_inference"] = sd
            args["use_augment"] = False
            args["use_long_input"] = False
            args["use_shuffle_diagnostic"] = False
            args["model_aggregate_type"] = "sum"

            train_save_path = os.path.join(
                root_path,
                "results/8concepts/{}_{}_{}_{}/train".format(args["patient"], data_type, model_architecture, suffix),
            )
            valid_save_path = os.path.join(
                root_path,
                "results/8concepts/{}_{}_{}_{}/valid".format(args["patient"], data_type, model_architecture, suffix),
            )
            test_save_path = os.path.join(
                root_path,
                "results/8concepts/{}_{}_{}_{}/test".format(args["patient"], data_type, model_architecture, suffix),
            )
            memory_save_path = os.path.join(
                root_path,
                "results/8concepts/{}_{}_{}_{}/memory".format(args["patient"], data_type, model_architecture, suffix),
            )
            os.makedirs(train_save_path, exist_ok=True)
            os.makedirs(valid_save_path, exist_ok=True)
            os.makedirs(test_save_path, exist_ok=True)
            os.makedirs(memory_save_path, exist_ok=True)
            args["train_save_path"] = train_save_path
            args["valid_save_path"] = valid_save_path
            args["test_save_path"] = test_save_path
            args["memory_save_path"] = memory_save_path

            os.environ["WANDB_MODE"] = "offline"
            # os.environ['WANDB_API_KEY'] = '5a6051ed615a193c44eb9f655b81703925460851'
            wandb.login()
            if use_lfp:
                run_name = "LFP Concept level {} MultiEncoder".format(args["patient"])
            else:
                run_name = "Clusterless Concept level {} MultiEncoder".format(args["patient"])
            wandb.init(project="24_Concepts", name=run_name, reinit=True, entity="24")

            trainer = pipeline(args)

            print("Start training")
            # start_time = time.time()

            trainer.train(args["epochs"], 1)
    print("done: ", patient)
    print()
