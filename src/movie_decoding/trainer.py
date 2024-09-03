import copy
import json
import os
import random
import time
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ensemble import Ensemble
from ray import train
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from torch import Tensor
from tqdm import tqdm
from utils.initializer import initialize_inference_dataloaders, initialize_model
from utils.meters import Meter, TestMeter, ValidMeter
from utils.permutation import Permutate

import wandb
from movie_decoding.param.base_param import device_name


class Trainer:
    def __init__(
        self,
        model: Ensemble,
        evaluator,
        optimizers,
        lr_scheduler,
        data_loaders,
        config,
    ):
        super().__init__()

        self.model = model
        self.evaluator = evaluator
        self.optimizer = optimizers
        self.lr_scheduler = lr_scheduler
        self.train_loader = data_loaders["train"]
        self.valid_loader = data_loaders["valid"]
        self.inference_loader = data_loaders["inference"]
        self.device = device_name
        self.config = config

        pos_weight_train = torch.tensor(self.train_loader.dataset.pos_weight, dtype=torch.float, device=self.device)
        # pos_weight_val = torch.tensor(self.valid_loader.dataset.pos_weight, dtype=torch.float, device=self.device)
        # self.bce_loss = nn.BCELoss(reduction="none")
        # self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_train, reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.mse_loss = nn.MSELoss(reduction="none")
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

    def extract_feature(self, feature: Union[Tensor, List[Tensor], Tuple[Tensor]]) -> Tuple[Tensor, Tensor]:
        if not self.config["use_lfp"] and self.config["use_spike"]:
            spike = feature.to(self.device)
            lfp = None
        elif self.config["use_lfp"] and not self.config["use_spike"]:
            lfp = feature.to(self.device)
            spike = None
        else:
            assert isinstance(feature, list) or isinstance(feature, tuple), "Tensor must be a list or tuple"
            spike = feature[1].to(self.device)
            lfp = feature[0].to(self.device)

        return spike, lfp

    def train(self, epochs, fold):
        best_f1 = -1
        self.model.train()
        for epoch in tqdm(range(epochs)):
            meter = Meter(fold)

            frame_index = np.empty(0)
            y_pred = np.empty((0, self.config["num_labels"]))
            y_true = np.empty((0, self.config["num_labels"]))

            for i, (feature, target, index) in enumerate(self.train_loader):
                target = target.to(self.device)
                spike, lfp = self.extract_feature(feature)
                # forward pass
                spike_emb, lfp_emb, output = self.model(lfp, spike)
                # mse_loss = self.mse_loss(output, target)
                mse_loss = self.bce_loss(output, target)
                # weight_mask = torch.where(target > 0.5, torch.tensor(1.5).to(self.device), torch.tensor(0.5).to(self.device))
                # mse_loss = torch.mean(mse_loss * weight_mask)
                mse_loss = torch.mean(mse_loss)
                # if self.config['use_lfp'] and self.config['use_spike']:
                #     kl_loss = self.kl_loss(F.log_softmax(spike_emb, dim=1), F.log_softmax(lfp_emb, dim=1))
                # else:
                #     kl_loss = 0
                loss = mse_loss
                # loss[mask] *= self.train_loader.label_weights
                # loss = torch.mean(loss)
                # backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # prediction
                output = torch.sigmoid(output)
                pred = np.round(output.cpu().detach().numpy())
                true = np.round(target.cpu().detach().numpy())
                y_pred = np.concatenate([y_pred, pred], axis=0)
                y_true = np.concatenate([y_true, true], axis=0)
                accuracy = self.evaluator.calculate_accuracy(true, pred)
                f1 = self.evaluator.calculate_f1(true, pred)
                frame_index = np.concatenate([frame_index, index], axis=0)
                meter.add(loss.item(), f1, accuracy)
            # np.save('train_indices.npy', frame_index)
            log_info = meter.dump_wandb()
            self.lr_scheduler.step()

            if (epoch + 1) % self.config["validation_step"] == 0:
                # stats = self.validation(fold)
                # log_info.update(stats)
                model_save_path = os.path.join(
                    self.config["train_save_path"],
                    "model_weights_epoch{}.tar".format(epoch + 1),
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "args": self.config,
                    },
                    model_save_path,
                )

                model_save_path = os.path.join(
                    self.config["train_save_path"],
                    "best_weights_fold{}.tar".format(fold + 1),
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "args": self.config,
                    },
                    model_save_path,
                )
                print()
                print("WELCOME MEMORY TEST at: ", epoch)
                stats_m = self.memory(epoch=epoch + 1, phase="free_recall1", alongwith=[])
                # self.memory(1, epoch=epoch+1, phase='all')
                overall_p = list(stats_m.values())
                print("P: ", overall_p)
                overall_significant = len([x for x in overall_p if not np.isnan(x) and 0 <= x < 0.1])
                overall_valid = len([x for x in overall_p if not np.isnan(x)])
                jack_p = stats_m["Jack"]
                print(
                    "Jack p: ",
                    jack_p,
                    "overall: ",
                    f"{overall_significant} / {overall_valid}",
                )
                # train.report(
                #     {"recall": overall_significant, "jack": jack_p},
                # )
            wandb.log(log_info)

    def validation(self, fold):
        self.model.eval()

        meter = ValidMeter(fold)
        with torch.no_grad():
            y_pred = np.empty((0, self.config["num_labels"]))
            y_true = np.empty((0, self.config["num_labels"]))
            y_score = np.empty((0, self.config["num_labels"]))
            frame_index = np.empty(0)
            for i, (feature, target, index) in enumerate(self.valid_loader):
                target = target.to(self.device)
                spike, lfp = self.extract_feature(feature)
                # forward pass
                spike_emb, lfp_emb, output = self.model(lfp, spike)
                # mse_loss = self.mse_loss(output, target)
                mse_loss = self.bce_loss(output, target)
                # weight_mask = torch.where(target > 0.5, torch.tensor(1.5).to(self.device), torch.tensor(0.5).to(self.device))
                # mse_loss = torch.mean(mse_loss * weight_mask)
                mse_loss = torch.mean(mse_loss)
                # if self.config['use_lfp'] and self.config['use_spike']:
                #     kl_loss = self.kl_loss(F.log_softmax(spike_emb, dim=1), F.log_softmax(lfp_emb, dim=1))
                # else:
                #     kl_loss = 0
                loss = mse_loss

                # prediction
                output = torch.sigmoid(output)
                pred = np.round(output.cpu().detach().numpy())
                true = np.round(target.cpu().detach().numpy())
                y_pred = np.concatenate([y_pred, pred], axis=0)
                y_true = np.concatenate([y_true, true], axis=0)
                frame_index = np.concatenate([frame_index, index], axis=0)

                avg_score = output.cpu().detach().numpy()
                y_score = np.concatenate([y_score, avg_score], axis=0)

                accuracy = self.evaluator.calculate_accuracy(true, pred)
                f1 = self.evaluator.calculate_f1(true, pred)

                meter.add(loss.item(), f1, accuracy)
        # print(f'AVG Activation Score is: {np.mean(y_score, axis=0)}')
        # np.save('valid_indices.npy', frame_index)
        log_info = meter.dump_wandb()
        for label in range(self.config["num_labels"]):
            stats = self.evaluator.evaluate_metrics(y_true[:, label], y_pred[:, label], frame_index, label=label)
            log_info.update(stats)

        f1s = f1_score(y_true, y_pred, zero_division=np.nan, average=None)
        print("F1: ", list(f1s))
        return log_info

    def inference(self, fold):
        model_saved_path = os.path.join(self.config["train_save_path"], "best_weights_fold{}.tar".format(fold + 1))
        self.model.load_state_dict(torch.load(model_saved_path)["model_state_dict"])
        self.model.eval()

        meter = TestMeter(fold)
        with torch.no_grad():
            y_pred = np.empty((0, self.config["num_labels"]))
            y_score = np.empty((0, self.config["num_labels"]))
            y_true = np.empty((0, self.config["num_labels"]))
            frame_index = np.empty((0))
            for i, (feature, target, index) in enumerate(self.valid_loader):
                target = target.to(self.device)
                spike, lfp = self.extract_feature(feature)
                # forward pass
                spike_emb, lfp_emb, output = self.model(lfp, spike)
                # mse_loss = self.mse_loss(output, target)
                mse_loss = self.bce_loss(output, target)
                weight_mask = torch.where(
                    target > 0.5,
                    torch.tensor(1.5).to(self.device),
                    torch.tensor(0.5).to(self.device),
                )
                mse_loss = torch.mean(mse_loss * weight_mask)
                # mse_loss = torch.mean(mse_loss)

                if self.config["use_lfp"] and self.config["use_spike"]:
                    kl_loss = self.kl_loss(F.log_softmax(spike_emb, dim=1), F.log_softmax(lfp_emb, dim=1))
                else:
                    kl_loss = 0
                loss = mse_loss + kl_loss

                # prediction
                output = torch.sigmoid(output)
                pred = np.round(output.cpu().detach().numpy())
                score = output.cpu().detach().numpy()
                true = np.round(target.cpu().detach().numpy())
                y_pred = np.concatenate([y_pred, pred], axis=0)
                y_score = np.concatenate([y_score, score], axis=0)
                y_true = np.concatenate([y_true, true], axis=0)
                frame_index = np.concatenate([frame_index, index], axis=0)
                accuracy = self.evaluator.calculate_accuracy(true, pred)
                f1 = self.evaluator.calculate_f1(true, pred)
                meter.add(loss.item(), f1, accuracy)

        self.evaluator.prediction_visualization(y_true, y_pred, frame_index.astype(int))
        for i in range(self.config["num_labels"]):
            try:
                self.evaluator.roc_analysis(y_true[:, i], y_score[:, i], label=i)
            except:
                print(
                    self.evaluator.classes[i],
                    ": does not appear in the testing dataset",
                )
            stats = self.evaluator.evaluate_metrics(y_true[:, i], y_pred[:, i], frame_index, label=i)

        log_info = meter.dump_wandb()
        print(log_info)
        result_save_path = os.path.join(self.config["test_save_path"], "results_fold{}.txt".format(fold + 1))
        with open(result_save_path, "w") as file:
            file.write(json.dumps(log_info))

        activation_score_save_path = os.path.join(
            self.config["test_save_path"], "activation_score_fold{}".format(fold + 1)
        )
        np.save(activation_score_save_path, y_score)
        label_save_path = os.path.join(self.config["test_save_path"], "label_fold{}".format(fold + 1))
        np.save(label_save_path, y_true)

    def permutation(self, fold):
        def permutation_p(label, activation):
            permutations = 500
            concept_ps = []  # empirical p-value for actual concept is greater than permuted samples
            for i, concept in enumerate(self.evaluator.classes):
                concept_indices = np.where(label[:, i] == 1)[0]
                if len(concept_indices) > 0:
                    target_activations = activation[:, i][concept_indices]
                    mask = np.ones(len(activation), dtype=bool)
                    mask[concept_indices] = False
                    p_values = 0
                    sig_counter = []
                    avg_activations = np.mean(target_activations)
                    for perm in range(permutations):
                        # get random indices of same length as temp_activations
                        sampled_activations = np.random.choice(
                            activation[:, i][mask],
                            size=len(target_activations),
                            replace=False,
                        )
                        if avg_activations > np.mean(sampled_activations):
                            sig_counter.append(1)
                        else:
                            sig_counter.append(0)

                    # concept_Ps.append(p_values / permutations)
                    concept_ps.append(1 - sum(sig_counter) / permutations)
                else:
                    concept_ps.append(np.nan)
            concept_ps = np.round(concept_ps, 5)
            return concept_ps

        statistic_dict = {
            "model": [],
            "null model": [],
        }

        label_path = os.path.join(self.config["test_save_path"], "label_fold{}.npy".format(fold + 1))
        activation_path = os.path.join(
            self.config["test_save_path"],
            "activation_score_fold{}.npy".format(fold + 1),
        )

        test_label = np.load(label_path)
        test_activation = np.load(activation_path)
        # activation = np.round(activation)
        ps = permutation_p(test_label, test_activation)

        for i in range(len(ps)):
            print(self.evaluator.classes[i] + ": " + str(ps[i]))
            statistic_dict["model"].append(ps[i])

        label_path = os.path.join(self.config["test_save_path"], "train_label_fold{}.npy".format(fold + 1))
        label = np.load(label_path)
        class_value, class_count = np.unique(label, axis=0, return_counts=True)
        class_weight_dict = {key.tobytes(): value / label.shape[0] for key, value in zip(class_value, class_count)}
        data_weights = np.array([class_weight_dict[l.tobytes()] for l in label])
        p = data_weights / np.sum(data_weights)
        avg_ps = np.empty((0, 12))
        for i in range(10):
            sampled_indices = np.random.choice(np.arange(label.shape[0]), size=test_activation.shape[0], p=p)
            activation_null = label[sampled_indices]
            ps = permutation_p(test_label, activation_null)
            avg_ps = np.concatenate([avg_ps, ps.reshape(1, 12)], axis=0)
        ps = np.round(np.average(avg_ps, axis=0), 5)
        for i in range(len(ps)):
            print(self.evaluator.classes[i] + ": " + str(ps[i]))
            statistic_dict["null model"].append(ps[i])

        df = pd.DataFrame(statistic_dict)
        df.index = self.evaluator.classes
        df.to_csv(os.path.join(self.config["test_save_path"], "p_values.csv"))

    def memory(self, epoch=-1, phase: str = "FR1", alongwith=[]):
        device = device_name
        torch.manual_seed(self.config["seed"])
        np.random.seed(self.config["seed"])
        random.seed(self.config["seed"])
        self.config["free_recall_phase"] = phase
        if self.config["patient"] == "i728" and "1" in phase:
            self.config["free_recall_phase"] = "FR1a"
            dataloaders = initialize_inference_dataloaders(self.config)
        else:
            dataloaders = initialize_inference_dataloaders(self.config)
        model = initialize_model(self.config)
        # model = torch.compile(model)
        model = model.to(device)

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # print('number of params:', n_parameters)

        # load the model with best F1-score
        # model_dir = os.path.join(self.config['train_save_path'], 'best_weights_fold{}.tar'.format(fold + 1))
        model_dir = os.path.join(self.config["train_save_path"], "model_weights_epoch{}.tar".format(epoch))
        model.load_state_dict(torch.load(model_dir)["model_state_dict"])
        # print('Resume model: %s' % model_dir)
        model.eval()

        predictions_all = np.empty((0, self.config["num_labels"]))
        predictions_length = {}
        with torch.no_grad():
            if self.config["patient"] == "i728" and "1" in phase:
                # load the best epoch number from the saved "model_results" structure
                for ph in ["FR1a", "FR1b"]:
                    predictions = np.empty((0, self.config["num_labels"]))
                    self.config["free_recall_phase"] = ph
                    dataloaders = initialize_inference_dataloaders(self.config)
                    # y_true = np.empty((0, self.config['num_labels']))
                    for i, (feature, index) in enumerate(dataloaders["inference"]):
                        # target = target.to(self.device)
                        spike, lfp = self.extract_feature(feature)
                        # forward pass

                        # start_time = time.time()
                        spike_emb, lfp_emb, output = model(lfp, spike)
                        # end_time = time.time()
                        # print('inference time: ', end_time - start_time)
                        output = torch.sigmoid(output)
                        pred = output.cpu().detach().numpy()
                        predictions = np.concatenate([predictions, pred], axis=0)

                    if self.config["use_overlap"]:
                        fake_activation = np.mean(predictions, axis=0)
                        predictions = np.vstack((fake_activation, predictions, fake_activation))

                    predictions_all = np.concatenate([predictions_all, predictions], axis=0)
                predictions_length[phase] = len(predictions_all)
            else:
                self.config["free_recall_phase"] = phase
                dataloaders = initialize_inference_dataloaders(self.config)
                predictions = np.empty((0, self.config["num_labels"]))
                # y_true = np.empty((0, self.config['num_labels']))
                for i, (feature, index) in enumerate(dataloaders["inference"]):
                    # target = target.to(self.device)
                    spike, lfp = self.extract_feature(feature)
                    # forward pass

                    # start_time = time.time()
                    spike_emb, lfp_emb, output = model(lfp, spike)
                    # end_time = time.time()
                    # print('inference time: ', end_time - start_time)
                    output = torch.sigmoid(output)
                    pred = output.cpu().detach().numpy()
                    predictions = np.concatenate([predictions, pred], axis=0)

                if self.config["use_overlap"]:
                    fake_activation = np.mean(predictions, axis=0)
                    predictions = np.vstack((fake_activation, predictions, fake_activation))

                predictions_length[phase] = len(predictions)
                predictions_all = np.concatenate([predictions_all, predictions], axis=0)

        # np.save(os.path.join(self.config['memory_save_path'], 'free_recall_{}_results.npy'.format(phase)), predictions)
        save_path = os.path.join(self.config["memory_save_path"], "prediction")
        os.makedirs(save_path, exist_ok=True)
        np.save(
            os.path.join(save_path, "epoch{}_free_recall_{}_results.npy".format(epoch, phase)),
            predictions_all,
        )

        for ph in alongwith:
            self.config["free_recall_phase"] = ph
            dataloaders = initialize_inference_dataloaders(self.config)
            with torch.no_grad():
                # load the best epoch number from the saved "model_results" structure
                predictions = np.empty((0, self.config["num_labels"]))
                # y_true = np.empty((0, self.config['num_labels']))
                for i, (feature, index) in enumerate(dataloaders["inference"]):
                    # target = target.to(self.device)
                    spike, lfp = self.extract_feature(feature)
                    # forward pass

                    # start_time = time.time()
                    spike_emb, lfp_emb, output = model(lfp, spike)
                    # end_time = time.time()
                    # print('inference time: ', end_time - start_time)
                    output = torch.sigmoid(output)
                    pred = output.cpu().detach().numpy()
                    predictions = np.concatenate([predictions, pred], axis=0)

                if self.config["use_overlap"]:
                    fake_activation = np.mean(predictions, axis=0)
                    predictions = np.vstack((fake_activation, predictions, fake_activation))

            predictions_length[ph] = len(predictions)
            predictions_all = np.concatenate([predictions_all, predictions], axis=0)

        smoothed_data = np.zeros_like(predictions_all)
        for i in range(predictions_all.shape[1]):  # Loop through each feature
            smoothed_data[:, i] = np.convolve(predictions_all[:, i], np.ones(4) / 4, mode="same")
        predictions = predictions_all

        smoothed_data = smoothed_data[:, 0:8]
        predictions = predictions[:, 0:8]

        # Perform Statistic Method
        sts = Permutate(
            config=self.config,
            phase=phase,
            epoch=epoch,
            alongwith=alongwith,
            phase_length=predictions_length,
        )
        """method John"""
        # print('***** METHOD JOHN *****')
        # sts.method_john1(predictions)

        # """method John2"""
        # print('***** METHOD JOHN 2*****')
        # sts.method_john2(predictions)

        """method Soraya"""
        # print('***** METHOD SORAYA *****')
        stats = sts.method_soraya(smoothed_data)

        """curve shape"""
        sts.method_curve_shape(smoothed_data)

        """method hoteling"""
        # print('***** METHOD HOTEL *****')
        # sts.method_hotel()

        """plot p value curve"""
        # print('***** METHOD 4-6-8 *****')
        sts.method_pvalue_curve(predictions)
        # if epoch == self.config['epochs']:
        #     """plot p value curve"""
        #     sts.method_pvalue_curve(predictions)
        return stats
