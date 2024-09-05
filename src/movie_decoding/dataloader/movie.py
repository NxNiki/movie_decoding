import copy
import glob
import os
import pickle
import random
import re
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from scipy.ndimage import convolve1d
from scipy.stats import zscore
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SubsetRandomSampler, WeightedRandomSampler
from torchvision.transforms import transforms

from movie_decoding.param.param_data import SF


class NeuronDataset:
    def __init__(self, config):
        self.patient = config["patient"]
        self.use_spontaneous = config["use_spontaneous"]
        self.use_spike = config["use_spike"]
        self.use_lfp = config["use_lfp"]
        self.use_overlap = config["use_overlap"]
        self.use_combined = config["use_combined"]
        self.lfp_data_mode = config["lfp_data_mode"]
        self.spike_data_mode = config["spike_data_mode"]
        self.spike_data_sd = config["spike_data_sd"]

        # assume in label/sec
        self.movie_sampling_rate = 30
        self.movie_label_path = config["movie_label_path"]

        self.resolution = 4
        self.lfp_sf = SF  # Hz
        self.ml_label = np.load(self.movie_label_path)
        # self.ml_label = np.append(self.ml_label, np.zeros((1, self.ml_label.shape[1])), axis=0)

        self.ml_label = np.repeat(self.ml_label, self.resolution, axis=1)

        # add face
        # face_df = pd.read_csv('/mnt/SSD2/yyding/Datasets/12concepts/face_detection_results.csv')
        # face = list(face_df['Face Detected'])
        # face = np.array(face).reshape(1, -1)
        # face = face[:, :self.ml_label.shape[1]]
        # self.ml_label = np.concatenate((self.ml_label, face), axis=0)

        self.smoothed_ml_label = np.copy(self.ml_label)  # self.smooth_label()
        self.lfp_data = []
        self.spike_data = []
        self.data = []
        self.label = []
        self.smoothed_label = []
        self.lfp_channel_by_region = {}

        def sort_filename(filename):
            """Extract the numeric part of the filename and use it as the sort key"""
            return [int(x) if x.isdigit() else x for x in re.findall(r"\d+|\D+", filename)]

        if self.patient in ["564", "565"]:
            categories = ["Movie_1", "Movie_2"]
        else:
            categories = ["Movie_1"]

        if self.use_spontaneous:
            categories.append("Control1")
            categories.append("Control2")

        # create spike data
        if self.use_spike:
            sample_size = []
            for c, category in enumerate(categories):
                version = self.spike_data_mode
                for sd in self.spike_data_sd:
                    spike_path = os.path.join(
                        config["spike_path"],
                        self.patient,
                        version,
                        "time_{}".format(category.lower()),
                    )
                    spike_files = glob.glob(os.path.join(spike_path, "*.npz"))
                    spike_files = sorted(spike_files, key=sort_filename)
                    spike_data = self.load_clustless(spike_files, sd)
                    self.spike_data.append(spike_data)
                    sample_size.append(spike_data.shape[0])

            # if self.patient == '564':
            #     min_length = min(arr.shape[0] for arr in self.spike_data)
            #     self.spike_data = [arr[:min_length] for arr in self.spike_data]
            #     self.label = [arr[:min_length] for arr in self.label]
            #     self.smoothed_label = [arr[:min_length] for arr in self.smoothed_label]
            self.spike_data = np.concatenate(self.spike_data, axis=0)
            self.data.append(self.spike_data)

        # create lfp data
        if self.use_lfp:
            sample_size = []
            for c, category in enumerate(categories):
                version = self.lfp_data_mode
                # value = self.lfp_data.setdefault(version, [])
                lfp_path = os.path.join(
                    config["lfp_path"],
                    self.patient,
                    version,
                    "spectrogram_{}".format(category.lower()),
                )
                lfp_files = glob.glob(os.path.join(lfp_path, "*.npz"))
                lfp_files = sorted(lfp_files, key=sort_filename)
                lfp_data = self.load_lfp(lfp_files)
                self.lfp_data.append(lfp_data)
                sample_size.append(lfp_data.shape[0])

            # if self.patient == '564':
            #     min_length = min(arr.shape[0] for arr in self.lfp_data['sf2000'])
            #     self.lfp_data = {key: [arr[:min_length] for arr in value_list] for key, value_list in self.lfp_data.items()}
            #     self.label = [arr[:min_length] for arr in self.label]
            #     self.smoothed_label = [arr[:min_length] for arr in self.smoothed_label]
            # self.lfp_data = np.concatenate(self.lfp_data, axis=0)

            # if self.use_spontaneous:
            #     target_length = 12800 - 9740
            #     num_c1, num_c2 = len(self.label[1]), len(self.label[2])
            #     total_length = num_c1 + num_c2
            #     ratio_c1 = num_c1 / total_length
            #     new_num_c1 = int(target_length * ratio_c1)
            #     new_num_c2 = target_length - new_num_c1

            #     indices_c1 = np.random.choice(num_c1, new_num_c1, replace=False)
            #     indices_c2 = np.random.choice(num_c2, new_num_c2, replace=False)
            #     indices_c1 = np.sort(indices_c1)
            #     indices_c2 = np.sort(indices_c2)

            #     # self.lfp_data[self.lfp_data_mode][1] = self.lfp_data[self.lfp_data_mode][1][indices_c1]
            #     # self.lfp_data[self.lfp_data_mode][2] = self.lfp_data[self.lfp_data_mode][2][indices_c2]
            #     self.lfp_data[1] = self.lfp_data[1][indices_c1]
            #     self.lfp_data[2] = self.lfp_data[2][indices_c2]

            #     self.label[1] = self.label[1][indices_c1]
            #     self.label[2] = self.label[2][indices_c2]

            #     self.smoothed_label[1] = self.smoothed_label[1][indices_c1]
            #     self.smoothed_label[2] = self.smoothed_label[2][indices_c2]

            # self.lfp_data = {key: np.concatenate(value_list, axis=0) for key, value_list in self.lfp_data.items()}
            self.lfp_data = np.concatenate(self.lfp_data, axis=0)
            self.data.append(self.lfp_data)

        # for c, category in enumerate(categories):
        #     size = sample_size[c]
        #     if 'control' in category.lower():
        #         self.label.append(np.zeros((size, len(self.ml_label)), dtype=np.float32))
        #         self.smoothed_label.append(np.zeros((size, len(self.ml_label)), dtype=np.float32))
        #     else:
        #         self.label.append(self.ml_label.transpose().astype(np.float32))
        #         self.smoothed_label.append(self.smoothed_ml_label.transpose().astype(np.float32))

        for c, category in enumerate(self.spike_data_sd):
            size = sample_size[c]
            self.label.append(self.ml_label.transpose().astype(np.float32))
            self.smoothed_label.append(self.smoothed_ml_label.transpose().astype(np.float32))

        self.label = np.concatenate(self.label, axis=0)
        self.smoothed_label = np.concatenate(self.smoothed_label, axis=0)

        if self.use_overlap:
            self.label = self.label[1:-1]
            self.smoothed_label = self.smoothed_label[1:-1]
        # filter low occuracne sampels
        class_value, class_count = np.unique(self.label[:, 0:8], axis=0, return_counts=True)
        occurrence_threshold = 200 * len(self.spike_data_sd)
        good_indices = np.where(class_count >= occurrence_threshold)[0]
        indices_of_good_samples = []
        for index in good_indices:
            label = class_value[index]
            label_indices = np.where((self.label[:, 0:8] == label[None, :]).all(axis=1))[0]
            indices_of_good_samples.extend(label_indices)
        indices_of_good_samples = sorted(indices_of_good_samples)

        self.label = self.label[indices_of_good_samples]
        self.smoothed_label = self.smoothed_label[indices_of_good_samples]
        if self.use_combined:
            self.data = {
                "clusterless": self.data[0][indices_of_good_samples],
                "lfp": self.data[1][indices_of_good_samples],
            }
        else:
            self.data = self.data[0][indices_of_good_samples]
        del self.lfp_data
        del self.spike_data

        print("Neuron Data Loaded")
        self.preprocess_data()
        if config["use_augment"]:
            self.time_backword()
        if config["use_shuffle_diagnostic"]:
            # self.brute_shuffle()
            self.circular_shift()

    def smooth_label(self):
        sigma = 1
        kernel = np.exp(-(np.arange(-1, 2) ** 2) / (2 * sigma**2))
        kernel /= np.sum(kernel)
        # kernel = np.tile(kernel, (12, 1))

        smoothed_label = convolve1d(self.ml_label, kernel, axis=1)
        max_val = np.max(smoothed_label, axis=1)
        # smoothed_label = smoothed_label / max_val[:, np.newaxis]
        return np.round(smoothed_label, 2)

    @staticmethod
    def channel_max(data: np.ndarray[float]) -> Tuple[np.ndarray[float], np.ndarray[float]]:
        """

        :param data:
        :return:
        """
        b, c, h, w = data.shape
        normalized_data = data.transpose(2, 0, 1, 3).reshape(h, -1)
        vmax = np.max(normalized_data, axis=1)
        vmin = np.min(normalized_data, axis=1)
        epsilon = 1e-10
        vmax = np.where(vmax == 0, epsilon, vmax)
        return vmax, vmin

    def load_clustless(self, files, sds):
        if not files:
            raise ValueError("input files are empty!")

        spike = []
        for file in files:
            data = np.load(file)["data"]
            spike.append(data[:, :, None])

        spike = np.concatenate(spike, axis=2)
        # sd1 = spike[:, 0:1, :, :]
        # sd2 = spike[:, 1:2, :, :]
        # sd3 = spike[:, 2:3, :, :]
        # spike = np.maximum(np.maximum(sd1, sd2), sd3)

        # data_max = np.max(spike)
        # for i in range(5):
        #     spike[(spike > i/5 * data_max) & (spike <= (i+1)/5 * data_max)] = i+1
        # spike[spike < self.spike_data_sd] = 0
        # vmax, vmin = self.channel_max(spike)
        # normalized_spike = 2 * (spike - vmin[None, None, :, None]) / (vmax[None, None, :, None] - vmin[None, None, :, None]) - 1
        spike[spike < sds] = 0
        # spike[spike > 500] = 0
        vmax = np.max(spike)
        normalized_spike = spike / vmax
        return normalized_spike
        # outlier = 500
        # spike[np.abs(spike) > outlier] = 0
        # # p_n = positive - negative
        # # non_zero_count = (np.abs(negative) != 0).astype(int) + (positive != 0).astype(int)
        # # p_n = np.divide(np.abs(negative) + positive, non_zero_count, out=np.zeros_like(positive, dtype=np.float32), where=non_zero_count!=0)
        # b, c, h, w = spike.shape
        # normalized_spike = spike.transpose(2, 0, 1, 3).reshape(h, -1)
        # vmax = np.max(normalized_spike, axis=1)
        # epsilon = 1e-10
        # vmax = np.where(vmax == 0, epsilon, vmax)
        # normalized_spike = spike / vmax[None, None, :, None]

        # negative = normalized_spike[:, 0]
        # positive = normalized_spike[:, 1]
        # return np.abs(negative)[:, None]

    # def load_clustless(files):
    #     spike = []
    #     for file in files:
    #         data = np.load(file)['data']
    #         spike.append(data[:, None])

    #     spike = np.concatenate(spike, axis=1)

    #     # outlier = spike.flatten()[spike.flatten() != 0]
    #     # outlier = np.percentile(outlier, 99.99)
    #     outlier = 500
    #     spike[np.abs(spike) > outlier] = 0

    #     b, h, w = spike.shape
    #     normalized_spike = spike.transpose(1, 0, 2).reshape(h, -1)
    #     # normalized_spike = zscore(normalized_spike, axis=1)
    #     normalized_spike = normalized_spike.reshape(h, b, w).transpose(1, 0, 2)

    #     vmin = np.min(normalized_spike)
    #     vmax = np.max(normalized_spike)
    #     normalized_spike = (normalized_spike - vmin) / (vmax - vmin)
    #     return normalized_spike[:, None]

    def load_lfp(self, files):
        lfp = []
        for file in files:
            data = np.load(file)["data"]

            """
            filter out noisy channel
            """
            noisy_channel = {"FUS": [0, 3], "PARS": [6], "PHC": [1]}
            soz = ["LTC", "HPC"]
            region = file.split("marco_lfp_spectrum_")[-1].split(".npz")[0]
            if region in soz:
                continue
            if region in noisy_channel:
                mask = np.ones(data.shape[1], dtype=bool)
                num = len(mask) // 8
                ignore_list = []
                for i in noisy_channel[region]:
                    tmp = list(np.arange(i, len(mask), num))
                    ignore_list += tmp
                mask[ignore_list] = False
                data = data[:, mask, :]
            """
            filter out noisy channel
            """

            self.lfp_channel_by_region[file.split("marco_lfp_spectrum_")[-1].split(".npz")[0]] = data.shape[1]
            if len(data.shape) == 2:
                lfp.append(data[:, None, :])
            else:
                lfp.append(data)

        lfp = np.concatenate(lfp, axis=1)
        vmin = np.min(lfp)
        vmax = np.max(lfp)
        normalized_lfp = (lfp - vmin) / (vmax - vmin)
        return normalized_lfp[:, None]

    def preprocess_data(self):
        pass

    def time_backword(self):
        fliped = np.flip(self.data, axis=-1)
        self.data = np.concatenate((self.data, fliped), axis=0)
        self.label = np.repeat(self.label, 2, axis=0)
        self.smoothed_label = np.repeat(self.smoothed_label, 2, axis=0)

    def brute_shuffle(self):
        b, c, h, w = self.data.shape
        data = self.data.transpose(2, 0, 1, 3).reshape(h, -1)
        shuffled_data = np.apply_along_axis(np.random.permutation, axis=1, arr=data)
        self.data = shuffled_data.reshape(h, b, c, w).transpose(1, 2, 0, 3)

    def circular_shift(self):
        b, c, h, w = self.data.shape
        shift_amount = np.random.randint(100, b - 100)
        self.data = np.roll(self.data, shift=shift_amount, axis=0)

    def __len__(self):
        return len(self.label)


class MyDataset(Dataset):
    def __init__(self, lfp_data, spike_data, label, indices, transform=None, pos_weight=None):
        self.lfp_data = None
        self.spike_data = None
        if lfp_data is not None:
            self.lfp_data = lfp_data
        if spike_data is not None:
            self.spike_data = spike_data
        self.label = label
        self.transform = transform
        self.pos_weight = pos_weight
        self.indices = indices

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        label = self.label[index]
        idx = self.indices[index]
        if self.lfp_data is not None and self.spike_data is None:
            # lfp = {key: value[index] for key, value in self.lfp_data.items()}
            lfp = self.lfp_data[index]
            return lfp, label, idx
        elif self.lfp_data is None and self.spike_data is not None:
            spike = self.spike_data[index]
            return spike, label, idx

        lfp = self.lfp_data[index]
        spike = self.spike_data[index]
        # if self.transform is not None:
        #     neuron_feature = self.transform(neuron_feature)
        # if self.transform:
        #     random_number = random.random()
        #     if random_number < 0.5:
        #         neuron_feature = random_shift(neuron_feature, 2)
        return (lfp, spike), label, idx


def create_weighted_loaders(
    dataset,
    config,
    batch_size=128,
    seed=42,
    p_val=0.1,
    batch_sample_num=2048,
    shuffle=True,
    transform=None,
    extras={},
):
    # assert 0 < p_val < 1.0, 'p_val must be greater than 0 and smaller than 1'
    if p_val > 0:
        assert 0 < p_val < 1.0, "p_val must be greater than 0 and smaller than 1"
        dataset_size = len(dataset)
        all_indices = list(range(dataset_size))

        class_value, class_count = np.unique(dataset.label, axis=0, return_counts=True)
        class_weight_dict = {key.tobytes(): dataset_size / value for key, value in zip(class_value, class_count)}
        # if config['use_spontaneous']:
        #     spontaneous_class = class_value[1].tobytes()
        #     data_weights = np.array([class_weight_dict[label.tobytes()] if label.tobytes() != spontaneous_class
        #                              else class_weight_dict[label.tobytes()] * 10 for label in dataset.label])
        # else:
        #     data_weights = np.array([class_weight_dict[label.tobytes()] for label in dataset.label])
        data_weights = np.array([class_weight_dict[label.tobytes()] for label in dataset.label])
        # val_indices = []
        # train_indices = []
        # for cls in class_value:
        #     indices = np.where(np.all(dataset.label == cls, axis=1))[0]
        #     k = int(np.ceil(indices.size * p_val))
        #     if config['use_shuffle']:
        #         val = np.random.choice(indices, size=k, replace=False)
        #         train_mask = np.in1d(indices, val, invert=True)
        #         train = indices[train_mask]
        #     else:
        #         val = indices[-k:]
        #         train = indices[:-k]
        #     val_indices.append(val)
        #     train_indices.append(train)
        # val_indices = np.concatenate(val_indices, axis=0)
        # train_indices = np.concatenate(train_indices, axis=0)
        tag_combinations = np.apply_along_axis(lambda x: "".join(map(str, x)), 1, dataset.label)
        unique_combinations, indices = np.unique(tag_combinations, return_inverse=True)
        grouped_indices = {tag: np.where(indices == i)[0] for i, tag in enumerate(unique_combinations)}

        def find_continuous_chunks(index_array):
            chunks = np.split(index_array, np.where(np.diff(index_array) != 1)[0] + 1)
            return chunks

        train_indices = []
        val_indices = []

        for tag, idx_group in grouped_indices.items():
            continuous_chunks = find_continuous_chunks(idx_group)
            for chunk in continuous_chunks:
                chunk_length = len(chunk)
                if chunk_length > 1:
                    val_size_start = max(1, int(np.floor(chunk_length * 0.1)))
                    val_size_end = max(1, int(np.ceil(chunk_length * 0.1)))

                    val_indices_start = chunk[:val_size_start]
                    val_indices_end = chunk[-val_size_end:]
                    train_indices_chunk = chunk[val_size_start:-val_size_end] if chunk_length > 2 else []

                    val_indices.extend(val_indices_start)
                    val_indices.extend(val_indices_end)
                    train_indices.extend(train_indices_chunk)
                else:
                    val_indices.extend(chunk)

        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)

        train_label_save_path = os.path.join(config["test_save_path"], "train_label")
        np.save(train_label_save_path, dataset.label[train_indices])

        val_label_save_path = os.path.join(config["test_save_path"], "val_label")
        np.save(val_label_save_path, dataset.label[val_indices])

        if config["use_lfp"] and not config["use_combined"]:
            val_save_path = os.path.join(config["test_save_path"], "val_lfp")
            # np.save(val_save_path, {key: value[val_indices] for key, value in dataset.lfp_data.items()})
            np.save(val_save_path, dataset.data[val_indices])
        elif config["use_spike"] and not config["use_combined"]:
            val_save_path = os.path.join(config["test_save_path"], "val_clusterless")
            np.save(val_save_path, dataset.data[val_indices])
        elif config["use_combiend"]:
            val_save_path = os.path.join(config["test_save_path"], "val_lfp")
            np.save(val_save_path, dataset.data[val_indices])
            val_save_path = os.path.join(config["test_save_path"], "val_clusterless")
            np.save(val_save_path, dataset.data[val_indices])

        assert len(set(val_indices)) + len(set(train_indices)) == len(all_indices)
        if shuffle:
            # np.random.seed(seed)
            np.random.shuffle(train_indices)
            np.random.shuffle(val_indices)

        if config["use_combined"]:
            spike_train = dataset.data["clusterless"][train_indices]
            spike_val = dataset.data["clusterless"][val_indices]
            lfp_train = dataset.data["lfp"][train_indices]
            lfp_val = dataset.data["lfp"][val_indices]
        else:
            spike_train = dataset.data[train_indices] if config["use_spike"] else None
            spike_val = dataset.data[val_indices] if config["use_spike"] else None
            lfp_train = dataset.data[train_indices] if config["use_lfp"] else None
            lfp_val = dataset.data[val_indices] if config["use_lfp"] else None

        label_train = dataset.smoothed_label[train_indices]
        label_val = dataset.smoothed_label[val_indices]
        # label_train = dataset.label[train_indices]
        # label_val = dataset.label[val_indices]
        train_pos = label_train.sum(axis=0)
        train_neg = label_train.shape[0] - train_pos
        train_pos_weights = train_neg / train_pos
        val_pos = label_val.sum(axis=0)
        val_neg = label_val.shape[0] - val_pos
        val_pos_weights = val_neg / val_pos

        train_dataset = MyDataset(
            lfp_train,
            spike_train,
            label_train,
            train_indices,
            transform=transform,
            pos_weight=train_pos_weights,
        )
        val_dataset = MyDataset(lfp_val, spike_val, label_val, val_indices, pos_weight=val_pos_weights)
        test_dataset = None

        num_workers = 1
        pin_memory = True

        sampler_train = WeightedRandomSampler(
            weights=data_weights[train_indices],
            num_samples=batch_sample_num,
            replacement=True,
        )

        # sampler_val = WeightedRandomSampler(weights=data_weights[val_indices], num_samples=len(val_dataset), replacement=True)
        sampler_val = None

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=sampler_val,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
        )

        # test_loader = DataLoader(
        #     test_dataset,
        #     batch_size=batch_size,
        #     sampler=None,
        #     num_workers=num_workers,
        #     pin_memory=pin_memory,
        #     shuffle=False,
        # )
        test_loader = None
    elif p_val == 0:
        dataset_size = len(dataset)
        all_indices = list(range(dataset_size))

        class_value, class_count = np.unique(dataset.label[:, 0:8], axis=0, return_counts=True)

        class_weight_dict = {key.tobytes(): dataset_size / value for key, value in zip(class_value, class_count)}
        # if config['use_spontaneous']:
        #     spontaneous_class = class_value[1].tobytes()
        #     data_weights = np.array([class_weight_dict[label.tobytes()] if label.tobytes() != spontaneous_class
        #                              else class_weight_dict[label.tobytes()] * 10 for label in dataset.label])
        # else:
        #     data_weights = np.array([class_weight_dict[label.tobytes()] for label in dataset.label])
        data_weights = np.array([class_weight_dict[label.tobytes()] for label in dataset.label[:, 0:8][all_indices]])
        train_indices = np.array(all_indices)

        label_save_path = os.path.join(config["test_save_path"], "train_label")
        np.save(label_save_path, dataset.label[train_indices])

        if shuffle:
            # np.random.seed(seed)
            np.random.shuffle(train_indices)

        if config["use_combined"]:
            spike_train = dataset.data["clusterless"][train_indices]
            lfp_train = dataset.data["lfp"][train_indices]
        else:
            spike_train = dataset.data[train_indices] if config["use_spike"] else None
            lfp_train = dataset.data[train_indices] if config["use_lfp"] else None

        label_train = dataset.smoothed_label[train_indices]
        # label_train = dataset.label[train_indices]
        # label_val = dataset.label[val_indices]
        train_pos = label_train.sum(axis=0)
        train_neg = label_train.shape[0] - train_pos
        # train_pos_weights = np.sqrt(train_neg / train_pos)
        train_pos_weights = train_neg / train_pos

        train_dataset = MyDataset(
            lfp_train,
            spike_train,
            label_train,
            train_indices,
            transform=transform,
            pos_weight=train_pos_weights,
        )
        val_dataset = None
        test_dataset = None

        num_workers = 1
        pin_memory = True

        sampler_train = WeightedRandomSampler(
            weights=data_weights[train_indices],
            num_samples=batch_sample_num,
            replacement=True,
        )

        # sampler_val = WeightedRandomSampler(weights=data_weights[val_indices], num_samples=len(val_dataset), replacement=True)
        sampler_val = None

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
        )

        test_loader = None
        val_loader = None
    # Return the training, validation, test DataLoader objects
    return train_loader, val_loader, test_loader


def create_inference_loaders(
    dataset,
    batch_size=128,
    seed=42,
    batch_sample_num=2048,
    shuffle=False,
    extras={},
):
    num_workers = 1
    pin_memory = False
    inference_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
    )

    return inference_loader
