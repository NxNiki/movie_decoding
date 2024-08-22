import copy
import glob
import os
import pickle
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from scipy.ndimage import convolve1d
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torchvision.transforms import transforms

SF = 2000
OFFSET = {
    "555": 4.58,
    "562": 134.194,
    "564": [(792.79, 1945), (3732.44, 5091)],
    "566": 380.5814,
}
CONTROL = {
    "566": [(121, 1520), (1544, 2825)],
}
SPONTANEOUS_SAMPLES = {
    "555": 600000,
    "562": 550000,
    "564": [
        (int(417.79 * SF), int(777.79 * SF)),
        (int(3357.44 * SF), int(3717.44 * SF)),
    ],
    "566": 360 * SF,
}
SPONTANEOUS_FRAMES = {
    "555": 600000,
    "562": 550000,
    "564": [
        (int(417.79 * 30), int(777.79 * 30)),
        (int(3357.44 * 30), int(3717.44 * 30)),
    ],
}
SPIKE_CHANNELS = {
    "555": list(np.arange(1, 25))
    + list(np.arange(41, 46))
    + [47, 66, 68, 74]
    + list(np.arange(80, 89)),
    "562": list(np.arange(1, 25))
    + list(np.arange(34, 49))
    + list(np.arange(65, 72))
    + list(np.arange(73, 81)),
    "564": [2, 3]
    + list(np.arange(5, 10))
    + list(np.arange(11, 17))
    + [34, 36, 39, 41, 42, 44, 45, 47]
    + list(np.arange(49, 65))
    + [69, 76, 77],
}
SPIKE_WINDOWS = {
    "555": [(0, 24), (2, 26), (4, 28), (6, 30)],
    "562": [(0, 8), (7, 15), (15, 23), (22, 30)],
    "564": [(0, 8), (7, 15), (15, 23), (22, 30)],
}

SPIKE_CHANNEL = {"555": 37, "562": 50, "564": 59, "566": 56}
SPIKE_FRAME = {"555": 24, "562": 8, "564": 24, "566": 50}  # 8, 15, 24
LFP_CHANNEL = {"555": 208, "562": 160, "564": 160, "566": 56 * 8}


class NeuronDataset:
    def __init__(self, patient, use_spike=False, use_lfp=False, use_spontaneous=False):
        self.patient = patient
        self.use_spontaneous = use_spontaneous
        self.use_spike = use_spike
        self.use_lfp = use_lfp

        # assume in label/sec
        self.movie_sampling_rate = 30
        # self.movie_label_path = "/mnt/SSD5/yyding/Datasets/Character_TimeStamp/"
        self.movie_label_path = (
            "E://projects//Datasets//12concepts//12concepts_merged_more.npy"
        )
        # self.movie_label_path = 'E://projects//Datasets//12concepts//4concepts.npy'

        self.resolution = 4
        self.lfp_sf = SF  # Hz
        self.alignment_offset = OFFSET[patient]  # seconds
        self.ml_label = np.load(self.movie_label_path)
        # self.ml_label = np.append(self.ml_label, np.zeros((1, self.ml_label.shape[1])), axis=0)
        if patient == "564":
            self.sample_range = [
                (
                    self.alignment_offset[0][0] * self.lfp_sf,
                    self.alignment_offset[0][1] * self.lfp_sf,
                ),
                (
                    self.alignment_offset[1][0] * self.lfp_sf,
                    self.alignment_offset[1][1] * self.lfp_sf,
                ),
            ]

            self.time_range = [
                (self.alignment_offset[0][0], self.alignment_offset[0][1]),
                (self.alignment_offset[1][0], self.alignment_offset[1][1]),
            ]
            self.frame_range = [
                (
                    self.alignment_offset[0][0] * self.movie_sampling_rate,
                    self.alignment_offset[0][1] * self.movie_sampling_rate,
                ),
                (
                    self.alignment_offset[1][0] * self.movie_sampling_rate,
                    self.alignment_offset[1][1] * self.movie_sampling_rate,
                ),
            ]

            label_crop = min(
                self.ml_label.shape[-1],
                int(self.alignment_offset[0][1] - self.alignment_offset[0][0]),
                int(self.alignment_offset[1][1] - self.alignment_offset[1][0]),
            )
            self.ml_label = np.repeat(
                np.concatenate(
                    (self.ml_label[:, :label_crop], self.ml_label[:, :label_crop]),
                    axis=1,
                ),
                self.resolution,
                axis=1,
            )
            self.smoothed_ml_label = self.smooth_label()
            # create lfp data
            if lfp:
                self.lfp_data_path = (
                    "E://projects//Datasets//Neuron//lfp_data//{}//movie//".format(
                        patient
                    )
                )
                self.lfp_data = self.load_npz(self.lfp_data_path, "multi")
                first = self.lfp_data[
                    :, int(self.sample_range[0][0]) : int(self.sample_range[0][1])
                ]
                second = self.lfp_data[
                    :, int(self.sample_range[1][0]) : int(self.sample_range[1][1])
                ]
                overlap = min(first.shape[-1], second.shape[-1])
                if spontaneous:
                    self.ml_label = np.append(
                        self.ml_label, np.zeros((1, self.ml_label.shape[1])), axis=0
                    )
                    self.smoothed_ml_label = np.append(
                        self.smoothed_ml_label,
                        np.zeros((1, self.smoothed_ml_label.shape[1])),
                        axis=0,
                    )
                    self.spontaneous_lfp_data = np.concatenate(
                        (
                            self.lfp_data[
                                :,
                                SPONTANEOUS_SAMPLES[patient][0][
                                    0
                                ] : SPONTANEOUS_SAMPLES[patient][0][1],
                            ],
                            self.lfp_data[
                                :,
                                SPONTANEOUS_SAMPLES[patient][1][
                                    0
                                ] : SPONTANEOUS_SAMPLES[patient][1][1],
                            ],
                        ),
                        axis=1,
                    )
                self.lfp_data = np.copy(
                    np.concatenate((first[:, :overlap], second[:, :overlap]), axis=1)
                )
            if use_spike:
                self.sorted_channels = SPIKE_CHANNELS[patient]
                self.channel_path = "E://projects//Datasets//neuron//spike_data//{}//SortingCombinedSpikes".format(
                    patient
                )
                self.bin_len = 1  # ms

                self.spike_times = self.load_channels()
                self.spike_frames = [
                    np.round(i)
                    for i in self.movie_sampling_rate
                    * np.array(self.spike_times, dtype=object)
                    / 1000
                ]
                # min_spike_frame = min(np.min(neuron) for neuron in self.spike_frames)
                # max_spike_frame = max(np.max(neuron) for neuron in self.spike_frames)
                first, second = [], []
                for i in range(len(self.spike_frames)):
                    # first
                    in_movie = np.logical_and(
                        self.spike_frames[i] >= self.frame_range[0][0],
                        self.spike_frames[i] <= self.frame_range[0][1],
                    )
                    spike_frame = self.spike_frames[i][in_movie]
                    units_ts = np.arange(
                        np.floor(self.frame_range[0][0]),
                        np.ceil(self.frame_range[0][1]),
                        step=1,
                    )
                    units_firing, bin_edges = np.histogram(spike_frame, bins=units_ts)
                    min_len = min(len(units_ts), len(units_firing))
                    units_ts = units_ts[:min_len]
                    units_firing = units_firing[:min_len]
                    units_firing, units_ts = self.interpolate_neural_data(
                        units_firing, units_ts
                    )
                    first.append(units_firing)

                    # second
                    in_movie = np.logical_and(
                        self.spike_frames[i] >= self.frame_range[1][0],
                        self.spike_frames[i] <= self.frame_range[1][1],
                    )
                    spike_frame = self.spike_frames[i][in_movie]
                    units_ts = np.arange(
                        np.floor(self.frame_range[1][0]),
                        np.ceil(self.frame_range[1][1]),
                        step=1,
                    )
                    units_firing, bin_edges = np.histogram(spike_frame, bins=units_ts)
                    min_len = min(len(units_ts), len(units_firing))
                    units_ts = units_ts[:min_len]
                    units_firing = units_firing[:min_len]
                    units_firing, units_ts = self.interpolate_neural_data(
                        units_firing, units_ts
                    )
                    second.append(units_firing)
                first = np.array(first)
                second = np.array(second)
                overlap = min(first.shape[-1], second.shape[-1])
                self.spike_data = np.copy(
                    np.concatenate((first[:, :overlap], second[:, :overlap]), axis=1)
                )

                if bipolar:
                    self.spontaneous_frame_range = SPONTANEOUS_FRAMES[patient]
                    first, second = [], []
                    for i in range(len(self.spike_frames)):
                        # first
                        in_movie = np.logical_and(
                            self.spike_frames[i] >= self.spontaneous_frame_range[0][0],
                            self.spike_frames[i] <= self.spontaneous_frame_range[0][1],
                        )
                        spike_frame = self.spike_frames[i][in_movie]
                        units_ts = np.arange(
                            np.floor(self.frame_range[0][0]),
                            np.ceil(self.frame_range[0][1]),
                            step=1,
                        )
                        units_firing, bin_edges = np.histogram(
                            spike_frame, bins=units_ts
                        )
                        min_len = min(len(units_ts), len(units_firing))
                        units_ts = units_ts[:min_len]
                        units_firing = units_firing[:min_len]
                        units_firing, units_ts = self.interpolate_neural_data(
                            units_firing, units_ts
                        )
                        first.append(units_firing)

                        # second
                        in_movie = np.logical_and(
                            self.spike_frames[i] >= self.frame_range[1][0],
                            self.spike_frames[i] <= self.frame_range[1][1],
                        )
                        spike_frame = self.spike_frames[i][in_movie]
                        units_ts = np.arange(
                            np.floor(self.frame_range[1][0]),
                            np.ceil(self.frame_range[1][1]),
                            step=1,
                        )
                        units_firing, bin_edges = np.histogram(
                            spike_frame, bins=units_ts
                        )
                        min_len = min(len(units_ts), len(units_firing))
                        units_ts = units_ts[:min_len]
                        units_firing = units_firing[:min_len]
                        units_firing, units_ts = self.interpolate_neural_data(
                            units_firing, units_ts
                        )
                        second.append(units_firing)
                    first = np.array(first)
                    second = np.array(second)
                    overlap = min(first.shape[-1], second.shape[-1])
                    self.spontaneous_spike_data = np.copy(
                        np.concatenate(
                            (first[:, :overlap], second[:, :overlap]), axis=1
                        )
                    )

                    self.ml_label = np.append(
                        self.ml_label, np.zeros((1, self.ml_label.shape[1])), axis=0
                    )
                    self.smoothed_ml_label = np.append(
                        self.smoothed_ml_label,
                        np.zeros((1, self.smoothed_ml_label.shape[1])),
                        axis=0,
                    )

                del self.spike_times
                del self.spike_frames
        else:
            self.sample_range = [
                self.alignment_offset * self.lfp_sf,
                (self.alignment_offset + self.ml_label.shape[-1]) * self.lfp_sf,
            ]
            self.time_range = [
                self.alignment_offset,
                (self.alignment_offset + self.ml_label.shape[-1]),
            ]
            self.frame_range = [
                self.alignment_offset * self.movie_sampling_rate,
                (self.alignment_offset + self.ml_label.shape[-1])
                * self.movie_sampling_rate,
            ]
            self.ml_label = np.repeat(self.ml_label, self.resolution, axis=1)
            self.smoothed_ml_label = self.smooth_label()

            # create spike data
            if use_spike:

                def sort_filename(filename):
                    """Extract the numeric part of the filename and use it as the sort key"""
                    return [
                        int(x) if x.isdigit() else x
                        for x in re.findall(r"\d+|\D+", filename)
                    ]

                self.spike_path = "E://projects//Datasets//neuron//spike_data//{}//time_movie//".format(
                    patient
                )
                self.spike_files = glob.glob(os.path.join(self.spike_path, "*.npz"))
                self.spike_files = sorted(self.spike_files, key=sort_filename)
                self.spike_data = self.load_clustless()

            # create lfp data
            if use_lfp:
                self.lfp_data_path = (
                    "E://projects//Datasets//Neuron//lfp_data//{}//movie//".format(
                        patient
                    )
                )
                self.lfp_data = self.load_npz(self.lfp_data_path, "multi")
                # self.spontaneous_lfp_data = self.lfp_data[:, :int(self.sample_range[0])]
                # self.spontaneous_lfp_data = self.spontaneous_lfp_data[:, :220000]
                if use_spontaneous:
                    self.ml_label = np.append(
                        self.ml_label, np.zeros((1, self.ml_label.shape[1])), axis=0
                    )
                    self.smoothed_ml_label = np.append(
                        self.smoothed_ml_label,
                        np.zeros((1, self.smoothed_ml_label.shape[1])),
                        axis=0,
                    )
                    self.spontaneous_lfp_path = "E://projects//Datasets//Neuron//lfp_data//{}//spontaneous//".format(
                        patient
                    )
                    self.spontaneous_lfp_data = self.load_npz(
                        self.spontaneous_lfp_path, "multi"
                    )
                    # self.spontaneous_lfp_data = self.spontaneous_lfp_data[:, :SPONTANEOUS_SAMPLES[patient]]
                self.lfp_data = self.lfp_data[
                    :, int(self.sample_range[0]) : int(self.sample_range[1])
                ]

        # if not spike and lfp:
        #     self.new_spike_frames = []
        #     self.ml_data = copy.deepcopy(self.new_lfp_data)
        # elif spike and not lfp:
        #     self.new_lfp_data = []
        #     self.ml_data = copy.deepcopy(self.new_spike_frames)
        # else:
        #     self.ml_data = np.concatenate((self.new_lfp_data, self.new_spike_frames), axis=0)
        # del self.new_spike_frames
        # del self.new_lfp_data
        # self.visualization()
        self.smoothed_label = []
        self.label = []
        self.final_lfp_data = []
        self.final_spike_data = []
        self.spontaneous_label = []
        self.spontaneous_data = []
        if use_lfp:
            # TODO check data label size
            for second in range(self.ml_label.shape[-1]):
                window_left = second / self.resolution * self.lfp_sf
                window_right = (second + 1) / self.resolution * self.lfp_sf
                if window_left < 0 or window_right >= self.lfp_data.shape[-1]:
                    continue
                features = self.lfp_data[:, int(window_left) : int(window_right)]
                # label = self.ml_label[:, second]
                # smoothed_label = self.smoothed_ml_label[:, second]

                self.final_lfp_data.append(features)
                # self.label.append(label)
                # self.smoothed_label.append(smoothed_label)
            del self.lfp_data
            if self.use_spontaneous:
                sample_size = self.lfp_sf / self.resolution
                num_samples = int(self.spontaneous_lfp_data.shape[-1] // sample_size)
                for sample in range(num_samples):
                    window_left = sample * sample_size
                    window_right = (sample + 1) * sample_size
                    self.spontaneous_data.append(
                        self.spontaneous_lfp_data[
                            :, int(window_left) : int(window_right)
                        ]
                    )
                    self.spontaneous_label.append(
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                    )
        print("Neuron Data Loaded")
        self.preprocess_data()
        print("Done")

    def smooth_label(self):
        sigma = 1
        kernel = np.exp(-(np.arange(-1, 2) ** 2) / (2 * sigma**2))
        kernel /= np.sum(kernel)
        # kernel = np.tile(kernel, (12, 1))

        smoothed_label = convolve1d(self.ml_label, kernel, axis=1)
        max_val = np.max(smoothed_label, axis=1)
        # smoothed_label = smoothed_label / max_val[:, np.newaxis]
        return np.round(smoothed_label, 2)

    def recreate_label(self):
        new_label = []
        for i in range(self.label.shape[0]):
            each_label = self.label[i]
            tmp = []
            for item in each_label:
                if item[0] == 1.0:
                    tmp.append(1.0)
                elif item[1] == 1.0:
                    tmp.append(0.0)
                else:
                    tmp.append(0.0)
            new_label.append(tmp)
        self.label = new_label

    def load_npz(self, path, mode="multi"):
        def superVstack(a, b):
            # make it so you can vstack onto empty row
            if len(a) == 0:
                stack = b
            elif len(b) == 0:
                stack = a
            else:
                stack = np.vstack([a, b])
            return stack

        if mode == "multi":
            lfp_mat = []
            lfp_files = glob.glob(os.path.join(path, "marco_lfp_spectrum_*.npz"))
            for file in lfp_files:
                first_8_last_8 = np.load(file)["data"]
                first_8_last_8 = np.concatenate(
                    (first_8_last_8[:8, :], first_8_last_8[-8:, :]), axis=0
                )
                # first_8_last_8 = first_8_last_8[:8, :]
                lfp_mat = superVstack(lfp_mat, first_8_last_8)
        else:
            fn = os.path.join(path, "marco_lfp_john.npz")
            lfp_mat = np.load(fn)["data"]
        return np.array(lfp_mat).astype(np.float32)

    def load_clustless(self):
        spike = []
        for file in self.spike_files:
            data = np.load(file)["data"]
            if len(data.shape) == 2:
                spike.append(data[:, None, :])
            else:
                spike.append(data)
        # mean = np.mean(spike)
        # std = np.std(spike)
        # spike_standardized = [(arr - mean)/std for arr in spike]
        # normed_spike = [(x - x.mean()) / np.sqrt(x.var() + 1e-7) for x in spike]
        spike = np.concatenate(spike, axis=1)
        # keep_every = slice(None, None, 4)
        # spike = spike[:, :, keep_every]
        return spike

    def bin_spikes(self):
        max_duration = max(np.max(neuron) for neuron in self.spike_times)
        bins = np.zeros((len(self.spike_times), int(max_duration / self.bin_len) - 1))
        for index, spike_time in enumerate(self.spike_times):
            hist, bin_edges = np.histogram(
                spike_time,
                bins=[
                    i * self.bin_len for i in range(int(max_duration / self.bin_len))
                ],
            )
            bins[index] = hist

        return bins, max_duration

    def normalize_bins(self, bins, std_threshold=2):
        print("normalizing")
        norm_bins = bins.astype(float)
        for i, neuron in enumerate(norm_bins[:-1]):
            non_zero_vals = neuron[neuron != 0]
            dev = np.std(non_zero_vals)
            med = np.median(non_zero_vals)
            print(f"thres: {std_threshold * dev}, max: {np.max(neuron)}")
            print(non_zero_vals[non_zero_vals <= std_threshold * dev].shape)
            thres = max(med + std_threshold * dev, 1)
            max_in = np.max(non_zero_vals[non_zero_vals <= thres])
            norm_bins[i][norm_bins[i] > thres] = max_in
            # bins[i] = bins[i].astype(float)
            norm_bins[i] = norm_bins[i] / max_in if max_in else norm_bins[i]
        return norm_bins

    def interpolate_neural_data(self, data, original_timestamps):
        new_timestamps = np.arange(
            original_timestamps[0], original_timestamps[-1] + 1, step=1
        )
        f = interp1d(original_timestamps, data, axis=-1)
        new_data = f(new_timestamps)
        return new_data, new_timestamps

    def load_pickle(self, fn):
        if not os.path.exists(fn):
            return
        with open(fn, "rb") as f:
            lookup = pickle.load(f)
        return lookup

    def preprocess_data(self):
        if self.use_spike:
            length = self.spike_data.shape[0]
        if self.use_lfp:
            self.lfp_data = np.copy(self.final_lfp_data)
            del self.final_lfp_data
            self.lfp_data = np.array(self.lfp_data).astype(np.float32)
            length = self.lfp_data.shape[0]
        if self.use_lfp and self.use_spike:
            length = min(self.spike_data.shape[0], self.lfp_data.shape[0])

        self.label = np.array(self.ml_label).transpose()[:length, :].astype(np.float32)
        self.smoothed_label = (
            np.array(self.smoothed_ml_label).transpose()[:length, :].astype(np.float32)
        )

        if self.use_spontaneous:
            self.spontaneous_data = np.array(self.spontaneous_data).astype(np.float32)
            self.spontaneous_label = np.array(self.spontaneous_label).astype(np.float32)
            # TODO handle lfp spontaneous data and spike spontaneous data later
            if self.use_spike:
                self.spike_data = np.concatenate(
                    (self.spike_data, self.spontaneous_data), axis=0
                )
            if self.use_lfp:
                self.lfp_data = np.concatenate(
                    (self.lfp_data, self.spontaneous_data), axis=0
                )
            self.label = np.concatenate((self.label, self.spontaneous_label), axis=0)
            self.smoothed_label = np.concatenate(
                (self.smoothed_label, self.spontaneous_label), axis=0
            )

    def visualization(self):
        combined_bins = np.vstack((self.data, self.label))
        combined_bins = self.normalize_bins(combined_bins)
        figpath = "./bins.png"

        plt.figure()
        plt.imshow(combined_bins, aspect="auto", interpolation="nearest")
        # plt.plot(np.ones(bins.shape[1])*bins.shape[0]-1.5)
        plt.savefig(figpath)
        plt.show()

    def __len__(self):
        return len(self.label)


def random_shift(array, max_shift):
    shift = np.random.randint(-max_shift, max_shift + 1)
    shifted_array = np.roll(array, shift, axis=1)
    return shifted_array


class MyDataset(Dataset):
    def __init__(self, lfp_data, spike_data, label, indices, transform=None):
        self.lfp_data = None
        self.spike_data = None
        if lfp_data is not None:
            self.lfp_data = lfp_data
        if spike_data is not None:
            self.spike_data = spike_data
        self.label = label
        self.transform = transform
        self.indices = indices

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        label = self.label[index]
        idx = self.indices[index]
        if self.lfp_data is not None and self.spike_data is None:
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


def calculate_class_freq(labels):
    """class 1, class 2, class 3, class 5, no character"""
    label_key = [0, 1, 2, 3, 4]
    label_count = np.zeros(labels.shape[-1] + 1)
    for label in labels:
        indices = np.nonzero(label)
        if len(indices[0]) == 0:
            label_count[-1] += 1
        else:
            label_count[indices] += 1
    return label_key, label_count


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
    # p_val = 0.27
    assert 0 < p_val < 1.0, "p_val must be greater than 0 and smaller than 1"

    dataset_size = len(dataset)
    all_indices = list(range(dataset_size))

    class_value, class_count = np.unique(dataset.label, axis=0, return_counts=True)
    class_weight_dict = {
        key.tobytes(): dataset_size / value
        for key, value in zip(class_value, class_count)
    }
    if config.use_spontaneous:
        spontaneous_class = class_value[1].tobytes()
        data_weights = np.array(
            [
                class_weight_dict[label.tobytes()]
                if label.tobytes() != spontaneous_class
                else class_weight_dict[label.tobytes()] * 10
                for label in dataset.label
            ]
        )
    else:
        data_weights = np.array(
            [class_weight_dict[label.tobytes()] for label in dataset.label]
        )

    val_indices = []
    train_indices = []
    for cls in class_value:
        indices = np.where(np.all(dataset.label == cls, axis=1))[0]
        k = int(np.ceil(indices.size * p_val))
        if config.use_shuffle:
            val = np.random.choice(indices, size=k, replace=False)
            train_mask = np.in1d(indices, val, invert=True)
            train = indices[train_mask]
        else:
            val = indices[-k:]
            train = indices[:-k]
        val_indices.append(val)
        train_indices.append(train)
    val_indices = np.concatenate(val_indices, axis=0)
    train_indices = np.concatenate(train_indices, axis=0)

    label_save_path = os.path.join(
        config.test_save_path, "train_label_fold{}".format(2)
    )
    np.save(label_save_path, dataset.label[train_indices])
    assert len(set(val_indices)) + len(set(train_indices)) == len(all_indices)
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)

    if config.use_lfp and not config.use_spike:
        lfp_train = dataset.lfp_data[train_indices]
        spike_train = None
        lfp_val = dataset.lfp_data[val_indices]
        spike_val = None
    elif not config.use_lfp and config.use_spike:
        lfp_train = None
        spike_train = dataset.spike_data[train_indices]
        lfp_val = None
        spike_val = dataset.spike_data[val_indices]
    else:
        lfp_train = dataset.lfp_data[train_indices]
        lfp_val = dataset.lfp_data[val_indices]
        spike_train = dataset.spike_data[train_indices]
        spike_val = dataset.spike_data[val_indices]
    label_train = dataset.smoothed_label[train_indices]
    label_val = dataset.smoothed_label[val_indices]
    # label_train = dataset.label[train_indices]
    # label_val = dataset.label[val_indices]

    train_dataset = MyDataset(
        lfp_train, spike_train, label_train, train_indices, transform=transform
    )
    val_dataset = MyDataset(lfp_val, spike_val, label_val, val_indices)
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
