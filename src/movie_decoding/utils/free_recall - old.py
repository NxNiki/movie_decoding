import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# from params import sorted_channels, taylor_channel_path, tonmoy_channel_path, path_shift_index, bin_len
from scipy.interpolate import interp1d
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torchvision.transforms import transforms

SF = 1000
FREE_RECALL_LFP_TIME = {
    "555_1": (42 * 60 + 10, 44 * 60 + 19),
    "562_1": (43 * 60 + 58, 49 * 60 + 55),
    "562_2": (0 * 60 + 20, 5 * 60 + 50),
    "564_1": (1 * 3600 + 30 * 60 + 59, 1 * 3600 + 35 * 60 + 35),
    "564_2": (2 * 60 + 45, 7 * 60 + 48),
    "566_1": (48 * 60 + 52, 53 * 60 + 47),
    "566_2": (1 * 3600 + 4 * 60 + 30, 1 * 3600 + 13 * 60 + 38),
    "566_3": (4 * 60 + 38, 18 * 60 + 25),
    "566_4": (29 * 60 + 7, 35 * 60 + 12),
    "566_5": (0, 1 * 3600),
}
FREE_RECALL_SPIKE_TIME = {
    "555_1": (42 * 60 + 10, 44 * 60 + 19),
    "562_1": (43 * 60 + 58, 49 * 60 + 55),
    "562_2": (
        4689.152 + 7200.256 * 4 + 3148.032 + 20,
        4689.152 + 7200.256 * 4 + 3148.032 + 5 * 60 + 50,
    ),
    "564_1": (1 * 3600 + 30 * 60 + 59, 1 * 3600 + 35 * 60 + 35),
    "564_2": (
        6428.672 + 7200.256 * 4 + 5063.424 + 2 * 60 + 45,
        6428.672 + 7200.256 * 4 + 5063.424 + 7 * 60 + 48,
    ),
}
OFFSET = {
    "555_1": 0,
    "562_1": 0,
    "562_2": 0,
    "564_1": 0,
    "564_2": 0,
    "566_1": 0,
    "566_2": 0,
    "566_3": 0,
    "566_4": 0,
    "566_5": 0,
}
SPONTANEOUS_SAMPLES = {
    "555": 600000,
    "562": 3000000,
}
SPIKE_CHANNELS = {
    "555_1": list(np.arange(1, 25))
    + list(np.arange(41, 46))
    + [47, 66, 68, 74]
    + list(np.arange(80, 89)),
    "562_1": list(np.arange(1, 25))
    + list(np.arange(34, 49))
    + list(np.arange(65, 72))
    + list(np.arange(73, 81)),
    "562_2": list(np.arange(1, 25))
    + list(np.arange(34, 49))
    + list(np.arange(65, 72))
    + list(np.arange(73, 81)),
    "564_1": [2, 3]
    + list(np.arange(5, 10))
    + list(np.arange(11, 17))
    + [34, 36, 39, 41, 42, 44, 45, 47]
    + list(np.arange(49, 65))
    + [69, 76, 77],
    "564_2": [2, 3]
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


class InferenceDataset(Dataset):
    def __init__(
        self,
        data_path,
        patient,
        lfp=False,
        spike=False,
        bipolar=False,
        sleep=False,
        phase=1,
        hour=1,
    ):
        self.spike = spike
        self.lfp = lfp
        self.data_length = 0
        self.movie_sampling_rate = 30
        self.resolution = 4
        self.lfp_sf = SF  # Hz
        self.alignment_offset = OFFSET[patient + "_" + str(phase)]  # seconds
        self.num_samples = (
            int(
                FREE_RECALL_LFP_TIME[patient + "_" + str(phase)][1]
                - FREE_RECALL_LFP_TIME[patient + "_" + str(phase)][0]
            )
            * self.resolution
        )
        # self.sample_range = [(self.alignment_offset + FREE_RECALL_TIME[patient][0]) * self.lfp_sf,
        #                      (self.alignment_offset + FREE_RECALL_TIME[patient][1]) * self.lfp_sf]
        # # self.sample_range = [(self.alignment_offset + FREE_RECALL_TIME['562_2'][0]) * self.lfp_sf,
        # #                      (self.alignment_offset + FREE_RECALL_TIME['562_2'][1]) * self.lfp_sf]
        # self.frame_range = [(self.alignment_offset+FREE_RECALL_TIME[patient][0]) * self.movie_sampling_rate,
        #                     (self.alignment_offset+FREE_RECALL_TIME[patient][1]) * self.movie_sampling_rate]
        # # self.frame_range = [(self.alignment_offset + FREE_RECALL_TIME['562_2'][0]) * self.movie_sampling_rate,
        # #                     (self.alignment_offset + FREE_RECALL_TIME['562_2'][1]) * self.movie_sampling_rate]

        self.smoothed_label = []
        self.label = []
        self.final_lfp_data = []
        self.final_spike_data = []

        # if LFP
        if lfp:
            self.sample_range = [
                (
                    self.alignment_offset
                    + FREE_RECALL_LFP_TIME[patient + "_" + str(phase)][0]
                )
                * self.lfp_sf,
                (
                    self.alignment_offset
                    + FREE_RECALL_LFP_TIME[patient + "_" + str(phase)][1]
                )
                * self.lfp_sf,
            ]
            if phase in [1, 2]:
                self.lfp_data_path = os.path.join(data_path, patient, "movie")
                self.lfp_data = self.load_npz("multi")
                self.lfp_data = self.lfp_data[
                    :, int(self.sample_range[0]) : int(self.sample_range[1])
                ]
            elif phase in [3, 4]:
                self.lfp_data_path = os.path.join(data_path, patient, "postsleep")
                self.lfp_data = self.load_npz("multi")
                self.lfp_data = self.lfp_data[
                    :, int(self.sample_range[0]) : int(self.sample_range[1])
                ]
            elif phase in [5]:
                self.lfp_data_path = os.path.join(data_path, patient, "sleep")
                self.lfp_data = self.load_npz_by_chunk(hour=hour)

            # TODO check data label size
            sample_size = self.lfp_sf / self.resolution
            for sample in range(self.num_samples):
                window_left = sample * sample_size
                window_right = (sample + 1) * sample_size
                if window_left < 0 or window_right >= self.lfp_data.shape[-1]:
                    continue
                features = self.lfp_data[:, int(window_left) : int(window_right)]
                self.final_lfp_data.append(features)
            print("LFP Data Loaded")
        if spike:
            self.frame_range = [
                (
                    self.alignment_offset
                    + FREE_RECALL_SPIKE_TIME[patient + "_" + str(phase)][0]
                )
                * self.movie_sampling_rate,
                (
                    self.alignment_offset
                    + FREE_RECALL_SPIKE_TIME[patient + "_" + str(phase)][1]
                )
                * self.movie_sampling_rate,
            ]
            self.sorted_channels = SPIKE_CHANNELS[patient + "_" + str(phase)]
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
            self.spike_data = []
            for i in range(len(self.spike_frames)):
                in_movie = np.logical_and(
                    self.spike_frames[i] >= self.frame_range[0],
                    self.spike_frames[i] <= self.frame_range[1],
                )
                spike_frame = self.spike_frames[i][in_movie]
                units_ts = np.arange(self.frame_range[0], self.frame_range[1], step=1)
                units_firing, bin_edges = np.histogram(spike_frame, bins=units_ts)
                min_len = min(len(units_ts), len(units_firing))
                units_ts = units_ts[:min_len]
                units_firing = units_firing[:min_len]

                units_firing, units_ts = self.interpolate_neural_data(
                    units_firing, units_ts
                )
                self.spike_data.append(units_firing)
            self.spike_data = np.array(self.spike_data)
            del self.spike_times
            del self.spike_frames

            # sample_size = 8 #np.ceil(self.movie_sampling_rate / self.resolution)
            for sample in range(self.num_samples):
                quotient, remainder = divmod(sample, self.resolution)
                # windows = [(0, 8), (7, 15), (15, 23), (22, 30)]
                # windows = [(0, 15), (5, 20), (10, 25), (15, 30)]
                # windows = [(0, 24), (2, 26), (4, 28), (6, 30)]
                windows = SPIKE_WINDOWS[patient]
                window_left = (
                    quotient * self.movie_sampling_rate + windows[remainder][0]
                )
                window_right = (
                    quotient * self.movie_sampling_rate + windows[remainder][1]
                )
                # window_right = np.ceil((sample + 1) * self.movie_sampling_rate / self.resolution)
                # window_left = window_right - np.ceil(self.movie_sampling_rate / self.resolution)
                if window_left < 0 or window_right >= self.spike_data.shape[-1]:
                    continue
                features = self.spike_data[:, int(window_left) : int(window_right)]
                self.final_spike_data.append(features)
            print("Spike Data Loaded")

        self.preprocess_data()
        print("Done")

    def interpolate_neural_data(self, data, original_timestamps):
        new_timestamps = np.arange(
            original_timestamps[0], original_timestamps[-1] + 1, step=1
        )
        f = interp1d(original_timestamps, data, axis=-1)
        new_data = f(new_timestamps)
        return new_data, new_timestamps

    def load_channels(self):
        spike_times = []
        channel_labels = []
        path = self.channel_path
        for channel in self.sorted_channels:
            try:
                spike_data = mat73.loadmat(
                    os.path.join(path, f"times_CSC{channel}.mat")
                )
                print(channel, " load with mat73")
            except:
                spike_data = loadmat(os.path.join(path, f"times_CSC{channel}.mat"))
                print(channel, " load with scipy")
            cluster_class = spike_data["cluster_class"]
            n_count = np.max(cluster_class, axis=0)[0]
            # print(f"channel {channel} has {n_count} neurons")
            # print(f"n_count: {int(n_count)}")
            for neuron in range(1, int(n_count) + 1):
                # print(neuron)
                spike_times.append(
                    (cluster_class[np.where(cluster_class[:, 0] == neuron)])[:, 1]
                )
                channel_labels.append(f"CSC{channel}_N{neuron}")
        return spike_times

    def load_npz(self, mode="multi"):
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
            lfp_files = glob.glob(
                os.path.join(self.lfp_data_path, "marco_lfp_spectrum_*.npz")
            )
            for file in lfp_files:
                first_8_last_8 = np.load(file)["data"]
                first_8_last_8 = np.concatenate(
                    (first_8_last_8[:8, :], first_8_last_8[-8:, :]), axis=0
                )
                # first_8_last_8 = first_8_last_8[:8, :]
                lfp_mat = superVstack(lfp_mat, first_8_last_8)
        else:
            fn = os.path.join(self.lfp_data_path, "marco_lfp_john.npz")
            lfp_mat = np.load(fn)["data"]
        return np.array(lfp_mat).astype(np.float32)

    def load_npz_by_chunk(self, hour=1):
        def superVstack(a, b):
            # make it so you can vstack onto empty row
            if len(a) == 0:
                stack = b
            elif len(b) == 0:
                stack = a
            else:
                stack = np.vstack([a, b])
            return stack

        lfp_mat = []
        lfp_files = glob.glob(
            os.path.join(
                self.lfp_data_path, "marco_lfp_spectrum_*_hour_{}.npz".format(hour)
            )
        )
        for file in lfp_files:
            first_8_last_8 = np.load(file)["data"]
            first_8_last_8 = np.concatenate(
                (first_8_last_8[:8, :], first_8_last_8[-8:, :]), axis=0
            )
            # first_8_last_8 = first_8_last_8[:8, :]
            lfp_mat = superVstack(lfp_mat, first_8_last_8)
        return np.array(lfp_mat).astype(np.float32)

    def load_pickle(self, fn):
        if not os.path.exists(fn):
            return
        with open(fn, "rb") as f:
            lookup = pickle.load(f)
        return lookup

    def preprocess_data(self):
        if self.spike:
            self.spike_data = np.copy(self.final_spike_data)
            del self.final_spike_data
            self.spike_data = np.array(self.spike_data).astype(np.float32)
            length = self.spike_data.shape[0]
        if self.lfp:
            self.lfp_data = np.copy(self.final_lfp_data)
            del self.final_lfp_data
            self.lfp_data = np.array(self.lfp_data).astype(np.float32)
            length = self.lfp_data.shape[0]
        if self.lfp and self.spike:
            length = min(self.spike_data.shape[0], self.lfp_data.shape[0])

        self.data_length = length
        # self.label = np.array(self.ml_label).transpose()[:length, :].astype(np.float32)
        # self.smoothed_label = np.array(self.smoothed_ml_label).transpose()[:length, :].astype(np.float32)

    def visualization(self):
        combined_bins = np.vstack((self.data, self.labels))
        combined_bins = self.normalize_bins(combined_bins)
        figpath = "./bins.png"

        plt.figure()
        plt.imshow(combined_bins, aspect="auto", interpolation="nearest")
        # plt.plot(np.ones(bins.shape[1])*bins.shape[0]-1.5)
        plt.savefig(figpath)
        plt.show()

    def __len__(self):
        return self.data_length


class MyDataset(Dataset):
    def __init__(self, lfp_data, spike_data, label, indices, transform=None):
        self.lfp_data = None
        self.spike_data = None
        if lfp_data is not None:
            self.lfp_data = np.array(lfp_data).astype(np.float32)
        if spike_data is not None:
            self.spike_data = np.array(spike_data).astype(np.float32)
        # self.label = np.array(label).astype(np.float32)
        self.transform = transform
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        # label = self.label[index]
        idx = self.indices[index]
        if self.lfp_data is not None and self.spike_data is None:
            lfp = self.lfp_data[index]
            return lfp, idx
        elif self.lfp_data is None and self.spike_data is not None:
            spike = self.spike_data[index]
            return spike, idx

        lfp = self.lfp_data[index]
        spike = self.spike_data[index]
        # if self.transform is not None:
        #     neuron_feature = self.transform(neuron_feature)
        # if self.transform:
        #     random_number = random.random()
        #     if random_number < 0.5:
        #         neuron_feature = random_shift(neuron_feature, 2)
        return (lfp, spike), idx


def create_inference_combined_loaders(
    dataset,
    config,
    batch_size=128,
    seed=42,
    batch_sample_num=2048,
    shuffle=False,
):
    num_workers = 1
    pin_memory = False

    dataset_size = len(dataset)
    all_indices = list(range(dataset_size))

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(all_indices)

    if config.use_lfp and not config.use_spike:
        lfp_inference = dataset.lfp_data[all_indices]
        spike_inference = None
    elif not config.use_lfp and config.use_spike:
        lfp_inference = None
        spike_inference = dataset.spike_data[all_indices]
    else:
        lfp_inference = dataset.lfp_data[all_indices]
        spike_inference = dataset.spike_data[all_indices]
    # label_inference = dataset.smoothed_label[all_indices]
    label_inference = None

    inference_dataset = MyDataset(
        lfp_inference, spike_inference, label_inference, all_indices
    )

    inference_loader = DataLoader(
        inference_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
    )
    return inference_loader
