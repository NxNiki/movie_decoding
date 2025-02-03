import glob
import json

# import librosa
# import librosa.display
# import torch
import math
import os
import re
import time
import warnings
from timeit import default_timer as timer

# from lfp_helper import *
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg as LA

# from general import *
from kneed import KneeLocator
from scipy import signal
from scipy.signal import butter, filtfilt, find_peaks, hilbert, iirnotch, lfilter, sosfiltfilt

# from spike_localization import patient_localization_mapping
from scipy.stats import zscore

from brain_decoding.config.file_path import DATA_PATH, MOVIE24_LABEL_PATH
from brain_decoding.dataloader.clusterless_clean import (
    cross_chan_binned_clean,
    cross_chan_event_detection,
    load_data_from_bundle,
    sort_file_name,
)
from brain_decoding.param.param_data import MOVIE24_ANNOTATION_FS, PREDICTION_FS, TWILIGHT_ANNOTATION_FS

SECONDS_PER_HOUR = 3600

OFFSET = {
    "555_1": 4.58,
    "562_1": 0,
    "563_1": 25.59,
    "564_1": [(792.79, 1945), (3732.44, 5091)],
    "564_2": [(792.79, 1945), (3732.44, 5091)],
    "565_1": [(917.796, 2490.63), (2692.16, 3599.14 - 43)],
    "565_2": [(3744.36, 4278.64), (4441.95, 6387.16 - 44)],
    "566_1": 1691273171.09898 - 1691272807.193047,
    "567_1": 1692748176.35772 - 1692747794.1907692,
    "568_1": 400.426,
    "570_1": 1706308502.12459 - 1706304396.2999392,
    "572_1": 1711143392.69897 - 1711142763.677046,
    "573_1": 1714775660.18883 - 1714775636.2359192,
    "i717_1": 1699291373.40026 - 1699290962.3770342,
    "i728_1": 1702317343.74561 - 1702316927.5185993,
}

# '566_1': 364.07, #380.5814,
# '567_1': 382.376,
# '568_1': 400.426,
# '570_1': 1706308502.12459 - 1706304396.2999392,
# '572_1': 1711143392.69897 - 1711142763.677046,
# '573_1': 1714775660.18883 - 1714775636.2359192,
# 'i717_1': 1699291373.40026 - 1699290962.3770342,
# 'i728_1': 416.374,
FREE_RECALL_TIME = {
    "555_1": (42 * 60 + 10, 44 * 60 + 19),
    "562_FR1": (43 * 60 + 58, 49 * 60 + 55),
    "562_FR2": (0 * 60 + 20, 5 * 60 + 50),
    "562_3": (1 * 3600 + 1 * 60 + 32, 1 * 3600 + 16 * 60 + 52),  # memory test
    "563_FR1": (42 * 60 + 59, 48 * 60 + 14),
    "563_FR2": (0 * 60 + 49, 5 * 60 + 53),
    "564_1": (1 * 3600 + 30 * 60 + 59, 1 * 3600 + 35 * 60 + 35),
    "564_2": (2 * 60 + 45, 7 * 60 + 48),
    "565_1": (1 * 3600 + 48 * 60 + 55, 1 * 3600 + 53 * 60 + 23),
    "565_2": (8 * 60 + 30, 14 * 60 + 1),
    "565_3": (4 * 60 + 30, 9 * 60 + 35),
    "566_FR1": (48 * 60 + 52, 53 * 60 + 47),
    "566_CR1": (1 * 3600 + 4 * 60 + 30, 1 * 3600 + 13 * 60 + 38),
    "566_FR2": (4 * 60 + 38, 18 * 60 + 25),
    "566_CR2": (29 * 60 + 7, 35 * 60 + 12),
    "566_5": (0, 1 * 3600),
    "566_6": (2 * 60 + 1, 25 * 60 + 20),  # anime
    "566_7": (25 * 60 + 44, 47 * 60 + 5),  # lion
    "567_FR1": (48 * 60 + 25, 52 * 60 + 38),
    "567_CR1": (1 * 3600 + 4 * 60 + 44, 1 * 3600 + 10 * 60 + 14),
    "567_FR2": (2 * 60 + 31, 8 * 60 + 0),
    "567_CR2": (19 * 60 + 27, 24 * 60 + 15),
    "567_Ctrl1": (0 * 60 + 20, 10 * 60 + 20),  # yoda
    "567_Ctrl2": (1 * 60 + 0, 41 * 60 + 0),  # house
    "567_Ctrl1R2": (28 * 60 + 22, 30 * 60 + 20),  # yoda post sleep recall
    "567_Ctrl2R1": (44 * 60 + 2, 47 * 60 + 37),  # house pre sleep recall
    "567_Ctrl2R2": (24 * 60 + 38, 28 * 60 + 2),  # house post sleep recall
    "568_FR1": (49 * 60 + 58, 52 * 60 + 37),
    "568_CR1": (1 * 3600 + 25 * 60 + 0, 1 * 3600 + 31 * 60 + 7),
    "568_FR2": (0 * 60 + 37, 5 * 60 + 44),
    "568_CR2": (19 * 60 + 23, 25 * 60 + 58),
    "568_Ctrl1": (1 * 60 + 0, 44 * 60 + 0),  # nature
    "568_Ctrl1R2": (0 * 60 + 14, 4 * 60 + 29),  # nature post sleep recall
    "570_FR1": (51 * 60 + 33, 58 * 60 + 5),
    "570_CR1": (1 * 3600 + 9 * 60 + 52, 1 * 3600 + 15 * 60 + 45),
    "570_FR2": (1 * 60 + 18, 6 * 60 + 33),
    "570_CR2": (17 * 60 + 45, 23 * 60 + 35),
    "572_FR1": (0 * 60 + 28, 4 * 60 + 30),
    "572_CR1": (16 * 60 + 48, 22 * 60 + 30),
    "572_FR2": (1 * 60 + 22, 5 * 60 + 56),
    "572_CR2": (17 * 60 + 53, 22 * 60 + 31),
    "573_FR1": (0 * 60 + 38, 5 * 60 + 55),
    "573_CR1": (25 * 60 + 44, 31 * 60 + 35),
    "573_FR2": (0 * 60 + 34, 8 * 60 + 30),
    "573_CR2": (20 * 60 + 30, 26 * 60 + 50),
    "i717_FR1": (1 * 60 + 20, 5 * 60 + 9),
    "i717_CR1": (40 * 60 + 36, 45 * 60 + 54),
    "i717_FR2": (2 * 60 + 42, 7 * 60 + 11),
    "i717_CR2": (39 * 60 + 48, 45 * 60 + 20),
    "i728_FR1a": (48 * 60 + 36, 55 * 60 + 45),
    "i728_FR1b": (1 * 60 + 52, 10 * 60 + 0),
    "i728_CR1": (26 * 60 + 58, 34 * 60 + 58),
    "i728_FR2": (0 * 60 + 46, 11 * 60 + 21),
    "i728_CR2": (22 * 60 + 3, 27 * 60 + 33),
    "i728_Ctrl1": (1 * 60 + 0, 13 * 60 + 0),  # control1
    "i728_Ctrl2": (5 * 60 + 0, 27 * 60 + 0),  # control2
    "i728_Ctrl1R1": (13 * 60 + 43, 15 * 60 + 0),
    "i728_Ctrl2R1": (28 * 60 + 38, 36 * 60 + 15),
}

SLEEP_TIME = (0, 10 * SECONDS_PER_HOUR)

CONTROL = {
    "566": [(121, 1520), (1544, 2825)],
}

TWILIGHT_TIME = {
    "567": (),
    "570": (1706304432.076939 - 1706304396.2999392, 45 * 60 + 1706304432.076939 - 1706304396.2999392),
}

MOVIE24_TIME = {
    "565": (),
    "566": (1691273171.2970471 - 1691272807.193047, 45 * 60 + 1691273171.2970471 - 1691272807.193047),
    "567": (1692748176.5677693 - 1692747794.1907692, 45 * 60 + 1692748176.5677693 - 1692747794.1907692),
    "570": (4105.921, 45 * 60 + 4105.921),
}

SCREENING_TIME = {
    "570": (0, 40 * 60),
}


def construct_movie_wf(spike_file, patient_number, category, phase):
    data = np.load(spike_file)
    json_path = spike_file.replace(".npz", ".json")
    with open(json_path, "r") as json_file:
        metadata = json.load(json_file)

    n_samples = metadata["params"]["n_samples"]
    win = int(metadata["params"]["waveform_win_length"])

    wf_1d = np.zeros(n_samples)
    wf = data["waveform"]
    indices = data["idx"]
    sf = int(metadata["params"]["SR_Hz"])

    for i, idx in enumerate(indices):
        i1 = idx - win // 2
        i2 = idx + win // 2
        if i1 < 0 or i2 > n_samples:
            print("skip some")
            continue
        if np.any(np.isnan(wf[i])):
            print("{} contains NaN values.".format(os.path.split(spike_file)[-1]))
            continue
        wf_1d[i1:i2] = np.abs(np.min(wf[i]))

    movie_label = np.load(MOVIE24_LABEL_PATH)
    if category == "movie" and isinstance(OFFSET[patient_number + "_" + str(phase)], list):
        if patient_number == "565":
            movie_sample_range = []
            num_samples = 0
            for alignment_offset in OFFSET[patient_number + "_" + str(phase)]:
                sample_range = [alignment_offset[0] * sf, alignment_offset[1] * sf]
                movie_sample_range.append(sample_range)
                num_samples += int((sample_range[1] - sample_range[0]) // sf * PREDICTION_FS)
        else:
            alignment_offset = OFFSET[patient_number + "_" + str(phase)][phase - 1]
            movie_sample_range = [alignment_offset[0] * sf, alignment_offset[1] * sf]
            num_samples = int((movie_sample_range[1] - movie_sample_range[0]) / sf * PREDICTION_FS)
    else:
        if category == "movie_24":
            start = MOVIE24_TIME[patient_number][0]
            end = MOVIE24_TIME[patient_number][1]
            movie_sample_range = [start * sf, end * sf]
            num_samples = int((end - start) * PREDICTION_FS)
        elif category == "control":
            alignment_offset = 0
            # control_length = min(z.shape[-1], movie_label.shape[-1] * sf)
            # movie_sample_range = [alignment_offset * sf, alignment_offset * sf + control_length]
            control_length = 250 * sf
            movie_sample_range = [150 * sf, 400 * sf]
            num_samples = control_length / sf * PREDICTION_FS
        elif category == "recall":
            alignment_offset = 0
            recall_start = FREE_RECALL_TIME[patient_number + "_" + str(phase)][0]
            recall_end = FREE_RECALL_TIME[patient_number + "_" + str(phase)][1]
            movie_sample_range = [(alignment_offset + recall_start) * sf, (alignment_offset + recall_end) * sf]
            num_samples = int((movie_sample_range[1] - movie_sample_range[0]) / sf * PREDICTION_FS)
        elif category == "twilight":
            start = TWILIGHT_TIME[patient_number][0]
            end = TWILIGHT_TIME[patient_number][1]
            movie_sample_range = [start * sf, end * sf]
            num_samples = int((end - start) * PREDICTION_FS)
        else:
            raise ValueError("undefined category: {category}")

    if patient_number == "565" and category == "movie":
        movie_wf = []
        for i, (s, e) in enumerate(movie_sample_range):
            movie_wf.append(wf_1d[int(s) : int(e)])
        movie_wf = np.concatenate(movie_wf, axis=0)
    else:
        movie_wf = wf_1d[int(movie_sample_range[0]) : int(movie_sample_range[1])]

    return movie_wf, num_samples, sf


def get_sleep(patient_number, desired_samplerate, mode):
    """
    {0: exp5, 1: exp6, 2: exp7}.
    since we agree to maintain each experiment individually, no longer need this 'phase' parameter
    """
    spike_path = "E://projects//Datasets//neuron//spike_data//{}//raw_{}//".format(patient_number, mode)
    spike_files = glob.glob(os.path.join(spike_path, "*.npz"))
    spike_files = sorted(spike_files, key=sort_file_name)
    for _, file in enumerate(spike_files):
        save_folder = "E://projects//Datasets//neuron//spike_data//{}//time_{}//".format(patient_number, mode)
        save_path = os.path.join(save_folder, os.path.split(file)[-1])
        os.makedirs(save_folder, exist_ok=True)
        data = np.load(file)
        json_path = file.replace(".npz", ".json")
        with open(json_path, "r") as json_file:
            metadata = json.load(json_file)

        n_samples = metadata["params"]["n_samples"]
        win = int(metadata["params"]["waveform_win_length"])

        wf_1d = np.zeros(n_samples)
        sf = int(metadata["params"]["SR_Hz"])
        wf = data["waveform"]
        indices = data["idx"]
        for i, idx in enumerate(indices):
            i1 = idx - win // 2
            i2 = idx + win // 2
            if i1 < 0 or i2 > n_samples:
                print("skip some")
                continue
            if np.any(np.isnan(wf[i])):
                print("{} contains NaN values.".format(os.path.split(file)[-1]))
                continue
            wf_1d[i1:i2] = wf[i]

        num_hours = len(wf_1d) // sf // SECONDS_PER_HOUR
        final_spike_data = []
        for h in range(num_hours):
            movie_wf = wf_1d[h * sf * SECONDS_PER_HOUR : (h + 1) * sf * SECONDS_PER_HOUR]
            num_samples = SECONDS_PER_HOUR * PREDICTION_FS
            for second in range(num_samples):
                window_left = second / PREDICTION_FS * sf
                window_right = (second + 1) / PREDICTION_FS * sf
                if window_left < 0 or window_right > movie_wf.shape[-1]:
                    continue
                features = movie_wf[int(window_left) : int(window_right)]
                # features = get_short(features)

                features = features.reshape(features.shape[0] // 4, 4)
                features = np.mean(features, axis=1)

                if np.any(np.isnan(features)):
                    print("{} contains NaN values. Fatal!!".format(os.path.split(file)[-1]))
                final_spike_data.append(features)

        final_spike_data = np.array(final_spike_data, dtype=np.float32)
        np.savez(save_path, data=final_spike_data)
        print(os.path.split(file)[-1])


def get_ready(patient_number, desired_samplerate, mode, category="recall", phase=-1):
    spike_path = "E://projects//Datasets//neuron//spike_data//{}//raw_{}//".format(patient_number, mode)
    spike_files = glob.glob(os.path.join(spike_path, "*.npz"))
    spike_files = sorted(spike_files, key=sort_file_name)

    for _, file in enumerate(spike_files):
        save_folder = "E://projects//Datasets//neuron//spike_data//{}//time_{}{}_{}//".format(
            patient_number, category, phase, mode
        )
        save_path = os.path.join(save_folder, os.path.split(file)[-1])
        os.makedirs(save_folder, exist_ok=True)
        data = np.load(file)
        json_path = file.replace(".npz", ".json")
        with open(json_path, "r") as json_file:
            metadata = json.load(json_file)

        n_samples = metadata["params"]["n_samples"]
        win = int(metadata["params"]["waveform_win_length"])

        wf_1d = np.zeros(n_samples)
        wf = data["waveform"]
        indices = data["idx"]
        sf = int(metadata["params"]["SR_Hz"])

        high_sos = signal.butter(4, [300, 3000], "bandpass", output="sos", fs=32000)
        wf = signal.sosfiltfilt(high_sos, wf)
        notch_freq = np.arange(300, 3001, 60)
        notch = [signal.butter(4, (f - 2, f + 2), "bandstop", output="sos", fs=32000) for f in notch_freq]
        for sos in notch:
            wf = signal.sosfiltfilt(sos, wf)

        movie_label = np.load(MOVIE24_LABEL_PATH)
        # movie_label = np.repeat(movie_label, resolution, axis=1)
        if category == "recall":
            alignment_offset = 0
            movie_sample_range = [
                (alignment_offset + FREE_RECALL_TIME["566_4"][0]) * sf,
                (alignment_offset + FREE_RECALL_TIME["566_4"][1]) * sf,
            ]
            num_samples = int(FREE_RECALL_TIME["566_4"][1] - FREE_RECALL_TIME["566_4"][0]) * PREDICTION_FS
        elif category == "anime":
            alignment_offset = 0
            movie_sample_range = [
                (alignment_offset + CONTROL[patient_number][0][0]) * sf,
                (alignment_offset + CONTROL[patient_number][0][1]) * sf,
            ]
            num_samples = int(CONTROL[patient_number][0][1] - CONTROL[patient_number][0][0]) * PREDICTION_FS
        elif category == "lion":
            alignment_offset = 0
            movie_sample_range = [
                (alignment_offset + CONTROL[patient_number][1][0]) * sf,
                (alignment_offset + CONTROL[patient_number][1][1]) * sf,
            ]
            num_samples = int(CONTROL[patient_number][1][1] - CONTROL[patient_number][1][0]) * PREDICTION_FS
        elif category == "control":
            alignment_offset = 1 * 60
            movie_sample_range = [alignment_offset * sf, (alignment_offset + movie_label.shape[-1]) * sf]
            num_samples = movie_label.shape[-1] * PREDICTION_FS
        else:
            alignment_offset = OFFSET[patient_number]  # seconds
            movie_sample_range = [alignment_offset * sf, (alignment_offset + movie_label.shape[-1]) * sf]
            num_samples = movie_label.shape[-1] * PREDICTION_FS
        wf = wf[int(movie_sample_range[0]) : int(movie_sample_range[1])]
        final_spike_data = np.zeros_like(wf)

        # negative_idx = flt_data[channel] < 0
        # negative = flt_data[channel][negative_idx]
        negative = wf
        envelope = hilbert(negative)
        envelope = np.abs(envelope)
        threshold = 3 * np.median(np.abs(envelope)) / 0.6745
        peaks, _ = find_peaks(-wf, distance=80, height=threshold)
        for peak in peaks:
            # start = max(0, peak - 30)
            # end = min(len(wf), peak + 30)
            start = peak - 30
            end = peak + 30

            if start < 0 or end > n_samples:
                print("skip some")
                continue
            if np.any(np.isnan(wf[start:end])):
                print("{} contains NaN values.".format(os.path.split(file)[-1]))
                continue
            final_spike_data[start:end] = wf[start:end]

        final_spike_data = final_spike_data.reshape(-1, 160)
        final_spike_data = -np.min(final_spike_data, axis=-1)

        if np.any(np.isnan(final_spike_data)):
            print("{} contains NaN values.".format(os.path.split(file)[-1]))

        final_spike_data = np.array(final_spike_data, dtype=np.float32)
        np.savez(save_path, data=final_spike_data)
        print(os.path.split(file)[-1])


def get_oneshot_blur(patient_number, desired_samplerate, mode, category="recall", phase=-1):
    spike_path = "E://projects//Datasets//neuron//spike_data//{}//raw_{}//".format(patient_number, mode)
    spike_files = glob.glob(os.path.join(spike_path, "*.npz"))
    spike_files = sorted(spike_files, key=sort_file_name)

    for _, file in enumerate(spike_files):
        save_folder = "E://projects//Datasets//neuron//spike_data//{}//time_{}{}_{}//".format(
            patient_number, category, phase, mode
        )
        save_path = os.path.join(save_folder, os.path.split(file)[-1])
        os.makedirs(save_folder, exist_ok=True)
        data = np.load(file)
        json_path = file.replace(".npz", ".json")
        with open(json_path, "r") as json_file:
            metadata = json.load(json_file)

        n_samples = metadata["params"]["n_samples"]
        win = int(metadata["params"]["waveform_win_length"])

        wf_1d = np.zeros(n_samples)
        wf = data["waveform"]
        indices = data["idx"]
        sf = int(metadata["params"]["SR_Hz"])

        for i, idx in enumerate(indices):
            i1 = idx - win // 2
            i2 = idx + win // 2
            if i1 < 0 or i2 > n_samples:
                print("skip some")
                continue
            if np.any(np.isnan(wf[i])):
                print("{} contains NaN values.".format(os.path.split(file)[-1]))
                continue
            wf_1d[i1:i2] = wf[i]

        movie_label = np.load(MOVIE24_LABEL_PATH)
        # movie_label = np.repeat(movie_label, resolution, axis=1)
        if category == "movie" and isinstance(OFFSET[patient_number + "_" + str(phase)], list):
            if patient_number == "565":
                movie_sample_range = []
                num_samples = 0
                for alignment_offset in OFFSET[patient_number + "_" + str(phase)]:
                    sample_range = [alignment_offset[0] * sf, alignment_offset[1] * sf]
                    movie_sample_range.append(sample_range)
                    num_samples += int((sample_range[1] - sample_range[0]) // sf * PREDICTION_FS)
            else:
                alignment_offset = OFFSET[patient_number + "_" + str(phase)][phase - 1]
                movie_sample_range = [alignment_offset[0] * sf, alignment_offset[1] * sf]
                num_samples = int((movie_sample_range[1] - movie_sample_range[0]) / sf * PREDICTION_FS)
        else:
            if category == "movie":
                alignment_offset = OFFSET[patient_number + "_" + str(phase)]  # seconds
                movie_sample_range = [alignment_offset * sf, (alignment_offset + movie_label.shape[-1]) * sf]
                num_samples = int(movie_label.shape[-1] * PREDICTION_FS)
            elif category == "control":
                alignment_offset = 0
                # control_length = min(z.shape[-1], movie_label.shape[-1] * sf)
                # movie_sample_range = [alignment_offset * sf, alignment_offset * sf + control_length]
                control_length = 250 * sf
                movie_sample_range = [150 * sf, 400 * sf]
                num_samples = control_length / sf * PREDICTION_FS
            elif category == "recall":
                alignment_offset = 0
                recall_start = FREE_RECALL_TIME[patient_number + "_" + str(phase)][0]
                recall_end = FREE_RECALL_TIME[patient_number + "_" + str(phase)][1]
                movie_sample_range = [(alignment_offset + recall_start) * sf, (alignment_offset + recall_end) * sf]
                num_samples = int((movie_sample_range[1] - movie_sample_range[0]) / sf * PREDICTION_FS)
        if patient_number == "565" and category == "movie":
            movie_wf = []
            for i, (s, e) in enumerate(movie_sample_range):
                movie_wf.append(wf_1d[int(s) : int(e)])
            movie_wf = np.concatenate(movie_wf, axis=0)
        else:
            movie_wf = wf_1d[int(movie_sample_range[0]) : int(movie_sample_range[1])]

        final_spike_data = []
        for second in range(num_samples):
            window_left = second / PREDICTION_FS * sf
            window_right = (second + 1) / PREDICTION_FS * sf
            if window_left < 0 or window_right > movie_wf.shape[-1]:
                continue
            features = movie_wf[int(window_left) : int(window_right)]
            # features = get_short(features)
            window_size = 160
            binned_features = []
            for i in range(0, len(features), window_size):
                window = features[i : i + window_size]
                # binned_features.append(np.ptp(window))
                binned_features.append(np.abs(np.min(window)))
            features = np.array(binned_features)

            if np.any(np.isnan(features)):
                print("{} contains NaN values.".format(os.path.split(file)[-1]))
            final_spike_data.append(features)

        final_spike_data = np.array(final_spike_data, dtype=np.float32)
        np.savez(save_path, data=final_spike_data)
        print(os.path.split(file)[-1], np.max(final_spike_data))


def get_exp_range(
    time_window: Tuple[int, int], sampling_frequency: float, annotation_fs: int = 1
) -> Tuple[Tuple[float, float], int]:
    start, end = time_window
    exp_sample_range = (start * sampling_frequency, end * sampling_frequency)
    num_samples = int((end - start) * PREDICTION_FS / annotation_fs)
    return exp_sample_range, num_samples


def get_oneshot_clean(
    patient_id: str, desired_samplerate, mode: str, category: str, phase: int = None, version: str = "notch"
):
    spike_path = f"{SPIKE_ROOT_PATH}/{patient_id}/{mode}/"
    spike_files = glob.glob(os.path.join(spike_path, "*.csv"))
    spike_files = sorted(spike_files, key=sort_file_name)

    for bundle in range(0, len(spike_files), 8):
        bundle_csv = spike_files[bundle : bundle + 8]
        df = load_data_from_bundle(bundle_csv)
        # df_clean = cross_chan_event_detection(df, 2, 4)
        df_clean = cross_chan_binned_clean(df, 3, 4)

        for channel, data in df_clean.groupby("channel"):
            save_folder = f"{DATA_PATH}/{patient_id}/{version}/time_{category}_{phase}/"
            channel_name = re.sub(r"\d+", "", os.path.split(bundle_csv[0])[-1].split(".csv")[0])
            name_last = re.sub(r"\d+", "", os.path.split(bundle_csv[-1])[-1].split(".csv")[0])
            assert channel_name == name_last, "wrong bundle name"

            file = os.path.join(spike_path, f"{channel_name}{channel}.csv")
            save_path = os.path.join(save_folder, f"{channel_name}{channel}.npz")
            os.makedirs(save_folder, exist_ok=True)

            json_path = file.replace(".csv", ".json")
            with open(json_path, "r") as json_file:
                metadata = json.load(json_file)

            n_samples = metadata["params"]["n_samples"]
            sf = int(metadata["params"]["fs_estimate_Hz"])
            wf_2d = np.zeros((1, n_samples))

            data["amplitude"] = -data["amplitude"]
            sd = metadata["params"]["threshold"] / 3
            ratio_list = np.arange(3, 5.1, 0.1)
            ratio_list = np.round(ratio_list, 1)
            peak_num_list = []
            for ratio in ratio_list:
                dd = data[(data["amplitude"] >= sd * ratio) & (data["amplitude"] <= sd * 30)]
                amp = list(dd["amplitude"])
                peak_num_list.append(len(amp))

            cutoff_rr = 3.5
            print(cutoff_rr)
            step = 0.5
            for rr in np.arange(cutoff_rr, 30 + step, step):
                data1 = data[(data["amplitude"] >= sd * rr) & (data["amplitude"] < sd * (rr + step))]
                amplitudes1 = list(data1["amplitude"])
                indices1 = list(data1["index"])
                for i, (idx, amp) in enumerate(zip(indices1, amplitudes1)):
                    wf_2d[0, idx] = np.max([wf_2d[0, idx], rr])

            rr = 30 + step
            data1 = data[data["amplitude"] >= sd * rr]
            amplitudes1 = list(data1["amplitude"])
            indices1 = list(data1["index"])
            for i, (idx, amp) in enumerate(zip(indices1, amplitudes1)):
                wf_2d[0, idx] = np.max([wf_2d[0, idx], 32])

            movie_label = np.load(MOVIE24_LABEL_PATH)
            if category == "movie" and isinstance(OFFSET[patient_id + "_" + str(phase)], list):
                if patient_id == "565":
                    exp_sample_range = []
                    num_samples = 0
                    for alignment_offset in OFFSET[patient_id + "_" + str(phase)]:
                        sample_range = [alignment_offset[0] * sf, alignment_offset[1] * sf]
                        exp_sample_range.append(sample_range)
                        num_samples += int(
                            (sample_range[1] - sample_range[0]) // sf * PREDICTION_FS / MOVIE24_ANNOTATION_FS
                        )
                else:
                    alignment_offset = OFFSET[patient_id + "_" + str(phase)][phase - 1]
                    exp_sample_range = [alignment_offset[0] * sf, alignment_offset[1] * sf]
                    num_samples = int(
                        (exp_sample_range[1] - exp_sample_range[0]) / sf * PREDICTION_FS / MOVIE24_ANNOTATION_FS
                    )
            else:
                if category == "movie_24":
                    exp_sample_range, num_samples = get_exp_range(MOVIE24_TIME[patient_id], sf, MOVIE24_ANNOTATION_FS)
                elif category == "control":
                    alignment_offset = 0
                    # control_length = min(z.shape[-1], movie_label.shape[-1] * sf)
                    # exp_sample_range = [alignment_offset * sf, alignment_offset * sf + control_length]
                    control_length = 250 * sf
                    exp_sample_range = [150 * sf, 400 * sf]
                    num_samples = control_length / sf * PREDICTION_FS
                elif category == "recall":
                    alignment_offset = 0
                    recall_start = FREE_RECALL_TIME[patient_id + "_" + str(phase)][0]
                    recall_end = FREE_RECALL_TIME[patient_id + "_" + str(phase)][1]
                    exp_sample_range = [(alignment_offset + recall_start) * sf, (alignment_offset + recall_end) * sf]
                    num_samples = int((exp_sample_range[1] - exp_sample_range[0]) / sf * PREDICTION_FS)
                elif category == "sleep":
                    exp_sample_range, num_samples = get_exp_range(SLEEP_TIME, sf)
                elif category == "twilight":
                    exp_sample_range, num_samples = get_exp_range(TWILIGHT_TIME[patient_id], sf, TWILIGHT_ANNOTATION_FS)
                else:
                    raise ValueError("undefined category: {category}")

            if patient_id == "565" and category == "movie":
                movie_wf = []
                for i, (s, e) in enumerate(exp_sample_range):
                    movie_wf.append(wf_2d[:, int(s) : int(e)])
                movie_wf = np.concatenate(movie_wf, axis=0)
            else:
                movie_wf = wf_2d[:, int(exp_sample_range[0]) : int(exp_sample_range[1])]

            final_spike_data = []
            for second in range(num_samples):
                window_left = second / PREDICTION_FS * sf
                window_right = (second + 1) / PREDICTION_FS * sf

                """if overlap"""
                if window_left < 0 or window_right > movie_wf.shape[-1]:
                    continue
                features = movie_wf[:, int(window_left) : int(window_right)]
                # features = get_short(features)
                window_size = 160
                binned_features = []
                for i in range(0, features.shape[-1], window_size):
                    window = features[:, i : i + window_size]

                    non_zero_mask = window != 0
                    non_zero_sum = np.sum(window * non_zero_mask, axis=1)
                    non_zero_count = np.count_nonzero(non_zero_mask, axis=1)
                    res = np.divide(
                        non_zero_sum, non_zero_count, out=np.zeros_like(non_zero_sum), where=non_zero_count != 0
                    )
                    # res = [np.max(window)]
                    binned_features.append(res)
                features = np.column_stack(binned_features)
                if np.any(np.isnan(features)):
                    print("{} contains NaN values.".format(os.path.split(file)[-1]))
                final_spike_data.append(features)

            final_spike_data = np.array(final_spike_data, dtype=np.float32)
            num_samples_with_non_negative = np.sum(np.any(final_spike_data != 0, axis=(1, 2)))
            np.savez(save_path, data=final_spike_data)
            print(
                os.path.split(file)[-1],
                np.max(final_spike_data),
                np.min(final_spike_data),
                num_samples_with_non_negative,
            )


def get_oneshot_by_region(patient_number, desired_samplerate, mode, category="recall", phase=None, version="notch"):
    spike_path = f"{SPIKE_ROOT_PATH}/{patient_number}/raw_{mode}/"
    spike_files = glob.glob(os.path.join(spike_path, "*.npz"))
    spike_files = sorted(spike_files, key=sort_file_name)
    # region_map = patient_localization_mapping[patient_number]
    region_map = {}  # patient_localization_mapping not available

    # Create a reverse mapping from index to region
    index_to_region = {}
    for region, indices in region_map.items():
        for index in indices:
            index_to_region[index] = region
    grouped_paths = {region: [] for region in region_map}

    poyo_dict = {}
    for file in spike_files:
        max_magnitude = -np.inf
        index = int(re.search(r"clustless_CSC(\d+)\.npz", file).group(1))
        assert index is not None
        region = index_to_region[index]
        grouped_paths[region].append(file)

        movie_wf, num_samples, sf = construct_movie_wf(file, patient_number, category, phase)
        for second in range(num_samples):
            window_left = second / PREDICTION_FS * sf
            window_right = (second + 1) / PREDICTION_FS * sf
            if window_left < 0 or window_right > movie_wf.shape[-1]:
                continue
            features = movie_wf[int(window_left) : int(window_right)]
            magnitudes, times = np.unique(features, return_index=True)
            magnitudes = np.round(magnitudes)
            times = times / len(features)
            max_magnitude = max(max_magnitude, np.max(magnitudes))

            non_zero_mask = magnitudes != 0
            magnitudes_non_zero = magnitudes[non_zero_mask].tolist() if np.any(non_zero_mask) else []
            times_non_zero = times[non_zero_mask].tolist() if np.any(non_zero_mask) else []

            sp = poyo_dict.setdefault(second, {})
            ch = sp.setdefault(index, {})
            tm = ch.setdefault("timestamps", times_non_zero)
            mg = ch.setdefault("magnitudes", magnitudes_non_zero)
            rg = ch.setdefault("regions", [region] * len(magnitudes_non_zero))

            # tm.append(times_non_zero)
            # mg.append(magnitudes_non_zero)
            # rg.append(region)
        print("max mag: ", max_magnitude)
    save_folder = f"/mnt/SSD2/yyding/Datasets/neuron/spike_data/{patient_number}/{version}/time_{category}_{phase}/"
    os.makedirs(save_folder, exist_ok=True)

    with open(os.path.join(save_folder, "poyo_data.json"), "w") as json_file:
        json.dump(poyo_dict, json_file, indent=4)


if __name__ == "__main__":
    # version = "notch_CAR"
    version = "notch"
    SPIKE_ROOT_PATH = "/Users/XinNiuAdmin/Library/CloudStorage/Box-Box/Vwani_Movie/Clusterless/"

    get_oneshot_clean(
        "570", 2000, f"Experiment4_MovieParadigm_{version}", category="movie_24", phase=1, version=version
    )
