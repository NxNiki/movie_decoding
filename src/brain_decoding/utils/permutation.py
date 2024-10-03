import multiprocessing
import os
import re
import time
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from brain_decoding.config.config import PipelineConfig
from brain_decoding.config.file_path import PATIENTS_FILE_PATH, SURROGATE_FILE_PATH
from brain_decoding.dataloader.load_patients import load_patients
from brain_decoding.dataloader.patients import Experiment
from brain_decoding.param.param_data import LABELS
from brain_decoding.utils.check_free_recall import (
    find_area_above_threshold_yyding,
    find_target_activation_indices,
    get_empirical_concept_ps,
    get_empirical_concept_ps_hoteling,
    get_empirical_concept_ps_yyding,
    hl_envelopes_idx,
    ttest_ind,
    ttest_rel,
)


class Permutate:
    def __init__(self, config: PipelineConfig, phase: Union[str, List[str]], epoch, phase_length=Dict[str, float]):
        if isinstance(phase, str):
            phase = [phase]

        self.config = config
        self.phase = phase
        self.epoch = epoch

        self.recall_windows = Experiment()
        self.surrogate_windows = []
        self.cr_bins = []
        offset = 0
        for i, curr_phase in enumerate(phase):
            patient = str(self.config.experiment["patient"])
            patients = load_patients(patient, PATIENTS_FILE_PATH)
            extra_recall_windows = patients[patient][curr_phase]
            surrogate_windows = load_patients(patient, SURROGATE_FILE_PATH)
            surrogate_windows_cr = surrogate_windows[patient][curr_phase]["annotation"].values

            if i > 0:
                offset = offset + int(phase_length[phase[i - 1]] * 0.25 * 1000)

            self.recall_windows.extend_events(extra_recall_windows.events, offset)
            self.surrogate_windows.extend([cr_item + offset for cr_item in surrogate_windows_cr])
            self.cr_bins.extend([phase_length[curr_phase]])

        if self.config.model["merge_label"]:
            self.merge()

    def merge(self):
        recall_windows = []
        la = self.recall_windows["LA"].values
        ba = self.recall_windows["attacks/bomb/bus/explosion"].values
        wh = self.recall_windows["white house/DC"].values
        cia = self.recall_windows["CIA/FBI"].values
        hostage = self.recall_windows["hostage/exchange/sacrifice"].values
        handcuff = self.recall_windows["handcuff/chair/tied"].values
        jack = self.recall_windows["Jack Bauer"].values
        chloe = self.recall_windows["Chloe"].values
        bill = self.recall_windows["Bill"].values
        fayed = self.recall_windows["Abu Fayed"].values
        amar = self.recall_windows["Ahmed Amar"].values
        president = self.recall_windows["President"].values
        # merge Amar and Fayed
        # terrorist = fayed + amar
        # merge whiltehouse and president
        whitehouse = wh + president
        # merge CIA and Chloe
        cia = cia + chloe
        # No LA, BombAttacks
        recall_windows.append(whitehouse)
        recall_windows.append(cia)
        recall_windows.append(hostage)
        recall_windows.append(handcuff)
        recall_windows.append(jack)
        recall_windows.append(bill)
        recall_windows.append(fayed)
        recall_windows.append(amar)
        self.recall_windows = recall_windows

    def method_john1(self, predictions):
        activations = predictions
        concept_ps, labels = get_empirical_concept_ps(
            activations, self.recall_windows, bins_back=-4, activation_width=4
        )

        # print significant ones, non-sig, and NaN
        sig_idxs = [ii for ii, jj in enumerate(concept_ps) if jj < 0.05]
        nonsig_idxs = [ii for ii, jj in enumerate(concept_ps) if jj >= 0.05]
        nan_idxs = [ii for ii, jj in enumerate(concept_ps) if np.isnan(jj)]
        print("{}Permutation test sig. p-values:{}".format("\033[1m", "\033[0m"))
        [print(labels[ii] + ": " + str(concept_ps[ii])) for ii in sig_idxs]
        print("{}Permutation test nonsig. p-values:{}".format("\033[1m", "\033[0m"))
        [print(labels[ii] + ": " + str(concept_ps[ii])) for ii in nonsig_idxs]
        print("{}Concepts not recalled:{}".format("\033[1m", "\033[0m"))
        print(np.array(labels)[nan_idxs])

        # also return a txt output in the correct folder
        file_path = os.path.join(
            self.config.data["memory_save_path"],
            "epoch{}_free_recall_test_results_JOHN_{}.txt".format(self.epoch, self.phase),
        )
        if os.path.exists(file_path):
            os.remove(file_path)  # remove file before running
        else:
            print(file_path + " does not exist")
        with open(file_path, "a") as f:
            print("Permutation test sig. p-values:", file=f)
            [print(labels[ii] + ": " + str(concept_ps[ii]), file=f) for ii in sig_idxs]
            print("Permutation test nonsig. p-values:", file=f)
            [print(labels[ii] + ": " + str(concept_ps[ii]), file=f) for ii in nonsig_idxs]
            print("Concepts not recalled:", file=f)
            print(np.array(labels)[nan_idxs], file=f)

    def method_john2(self, predictions):
        activations = predictions
        bin_size = 0.25
        permutations = 1000

        time_bins = np.arange(0, len(activations) * bin_size, bin_size)
        max_bin_back = 16
        sig_vector_test = []
        bins_back = np.arange(-16, 1)
        activations_width = [4, 6, 8]
        for concept_i, concept_vocalizations in enumerate(self.recall_windows):  # for each concept
            if len(concept_vocalizations) <= 0:
                sig_vector_test.append(np.nan)
                continue

            target_activations = []
            target_activations_indices = []
            for concept_vocalization in concept_vocalizations:  # get the ranges for each concept mention
                # get the average activation score for each aw window prior to bb

                closest_end = np.abs(time_bins - concept_vocalization / 1000).argmin()
                # grab only those bins before concept mention
                temp_tas = []
                if (
                    closest_end - (np.max(np.abs(bins_back)) + np.max(activations_width)) >= 0
                ):  # if concept too close to beginning skip it
                    for bb in np.abs(bins_back):
                        for aw in activations_width:
                            temp_tas.append(
                                np.mean(
                                    activations[
                                        closest_end - (bb + aw) : closest_end - bb,
                                        concept_i,
                                    ]
                                )
                            )
                            target_activations_indices.extend(
                                np.arange(closest_end - (bb + aw), closest_end - bb)
                            )  # grab all the indices used in the average not just the bb start
                target_activations.append(temp_tas)
            # Create a mask to exclude the specified concept indices
            target_activations_indices = sorted(set(target_activations_indices))
            mask = np.ones(len(activations), dtype=bool)
            mask[target_activations_indices] = False  # remove these idxs from consideration
            start_indices = np.arange(0, np.max(np.abs(bins_back)) + np.max(activations_width) + 1)
            mask[start_indices] = False

            mask_bins = np.where(mask)[0]
            surrogate_activations = []
            for selected_index in mask_bins:
                temp_surrogate = []
                for bb in np.abs(bins_back):
                    for aw in activations_width:
                        if selected_index - (bb + aw) >= 0:
                            temp_act = activations[
                                selected_index - (bb + aw) : selected_index - bb,
                                concept_i,
                            ]
                            temp_surrogate.append(np.mean(temp_act))
                surrogate_activations.append(temp_surrogate)

            target_activations = np.array(list(filter(None, target_activations)))
            surrogate_activations = np.array(surrogate_activations)

            t_results = []
            p_results = []
            for ii in range(surrogate_activations.shape[0]):
                statistic, p_value = ttest_rel(target_activations.mean(0), surrogate_activations[ii])
                # statistic, p_value = mannwhitneyu(target_activations.mean(0), surrogate_activations[ii])
                if np.isnan(statistic).any():
                    continue
                t_results.append(statistic)
                p_results.append(p_value)

            positive_values = [x for x in t_results if x > 0]
            negative_values = [abs(x) for x in t_results if x < 0]
            yyding_s, yyding_p = ttest_ind(positive_values, negative_values)
            sig_vector_test.append(yyding_p)

        # print significant ones, non-sig, and NaN
        sig_idxs = [ii for ii, jj in enumerate(sig_vector_test) if jj < 0.05]
        nonsig_idxs = [ii for ii, jj in enumerate(sig_vector_test) if jj >= 0.05]
        nan_idxs = [ii for ii, jj in enumerate(sig_vector_test) if np.isnan(jj)]
        file_path = os.path.join(
            self.config.data["memory_save_path"],
            "epoch{}_free_recall_test_results_YYDING_{}.txt".format(self.epoch, self.phase),
        )
        if os.path.exists(file_path):
            os.remove(file_path)  # remove file before running
        else:
            print(file_path + " does not exist")
        with open(file_path, "a") as f:
            print("Permutation test sig. p-values:", file=f)
            [print(LABELS[ii] + ": " + str(sig_vector_test[ii]), file=f) for ii in sig_idxs]
            print("Permutation test nonsig. p-values:", file=f)
            [print(LABELS[ii] + ": " + str(sig_vector_test[ii]), file=f) for ii in nonsig_idxs]
            print("Concepts not recalled:", file=f)
            print(np.array(LABELS)[nan_idxs], file=f)

    def method_hotel(self, predictions):
        activations = predictions
        bins_back = np.arange(-16, 1)
        activations_width = [4, 6, 8]
        start_time = time.time()
        concept_Ps, labels = get_empirical_concept_ps_hoteling(
            activations, self.recall_windows, bins_back, activations_width
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Hotel Time: {elapsed_time} seconds")
        # print significant ones, non-sig, and NaN
        sig_idxs = [ii for ii, jj in enumerate(concept_Ps) if jj < 0.05]
        nonsig_idxs = [ii for ii, jj in enumerate(concept_Ps) if jj >= 0.05]
        nan_idxs = [ii for ii, jj in enumerate(concept_Ps) if np.isnan(jj)]
        print("{}Permutation test sig. p-values:{}".format("\033[1m", "\033[0m"))
        [print(labels[ii] + ": " + str(concept_Ps[ii])) for ii in sig_idxs]
        print("{}Permutation test nonsig. p-values:{}".format("\033[1m", "\033[0m"))
        [print(labels[ii] + ": " + str(concept_Ps[ii])) for ii in nonsig_idxs]
        print("{}Concepts not recalled:{}".format("\033[1m", "\033[0m"))
        print(np.array(labels)[nan_idxs])

        file_path = os.path.join(
            self.config.data["memory_save_path"],
            "epoch{}_free_recall_test_results_HOTEL_{}.txt".format(self.epoch, self.phase),
        )
        if os.path.exists(file_path):
            os.remove(file_path)  # remove file before running
        else:
            print(file_path + " does not exist")
        with open(file_path, "a") as f:
            print("Permutation test sig. p-values:", file=f)
            [print(labels[ii] + ": " + str(concept_Ps[ii]), file=f) for ii in sig_idxs]
            print("Permutation test nonsig. p-values:", file=f)
            [print(labels[ii] + ": " + str(concept_Ps[ii]), file=f) for ii in nonsig_idxs]
            print("Concepts not recalled:", file=f)
            print(np.array(labels)[nan_idxs], file=f)

    def method_fdr(self, predictions):
        pass

    def method_pvalue_curve(self, predictions):
        activations = predictions
        bins_back = np.arange(-16, 1)
        activations_width = [4, 6, 8]

        start_time = time.time()
        with multiprocessing.Pool(processes=4) as pool:
            args_list = [
                (activations, self.recall_windows, bb, aw, self.cr_bins) for aw in activations_width for bb in bins_back
            ]
            results = pool.starmap(get_empirical_concept_ps_yyding, args_list)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed Time: {elapsed_time} seconds")

        result_dict = {}
        count = 0
        for aw in activations_width:
            for bb in bins_back:
                result_dict[(bb, aw)] = results[count]
                count += 1

        final_dict = {}
        for (bb, aw), (concept_Ps, labels) in result_dict.items():
            for i in range(len(labels)):
                ll = final_dict.setdefault(labels[i], {})
                aa = ll.setdefault(aw, {})
                aa[bb] = concept_Ps[i]

        fig, axs = plt.subplots(3, 4, figsize=(12, 9))
        axs = axs.flatten()
        for i, (concept, value) in enumerate(final_dict.items()):
            for aw, vv in value.items():
                data = list(dict(sorted(vv.items())).values())
                data = np.array(data)
                if not np.any(np.isnan(data)):
                    axs[i].plot(bins_back, data, label=aw)
                    axs[i].axhline(y=0.05, color=(0.7, 0.7, 0.7), linestyle="dashed")
                    axs[i].set_title(concept)
                    axs[i].legend()
                else:
                    axs[i].plot([])
                    axs[i].set_title(concept)

                axs[i].set_xlim(-17, 1)
                axs[i].set_ylim(0, 1)

                axs[i].set_xticks(np.arange(-16, 1, 4))
        plt.tight_layout()
        save_path = os.path.join(self.config.data["memory_save_path"], "4-6-8")
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(
            save_path,
            "epoch{}_free_recall_pcurve_{}.png".format(self.epoch, self.phase),
        )
        plt.savefig(file_path)
        plt.cla()
        plt.clf()
        plt.close()

        """plot activations map"""
        fig, ax = plt.subplots(figsize=(4, 8))
        heatmap = ax.imshow(predictions, cmap="viridis", aspect="auto", interpolation="none")

        for concept_i, concept_vocalizations in enumerate(self.recall_windows):
            if not len(concept_vocalizations) > 0:
                continue
            for concept_vocalization in concept_vocalizations:
                t = 4 * concept_vocalization / 1000
                ax.axhline(
                    y=t,
                    color="red",
                    linestyle="-",
                    alpha=0.6,
                    xmin=concept_i / len(LABELS),
                    xmax=(concept_i + 1) / len(LABELS),
                )

        cbar = plt.colorbar(heatmap)
        cbar.ax.tick_params(labelsize=10)
        tick_positions = np.arange(0, len(predictions), 15 * 4)  # 15 seconds * 100 samples per second
        tick_labels = [int(pos * 0.25) for pos in tick_positions]
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)
        ax.set_xticks(np.arange(0, predictions.shape[1], 1))
        ax.set_xticklabels(LABELS, rotation=80)

        ax.set_ylabel("Time (s)")
        ax.set_xlabel("Concept")
        patient = str(self.config.experiment["patient"])
        plt.title(f"{patient} {self.phase} predictions")
        plt.tight_layout()
        save_path = os.path.join(self.config.data["memory_save_path"], "activation")
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(
            save_path,
            "epoch{}_free_recall_activations_{}.png".format(self.epoch, self.phase),
        )
        plt.savefig(file_path)
        plt.cla()
        plt.clf()
        plt.close()

    def method_soraya(self, predictions):
        result_df = pd.DataFrame()
        min_vocalizations = 2
        stats = {}
        for n_concept in range(len(LABELS)):
            p_values = []
            analysis_params = {
                "bin_size": 0.25,  # seconds
                "win_range_sec": [-5, 1],
                "rand_trial_separation_sec": 4,  # seconds
                "activation_threshold": 0,
                "threshold_type": "mean",  # "mean", "static", "dynamic_max"
                "n_permutations": 1500,
            }
            activations = predictions
            n_permutations = analysis_params["n_permutations"]
            bin_size = analysis_params["bin_size"]
            win_range_sec = analysis_params["win_range_sec"]
            rand_trial_separation_sec = analysis_params["rand_trial_separation_sec"]

            time = np.arange(0, activations.shape[0], 1) * bin_size  # time vector in seconds
            win_range_bins = [int(x / bin_size) for x in win_range_sec]
            rand_trial_separation_bins = rand_trial_separation_sec / bin_size

            concept = LABELS[n_concept]
            n_bins = np.abs(win_range_bins[1] - win_range_bins[0]) + 1
            activation = activations[:, n_concept]

            thresh = np.mean(activation)

            concept_vocalz_msec = self.recall_windows[n_concept]
            n_vocalizations = len(concept_vocalz_msec)
            n_rand_trials = n_vocalizations

            if n_vocalizations <= min_vocalizations:  # skip if too small vocalizations for this concept
                # print(LABELS[concept_iden]+' did not work')
                p_values.append(np.nan)
            else:
                _, target_activations_indices = find_target_activation_indices(
                    time, concept_vocalz_msec, win_range_bins, end_inclusive=True
                )
                voc_activations = []
                for activation_indices in target_activations_indices:
                    voc_activations.append(activation[activation_indices])

                if len(voc_activations) == 0:
                    p_values.append(np.nan)
                    # print(f"{concept} indices invalid \n")
                    continue
                for time_range in [[-4, 0]]:
                    # determine mean of AUC for real vocalisations
                    voc_auc = []
                    x_vals = np.arange(0, len(voc_activations[0]), 1)
                    range_start = int((time_range[0] - win_range_sec[0]) / bin_size)
                    range_end = int((time_range[1] - win_range_sec[0]) / bin_size)
                    for epoch in voc_activations:
                        area_above_threshold = find_area_above_threshold_yyding(
                            epoch, x_vals, thresh, bin_range=(range_start, range_end)
                        )
                        voc_auc.append(area_above_threshold)
                    mean_voc_auc = np.mean(voc_auc)

                    # mean_acts = np.mean(voc_activations,0)
                    # SE = np.std(voc_activations,0)/np.sqrt(np.shape(voc_activations)[1])
                    # upper_acts = mean_acts + SE
                    # mean_voc_auc = find_area_above_threshold_yyding(upper_acts, x_vals, thresh)

                    # determine mean of AUC for surrogate "trials"
                    surrogate_vocalz_msec = [s for s in self.surrogate_windows if s not in concept_vocalz_msec]
                    _, surrogate_indices = find_target_activation_indices(
                        time, surrogate_vocalz_msec, win_range_bins, end_inclusive=True
                    )
                    surrogate_bins = np.array(surrogate_indices)
                    n_surrogate_vocalizations = len(surrogate_bins)
                    # surrogate_mask = np.ones_like(activation, dtype=bool)
                    # surrogate_mask[target_activations_indices] = False
                    # if len(self.cr_bins) > 0:
                    #     surrogate_mask[self.cr_bins[0]:self.cr_bins[1]] = False
                    # surrogate_bins = np.where(surrogate_mask)[0]  # possible indices

                    mean_rand_trial_auc = []
                    for i in range(n_permutations):
                        # rng = np.random.default_rng(seed=i)  # Set seed for reproducibility
                        # _, random_trial_indices = find_random_trial_indices(
                        #     n_rand_trials,
                        #     surrogate_bins,
                        #     win_range_bins,
                        #     bins_apart_threshold=rand_trial_separation_bins,
                        #     rng=rng,
                        #     with_replacement=False,
                        # )
                        # random_trial_activations = [activation[idx] for idx in random_trial_indices]

                        random_trial_indices = np.random.choice(n_surrogate_vocalizations, n_rand_trials, replace=False)
                        random_trial_activations = [activation[idx] for idx in surrogate_bins[random_trial_indices]]

                        # find mean AUC for random "trial"
                        rand_trial_auc = []
                        for epoch in random_trial_activations:
                            area_above_threshold = find_area_above_threshold_yyding(
                                epoch,
                                x_vals,
                                thresh,
                                bin_range=(range_start, range_end),
                            )
                            rand_trial_auc.append(area_above_threshold)
                        mean_rand_trial_auc.append(np.mean(rand_trial_auc))

                    # find percentile for real AUC in surrogate distribution
                    # first_occurance_of_voc_in_rand_dist = np.abs(np.sort(mean_rand_trial_auc) - mean_voc_auc).argmin()
                    # pct = first_occurance_of_voc_in_rand_dist / n_permutations
                    # # calculate p value for this concept based on surrogate distribution
                    # p_value = 1 - pct

                    p_value = sum(mean_voc_auc < x for x in mean_rand_trial_auc) / len(mean_rand_trial_auc)
                    p_values.append(np.round(p_value, 3))

            # save output in a dataframe
            df = pd.DataFrame(
                {
                    "concept": concept,
                    "n_vocalizations": n_vocalizations,
                    "activation_threshold": thresh,
                    "p_value": f"({', '.join(map(str, p_values))})",
                },
                index=[0],
            )
            # Append the df for the current concept to the result_df
            result_df = pd.concat([result_df, df], ignore_index=True)
            stats[concept] = p_values[0]

        # save summary output
        save_path = os.path.join(self.config.data["memory_save_path"], "soraya")
        os.makedirs(save_path, exist_ok=True)
        save_csv_fp = os.path.join(
            save_path,
            "epoch{}_free_recall_test_results_AUC_{}.csv".format(self.epoch, self.phase),
        )
        result_df.to_csv(save_csv_fp, index=False)

        return stats

    def method_curve_shape(self, predictions):
        plot_bins_before = 20  # how many bins to plot on either side of vocalization
        plot_bins_after = 12
        min_vocalizations = 2
        bin_size = 0.25
        activations = predictions

        for concept_iden, vocalization_times in enumerate(self.recall_windows):
            if len(vocalization_times) <= min_vocalizations:
                # print(LABELS[concept_iden]+' did not work')
                continue

            time_bins = np.arange(0, len(activations) * bin_size, bin_size)  # all the time bins

            temp_activations = []
            for i, vocal_time in enumerate(vocalization_times):  # append activations around each vocalization
                # get bin closest to the vocalization time
                closest_end = np.abs(time_bins - vocal_time / 1000).argmin()

                # make sure you're not at beginning or end
                if plot_bins_before < closest_end < len(time_bins) - plot_bins_after:
                    concept_acts = activations[
                        closest_end - plot_bins_before : closest_end + plot_bins_after,
                        concept_iden,
                    ]
                    temp_activations.append(concept_acts)

            plot_bin_size = 0.25  # actually 0.266 but round
            plot_tick_bin = 1.0

            fig, axs = plt.subplots(1, 1, figsize=(3, 3))
            xr = np.arange(
                -plot_bins_before * plot_bin_size,
                plot_bins_after * plot_bin_size,
                plot_bin_size,
            )

            # plot the average activation in the pre-vocalization windows and its SE
            mean_acts = np.mean(temp_activations, 0)
            SE = np.std(temp_activations, 0) / np.sqrt(np.shape(temp_activations)[1])

            lmin, lmax = hl_envelopes_idx(mean_acts)
            mean_acts_envelope = np.interp(xr, xr[lmax], mean_acts[lmax])

            plt.plot(xr, mean_acts, color="k")
            plt.fill_between(
                xr,
                mean_acts - SE,
                mean_acts + SE,
                color="k",
                alpha=0.3,
                label="_nolegend_",
            )
            # plt.plot(xr, mean_acts_envelope, color='g', linestyle='--', label='Envelope')
            # plot the null activation for this concept and its SE across the whole FR period
            if len(self.cr_bins) > 1:
                mask = np.ones(len(activations), dtype=bool)
                mask[self.cr_bins[0] : self.cr_bins[1]] = False
                mean_concept_act = np.mean(activations[mask, concept_iden])
            else:
                mean_concept_act = np.mean(activations[:, concept_iden])  # get the average activation for this concept
            SE_concept_act = np.std(activations[:, concept_iden]) / np.sqrt(len(activations[:, concept_iden]) - 1)
            mean_acts_null = mean_concept_act * np.ones(len(xr))
            SE_acts_null = SE_concept_act * np.ones(len(xr))
            plt.plot(xr, mean_acts_null, "--", color="k")
            plt.fill_between(
                xr,
                mean_acts_null - SE_acts_null,
                mean_acts_null + SE_acts_null,
                color="b",
                alpha=0.3,
                label="_nolegend_",
            )

            axs.set_xlabel('"' + LABELS[concept_iden] + '" vocalization (s)')
            axs.set_ylabel("Model activation")
            xticks = np.arange(
                -plot_bins_before * plot_bin_size,
                plot_bins_after * plot_bin_size + 0.01,
                plot_tick_bin,
            )
            xlabels = [
                int(xx)
                for xx in np.arange(
                    -plot_bins_before * plot_bin_size,
                    plot_bins_after * plot_bin_size + 0.01,
                    plot_tick_bin,
                )
            ]
            axs.set_xticks(xticks, xlabels)
            axs.set_ylim(0, 0.8)
            axs.spines["right"].set_visible(False)
            axs.spines["top"].set_visible(False)
            axs.annotate(
                "N = " + str(len(vocalization_times)),
                (plot_bins_after * plot_bin_size * 0.5 * 0.75, 0.75),
            )

            # # remove empty subplot(s) if the number of channels is not a multiple of 2
            # if num_channels % 2 != 0:
            #     fig.delaxes(axs[num_channels - 1, 1])

            plt.tight_layout()
            label_without_punctuation = re.sub(r"[^\w\s]", "", LABELS[concept_iden])
            save_path = os.path.join(self.config.data["memory_save_path"], "curves", f"{self.epoch}")
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(
                os.path.join(save_path, f"{label_without_punctuation}.png"),
                bbox_inches="tight",
                dpi=200,
            )
            plt.cla()
            plt.clf()
            plt.close()


if __name__ == "__main__":
    pass
