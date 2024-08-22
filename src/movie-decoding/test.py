import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import time
import wandb
import copy
import json
import multiprocessing
from scipy.signal import hilbert
from statsmodels.stats.multitest import multipletests
import time
from tqdm import tqdm
from utils.meters import *
from models.ensemble import Ensemble
from utils.check_free_recall import *
from utils.initializer import *

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def method_curve_shape(predictions, patient, phase, save_path, use_clusterless=False, use_lfp=False, use_combined=False,alongwith=[], predictions_length={}):
    plot_bins_before = 20 # how many bins to plot on either side of vocalization
    plot_bins_after = 12
    min_vocalizations = 2
    bin_size = 0.25
    activations = predictions
    CR_bins = []
    if 'FR' in phase and any('CR' in element for element in alongwith):
        free_recall_windows_fr = eval('free_recall_windows' + '_' + patient + f'_{phase}')
        free_recall_windows_cr = eval('free_recall_windows' + '_' + patient + f'_{alongwith[0]}')
        surrogate_windows_fr = eval('surrogate_windows' + '_' + patient + f'_{phase}')
        surrogate_windows_cr = eval('surrogate_windows' + '_' + patient + f'_{alongwith[0]}')
        offset = int(predictions_length[phase] * 0.25) * 1000
        # CR_bins = [predictions_length[phase], predictions_length[phase] + predictions_length[alongwith[0]]]
        free_recall_windows = [fr + [cr_item + offset for cr_item in cr]  for fr, cr in zip(free_recall_windows_fr, free_recall_windows_cr)]
        surrogate_windows = surrogate_windows_fr + [cr_item + offset for cr_item in surrogate_windows_cr]
    elif 'FR' in phase and not any('CR' in element for element in alongwith):
        free_recall_windows = eval('free_recall_windows' + '_' + patient + f'_{phase}')
        surrogate_windows = eval('surrogate_windows' + '_' + patient + f'_{phase}')
    else:
        free_recall_windows = eval('free_recall_windows' + '_' + patient + f'_{phase}')
        surrogate_windows = eval('surrogate_windows' + '_' + patient + f'_{phase}')

    temp = []
    la = free_recall_windows[0]
    ba = free_recall_windows[1]
    wh = free_recall_windows[2]
    cia = free_recall_windows[3]
    hostage = free_recall_windows[4]
    handcuff = free_recall_windows[5]
    jack = free_recall_windows[6]
    chloe = free_recall_windows[7]
    bill = free_recall_windows[8]
    fayed = free_recall_windows[9]
    amar = free_recall_windows[10]
    president = free_recall_windows[11]
    # merge Amar and Fayed
    # terrorist = fayed + amar
    # merge whiltehouse and president
    whitehouse = wh + president
    # merge CIA and Chloe
    CIA = cia + chloe
    # No LA, BombAttacks
    temp.append(whitehouse)
    temp.append(CIA)
    temp.append(hostage)
    temp.append(handcuff)
    temp.append(jack)
    temp.append(bill)
    temp.append(fayed)
    temp.append(amar)
    free_recall_windows = temp

    for concept_iden,vocalization_times in enumerate(free_recall_windows):
        if len(vocalization_times) <= min_vocalizations:
                # print(LABELS[concept_iden]+' did not work')
                continue

        time_bins = np.arange(0,len(activations)*bin_size, bin_size) # all the time bins

        temp_activations = []
        for i,vocal_time in enumerate(vocalization_times): # append activations around each vocalization

            # get bin closest to the vocalization time
            closest_end = np.abs(time_bins-vocal_time/1000).argmin()

            # make sure you're not at beginning or end
            if plot_bins_before < closest_end < len(time_bins) - plot_bins_after: 

                concept_acts = activations[closest_end-plot_bins_before:closest_end+plot_bins_after,concept_iden]
                temp_activations.append(concept_acts)

        plot_bin_size = 0.25 # actually 0.266 but round
        plot_tick_bin = 1.0

        fig, axs = plt.subplots(1,1,figsize=(3,3))
        xr = np.arange(-plot_bins_before*plot_bin_size,plot_bins_after*plot_bin_size,plot_bin_size)

        # plot the average activation in the pre-vocalization windows and its SE
        mean_acts = np.mean(temp_activations,0)
        SE = np.std(temp_activations,0)/np.sqrt(np.shape(temp_activations)[1])
        
        # lmin, lmax = hl_envelopes_idx(mean_acts)
        # mean_acts_envelope = np.interp(xr, xr[lmax], mean_acts[lmax])

        plt.plot(xr,mean_acts,color='black')
        plt.fill_between(xr, mean_acts-SE, mean_acts+SE, color='gray', alpha=0.4, label='_nolegend_')
        # plt.plot(xr, mean_acts_envelope, color='g', linestyle='--', label='Envelope')
        # plot the null activation for this concept and its SE across the whole FR period
        if len(CR_bins) > 0:
            mask = np.ones(len(activations), dtype=bool)
            mask[CR_bins[0]: CR_bins[1]] = False
            mean_concept_act = np.mean(activations[mask,concept_iden])
        else:
            mean_concept_act = np.mean(activations[:,concept_iden]) # get the average activation for this concept
        SE_concept_act = np.std(activations[:,concept_iden])/np.sqrt(len(activations[:,concept_iden])-1)            
        mean_acts_null = mean_concept_act*np.ones(len(xr))
        SE_acts_null = SE_concept_act*np.ones(len(xr))            
        plt.plot(xr, mean_acts_null, '--', color='darkblue', alpha=0.8) 
        # plt.fill_between(xr, mean_acts_null-SE_acts_null, mean_acts_null+SE_acts_null,
        #                         color='b', alpha = 0.3, label='_nolegend_')

        axs.set_xlabel('"'+LABELS[concept_iden]+'" vocalization (s)')
        axs.set_ylabel('Model activation')
        xticks = np.arange(-plot_bins_before*plot_bin_size,plot_bins_after*plot_bin_size+0.01,plot_tick_bin)
        xlabels = [int(xx) for xx in np.arange(-plot_bins_before*plot_bin_size, plot_bins_after*plot_bin_size+0.01,plot_tick_bin)]
        axs.set_xticks(xticks,xlabels)
        axs.set_ylim(0, 1.0)
        axs.spines['right'].set_visible(False)
        axs.spines['top'].set_visible(False)
        axs.annotate('N = '+str(len(vocalization_times)),(plot_bins_after*plot_bin_size*0.5*0.75, 0.75))

        # # remove empty subplot(s) if the number of channels is not a multiple of 2
        # if num_channels % 2 != 0:
        #     fig.delaxes(axs[num_channels - 1, 1])

        plt.tight_layout()
        label_without_punctuation = re.sub(r'[^\w\s]','',LABELS[concept_iden])
        fig.savefig(os.path.join(save_path, f'{label_without_punctuation}.png'),
            bbox_inches='tight', dpi=200)
        plt.cla()
        plt.clf()   
        plt.close()


def method_soraya(predictions, patient, phase, save_path, use_clusterless=False, use_lfp=False, use_combined=False, alongwith=[], predictions_length={}):
    result_df = pd.DataFrame()
    min_vocalizations = 2
    p_stats = {}
    s_stats = {}
    CR_bins = []
    if 'FR' in phase and any('CR' in element for element in alongwith):
        free_recall_windows_fr = eval('free_recall_windows' + '_' + patient + f'_{phase}')
        free_recall_windows_cr = eval('free_recall_windows' + '_' + patient + f'_{alongwith[0]}')
        surrogate_windows_fr = eval('surrogate_windows' + '_' + patient + f'_{phase}')
        surrogate_windows_cr = eval('surrogate_windows' + '_' + patient + f'_{alongwith[0]}')
        offset = int(predictions_length[phase] * 0.25) * 1000
        CR_bins = [predictions_length[phase], predictions_length[phase] + predictions_length[alongwith[0]]]
        free_recall_windows = [fr + [cr_item + offset for cr_item in cr]  for fr, cr in zip(free_recall_windows_fr, free_recall_windows_cr)]
        surrogate_windows = surrogate_windows_fr + [cr_item + offset for cr_item in surrogate_windows_cr]
    elif 'FR' in phase and not any('CR' in element for element in alongwith):
        free_recall_windows = eval('free_recall_windows' + '_' + patient + f'_{phase}')
        surrogate_windows = eval('surrogate_windows' + '_' + patient + f'_{phase}')
    else:
        free_recall_windows = eval('free_recall_windows' + '_' + patient + f'_{phase}')
        surrogate_windows = eval('surrogate_windows' + '_' + patient + f'_{phase}')

    temp = []
    la = free_recall_windows[0]
    ba = free_recall_windows[1]
    wh = free_recall_windows[2]
    cia = free_recall_windows[3]
    hostage = free_recall_windows[4]
    handcuff = free_recall_windows[5]
    jack = free_recall_windows[6]
    chloe = free_recall_windows[7]
    bill = free_recall_windows[8]
    fayed = free_recall_windows[9]
    amar = free_recall_windows[10]
    president = free_recall_windows[11]
    # merge Amar and Fayed
    # terrorist = fayed + amar
    # merge whiltehouse and president
    whitehouse = wh + president
    # merge CIA and Chloe
    CIA = cia + chloe
    # No LA, BombAttacks
    temp.append(whitehouse)
    temp.append(CIA)
    temp.append(hostage)
    temp.append(handcuff)
    temp.append(jack)
    temp.append(bill)
    temp.append(fayed)
    temp.append(amar)
    free_recall_windows = temp

    for n_concept in range(len(LABELS)):
        p_values = []
        s_scores = []
        analysis_params = {
            "bin_size": 0.25,  # seconds
            "win_range_sec": [-5, 1],
            "rand_trial_separation_sec": 4,  # seconds
            "activation_threshold": 0,
            "threshold_type": "mean",  # "mean", "static", "dynamic_max"
            "penalize_sub_threshold": True,
            "n_permutations": 1500,
        }
        activations = predictions
        n_permutations = analysis_params['n_permutations']
        bin_size = analysis_params['bin_size']
        win_range_sec = analysis_params['win_range_sec']
        rand_trial_separation_sec = analysis_params['rand_trial_separation_sec']

        time = np.arange(0, activations.shape[0], 1) * bin_size  # time vector in seconds
        win_range_bins = [int(x / bin_size) for x in win_range_sec]
        rand_trial_separation_bins = rand_trial_separation_sec / bin_size

        concept = LABELS[n_concept]
        n_bins = np.abs(win_range_bins[1] - win_range_bins[0]) + 1
        activation = activations[:, n_concept]

        thresh = np.mean(activation)

        concept_vocalz_msec = free_recall_windows[n_concept]
        n_vocalizations = len(concept_vocalz_msec)
        n_rand_trials = n_vocalizations

        if n_vocalizations <= min_vocalizations:  # skip if no vocalizations for this concept
            p_values.append(np.nan)
            s_scores.append(np.nan)
        else:
            _, target_activations_indices = find_target_activation_indices(
                time, concept_vocalz_msec, win_range_bins, end_inclusive=True
            )
            voc_activations = []
            for activation_indices in target_activations_indices:
                voc_activations.append(activation[activation_indices])

            if len(voc_activations) == 0:
                p_values.append(np.nan)
                s_scores.append(np.nan)
                # print(f"{concept} indices invalid \n")
                continue
            for time_range in [[-4, 0]]:
                # determine mean of AUC for real vocalisations
                voc_auc = []
                x_vals = np.arange(0, len(voc_activations[0]), 1)
                range_start = int((time_range[0] - win_range_sec[0]) / bin_size)
                range_end = int((time_range[1] - win_range_sec[0]) / bin_size)
                for epoch in voc_activations:
                    area_above_threshold = find_area_above_threshold_yyding(epoch, x_vals, thresh, bin_range=(range_start, range_end))
                    voc_auc.append(area_above_threshold)
                mean_voc_auc = np.mean(voc_auc)

                # mean_acts = np.mean(voc_activations,0)
                # SE = np.std(voc_activations,0)/np.sqrt(np.shape(voc_activations)[1])
                # upper_acts = mean_acts + SE
                # mean_voc_auc = find_area_above_threshold_yyding(upper_acts, x_vals, thresh)

                # determine mean of AUC for surrogate "trials"
                surrogate_vocalz_msec = [s for s in surrogate_windows if s not in concept_vocalz_msec]
                _, surrogate_indices = find_target_activation_indices(
                    time, surrogate_vocalz_msec, win_range_bins, end_inclusive=True
                )
                surrogate_bins = np.array(surrogate_indices)
                n_surrogate_vocalizations = len(surrogate_bins)

                # surrogate_mask = np.ones_like(activation, dtype=bool)
                # surrogate_mask[target_activations_indices] = False
                # if len(CR_bins) > 0:
                #     surrogate_mask[CR_bins[0]:CR_bins[1]] = False
                # surrogate_bins = np.where(surrogate_mask)[0]  # possible indices

                mean_rand_trial_auc = []
                np.random.seed(42)
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
                    random_trial_indices = np.random.choice(n_surrogate_vocalizations, n_rand_trials, replace=False)
                    random_trial_activations = [activation[idx] for idx in surrogate_bins[random_trial_indices]]

                    # find mean AUC for random "trial"
                    rand_trial_auc = []
                    for epoch in random_trial_activations:
                        area_above_threshold = find_area_above_threshold_yyding(epoch, x_vals, thresh, bin_range=(range_start, range_end))
                        rand_trial_auc.append(area_above_threshold)
                    mean_rand_trial_auc.append(np.mean(rand_trial_auc))

                # find percentile for real AUC in surrogate distribution
                # first_occurance_of_voc_in_rand_dist = np.abs(np.sort(mean_rand_trial_auc) - mean_voc_auc).argmin()
                # pct = first_occurance_of_voc_in_rand_dist / n_permutations
                # # calculate p value for this concept based on surrogate distribution
                # p_value = 1 - pct

                p_value = sum(mean_voc_auc < x for x in mean_rand_trial_auc) / len(mean_rand_trial_auc)
                p_values.append(np.round(p_value, 3))
                s_scores.append(np.round(mean_voc_auc, 3))

        # save output in a dataframe
        df = pd.DataFrame(
            {
                "concept": concept,
                "n_vocalizations": n_vocalizations,
                "activation_threshold": thresh,
                "p_value": f"({', '.join(map(str, p_values))})",
                "s_score": f"({', '.join(map(str, s_scores))})",
            },
            index=[0],
        )
        # Append the df for the current concept to the result_df
        result_df = pd.concat([result_df, df], ignore_index=True)
        p_stats[concept] = p_values[0]
        s_stats[concept] = s_scores[0]
    
    # save summary output
    # save_csv_fp = os.path.join(self.config['memory_save_path'], 'epoch{}_free_recall_test_results_AUC_{}.csv'.format(self.epoch, self.phase))
    result_df.to_csv(os.path.join(save_path, 'AUC.csv'), index=False)
    overall_p = list(p_stats.values())
    overall_s = list(s_stats.values())
    print('P: ', overall_p)
    print('S: ', overall_s)

def method_fdr(predictions, patient, phase, save_path, use_clusterless=False, use_lfp=False, use_combined=False, alongwith=[], predictions_length={}):
    activations = predictions
    CR_bins = []
    if phase == 'FR1' and 'CR1' in alongwith:
        free_recall_windows1 = eval('free_recall_windows' + '_' + patient + '_FR1')
        offset = int(predictions_length[phase] * 0.25) * 1000
        CR_bins = [predictions_length['FR1'], predictions_length['FR1'] + predictions_length['CR1']]
        free_recall_windows2 = eval('free_recall_windows' + '_' + patient + '_CR1')
        free_recall_windows = [fr1 + [cr1_item + offset for cr1_item in cr1]  for fr1, cr1 in zip(free_recall_windows1, free_recall_windows2)]
    elif phase == 'FR1a' and 'FR1b' in alongwith and 'CR1' not in alongwith:
        free_recall_windows1 = eval('free_recall_windows' + '_' + patient + '_FR1a')
        offset = int(predictions_length[phase] * 0.25) * 1000
        free_recall_windows2 = eval('free_recall_windows' + '_' + patient + '_FR1b')
        free_recall_windows = [fr1 + [cr1_item + offset for cr1_item in cr1]  for fr1, cr1 in zip(free_recall_windows1, free_recall_windows2)]
    elif phase == 'FR1a' and 'FR1b' in alongwith and 'CR1' in alongwith:
        free_recall_windows1 = eval('free_recall_windows' + '_' + patient + '_FR1a')
        offset = eval('offset_{}'.format(patient))
        offset = int(predictions_length['FR1a'] * 0.25) * 1000
        free_recall_windows2 = eval('free_recall_windows' + '_' + patient + '_FR1b')
        free_recall_windows = [fr1 + [cr1_item + offset for cr1_item in cr1]  for fr1, cr1 in zip(free_recall_windows1, free_recall_windows2)]
        offset += int(predictions_length['FR1b'] * 0.25) * 1000
        free_recall_windows3 = eval('free_recall_windows' + '_' + patient + '_CR1')
        free_recall_windows = [fr1 + [cr1_item + offset for cr1_item in cr1]  for fr1, cr1 in zip(free_recall_windows, free_recall_windows3)]
        CR_bins = [predictions_length['FR1a'] + predictions_length['FR1b'], predictions_length['FR1a'] + predictions_length['FR1b'] + predictions_length['CR1']]
    else:
        free_recall_windows = eval('free_recall_windows' + '_' + patient + f'_{phase}')

    bins_back = np.arange(-16, 1)
    activations_width = [4, 6, 8]

    start_time = time.time()
    with multiprocessing.Pool(processes=4) as pool:
        args_list = [(activations, free_recall_windows, bb, aw, CR_bins) for aw in activations_width for bb in bins_back]
        results = pool.starmap(getEmpiricalConceptPs_yyding, args_list)
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
        corrected_p_values = np.zeros((len(activations_width), len(bins_back)))

        for j, (aw, vv) in enumerate(value.items()):
            data = list(dict(sorted(vv.items())).values())
            data = np.array(data)

            # FDR
            significant, corrected_p_value, _, _ = multipletests(data, alpha=0.05, method='fdr_bh')
            corrected_p_values[j] = corrected_p_value

        # corrected_p_values = corrected_p_values.min(axis=0)
        corrected_p_values = corrected_p_values[0]
        if not np.any(np.isnan(data)):
            axs[i].plot(bins_back, corrected_p_values, label='FDR')
            axs[i].axhline(y=0.05, color=(0.7,0.7,0.7),linestyle='dashed')
            axs[i].set_title(concept)
            axs[i].legend()
        else:
            axs[i].plot([])
            axs[i].set_title(concept)
        
        axs[i].set_xlim(-17, 1)
        axs[i].set_ylim(0, 1)

        axs[i].set_xticks(np.arange(-16, 1, 4))

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'FDR.png'),
                    bbox_inches='tight', dpi=200)
    plt.cla()
    plt.clf()
    plt.close()

def perform_memory_test(config, phase='recall1', alongwith=[]):
    model_architecture = config['model_architecture']
    use_clusterless = config['use_clusterless']
    use_lfp = config['use_lfp']
    use_combined = config['use_combined']
    use_overlap = config['use_overlap']
    model_name = config['model_name']
    patient = config['patient']
    epoch = config['epoch']

    args = initialize_configs(architecture=model_architecture)
    args['seed'] = 42
    args['patient'] = patient
    args['device'] = 'cuda:2'
    args['use_spike'] = use_clusterless
    args['use_lfp'] = use_lfp
    args['use_combined'] = use_combined
    args['use_spontaneous'] = False
    if use_clusterless:
        args['use_shuffle'] = True
    elif use_lfp:
        args['use_shuffle'] = False

    args['use_bipolar'] = False
    args['use_sleep'] = False
    args['use_overlap'] = use_overlap
    args['free_recall_phase'] = phase
    args['model_architecture'] = model_architecture

    args['spike_data_mode'] = config['spike_data_mode']
    args['spike_data_mode_inference'] = config['spike_data_mode_inference']
    args['spike_data_sd'] = config['spike_data_sd']
    args['spike_data_sd_inference'] = config['spike_data_sd_inference']
    args['use_augment'] = config['use_augment']
    args['use_long_input'] = config['use_long_input']
    args['model_aggregate_type'] = config['model_aggregate_type']
    args['use_shuffle_diagnostic'] = False

    if config['patient'] == 'i728' and '1' in phase:
        args['free_recall_phase'] = 'FR1a'
        dataloaders = initialize_inference_dataloaders(args)
    else:
        dataloaders = initialize_inference_dataloaders(args)

    root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', model_name, 'memory')
    model = initialize_model(args)
    # model = torch.compile(model)
    model = model.to(args['device'])

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # load the model with best F1-score
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', model_name, 'train', f'model_weights_epoch{epoch}.tar')
    model.load_state_dict(torch.load(model_dir)['model_state_dict'])
    print('Resume model: %s' % model_dir)
    model.eval()

    predictions_all = np.empty((0, args['num_labels']))
    predictions_length = {}
    with torch.no_grad():
        if patient == 'i728' and '1' in phase and 'CR' not in phase:
            for ph in ['FR1a', 'FR1b']:
                predictions = np.empty((0, args['num_labels']))
                args['free_recall_phase'] = ph
                dataloaders = initialize_inference_dataloaders(args)
                for i, (feature, index) in enumerate(dataloaders['inference']):
                    if not args['use_lfp'] and args['use_spike']:
                        spike = feature.to(args['device'])
                        lfp = None
                    elif args['use_lfp'] and not args['use_spike']:
                        lfp = feature.to(args['device'])
                        spike = None
                    else:
                        assert isinstance(feature, list) or isinstance(feature, tuple), "Tensor must be a list or tuple"
                        spike = feature[1].to(args['device'])
                        lfp = feature[0].to(args['device'])
                    spike_emb, lfp_emb, output = model(lfp, spike)
                    output = torch.sigmoid(output)
                    pred = output.cpu().detach().numpy()
                    predictions = np.concatenate([predictions, pred], axis=0)

                if use_overlap:
                    fake_activation = np.mean(predictions, axis=0)
                    predictions = np.vstack((fake_activation, predictions, fake_activation))
                    
                predictions_all = np.concatenate([predictions_all, predictions], axis=0)
            predictions_length[phase] = len(predictions_all)
        else:
            args['free_recall_phase'] = phase
            dataloaders = initialize_inference_dataloaders(args)    
            predictions = np.empty((0, args['num_labels']))
            for i, (feature, index) in enumerate(dataloaders['inference']):
                if not args['use_lfp'] and args['use_spike']:
                    spike = feature.to(args['device'])
                    lfp = None
                elif args['use_lfp'] and not args['use_spike']:
                    lfp = feature.to(args['device'])
                    spike = None
                else:
                    assert isinstance(feature, list) or isinstance(feature, tuple), "Tensor must be a list or tuple"
                    spike = feature[1].to(args['device'])
                    lfp = feature[0].to(args['device'])
                spike_emb, lfp_emb, output = model(lfp, spike)
                output = torch.sigmoid(output)
                pred = output.cpu().detach().numpy()
                predictions = np.concatenate([predictions, pred], axis=0)
            
            if use_overlap:
                fake_activation = np.mean(predictions, axis=0)
                predictions = np.vstack((fake_activation, predictions, fake_activation))

            predictions_length[phase] = len(predictions)
            predictions_all = np.concatenate([predictions_all, predictions], axis=0)

    for ph in alongwith:
        args['free_recall_phase'] = ph
        dataloaders = initialize_inference_dataloaders(args)
        with torch.no_grad():
            # load the best epoch number from the saved "model_results" structure
            predictions = np.empty((0, args['num_labels']))
            # y_true = np.empty((0, self.config.num_labels))
            for i, (feature, index) in enumerate(dataloaders['inference']):
                # target = target.to(self.device)
                if not args['use_lfp'] and args['use_spike']:
                    spike = feature.to(args['device'])
                    lfp = None
                elif args['use_lfp'] and not args['use_spike']:
                    lfp = feature.to(args['device'])
                    spike = None
                else:
                    assert isinstance(feature, list) or isinstance(feature, tuple), "Tensor must be a list or tuple"
                    spike = feature[1].to(args['device'])
                    lfp = feature[0].to(args['device'])
                # forward pass

                # start_time = time.time()
                spike_emb, lfp_emb, output = model(lfp, spike)
                # end_time = time.time()
                # print('inference time: ', end_time - start_time)
                output = torch.sigmoid(output)
                pred = output.cpu().detach().numpy()
                predictions = np.concatenate([predictions, pred], axis=0)
            
            if use_overlap:
                fake_activation = np.mean(predictions, axis=0)
                predictions = np.vstack((fake_activation, predictions, fake_activation))

        predictions_length[ph] = len(predictions)
        predictions_all = np.concatenate([predictions_all, predictions], axis=0)


    smoothed_data = np.zeros_like(predictions_all)
    for i in range(predictions_all.shape[1]):  # Loop through each feature
        smoothed_data[:, i] = np.convolve(predictions_all[:, i], np.ones(4)/4, mode='same')
    predictions = predictions_all

    np.save('epoch{}_free_recall_{}_{}.npy'.format(epoch, phase, patient), predictions_all)
    
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', model_name, 'memory', 'new', f'epoch{epoch}_{phase}_{len(alongwith)}')
    os.makedirs(save_path, exist_ok=True)
    method_curve_shape(smoothed_data, patient, phase, save_path, use_clusterless=use_clusterless, use_lfp=use_lfp, use_combined=use_combined, alongwith=alongwith, predictions_length=predictions_length)
    method_soraya(smoothed_data, patient, phase, save_path, use_clusterless=use_clusterless, use_lfp=use_lfp, use_combined=use_combined, alongwith=alongwith, predictions_length=predictions_length)

def perform_memory_test_with_control(config, phase='recall1'):
    model_architecture = config['model_architecture']
    use_clusterless = config['use_clusterless']
    use_lfp = config['use_lfp']
    use_combined = config['use_combined']
    model_name = config['model_name']
    patient = config['patient']
    epoch = config['epoch']

    args = initialize_configs(architecture=model_architecture)
    args['seed'] = 42
    args['patient'] = patient
    args['device'] = 'cuda:2'
    args['use_spike'] = use_clusterless
    args['use_lfp'] = use_lfp
    args['use_combined'] = use_combined
    args['use_spontaneous'] = False
    if use_clusterless:
        args['use_shuffle'] = True
    elif use_lfp:
        args['use_shuffle'] = False

    args['use_bipolar'] = False
    args['use_sleep'] = False
    args['free_recall_phase'] = phase
    args['model_architecture'] = model_architecture


    root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', model_name, 'memory')

    dataloaders = initialize_inference_dataloaders(args)
    model = initialize_model(args)
    # model = torch.compile(model)
    model = model.to(args['device'])

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # load the model with best F1-score
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', model_name, 'train', f'model_weights_epoch{epoch}.tar')
    model.load_state_dict(torch.load(model_dir)['model_state_dict'])
    print('Resume model: %s' % model_dir)
    model.eval()

    with torch.no_grad():
        # load the best epoch number from the saved "model_results" structure
        predictions = np.empty((0, args['num_labels']))
        # y_true = np.empty((0, self.config.num_labels))
        for i, (feature, index) in enumerate(dataloaders['inference']):
            # target = target.to(self.device)
            if not args['use_lfp'] and args['use_spike']:
                spike = feature.to(args['device'])
                lfp = None
            elif args['use_lfp'] and not args['use_spike']:
                lfp = feature.to(args['device'])
                spike = None
            else:
                assert isinstance(feature, list) or isinstance(feature, tuple), "Tensor must be a list or tuple"
                spike = feature[1].to(args['device'])
                lfp = feature[0].to(args['device'])
            # forward pass

            # start_time = time.time()
            spike_emb, lfp_emb, output = model(lfp, spike)
            # end_time = time.time()
            # print('inference time: ', end_time - start_time)
            output = torch.sigmoid(output)
            pred = output.cpu().detach().numpy()
            predictions = np.concatenate([predictions, pred], axis=0)

    # smoothed_data = np.zeros_like(predictions)
    # for i in range(predictions.shape[1]):  # Loop through each feature
    #     smoothed_data[:, i] = np.convolve(predictions[:, i], np.ones(4)/4, mode='same')
    # predictions = smoothed_data.copy()
    # method_curve_shape(predictions, patient, phase, use_clusterless=use_clusterless, use_lfp=use_lfp)
    # method_soraya(predictions, patient, phase, use_clusterless=use_clusterless, use_lfp=use_lfp)    
    fig, ax = plt.subplots(figsize=(4, 8))
    heatmap = ax.imshow(predictions, cmap='viridis', aspect='auto', interpolation='none')

    cbar = plt.colorbar(heatmap)
    cbar.ax.tick_params(labelsize=10)
    tick_positions = np.arange(0, len(predictions), 15*4)  # 15 seconds * 100 samples per second
    tick_labels = [int(pos * 0.25) for pos in tick_positions]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    ax.set_xticks(np.arange(0, predictions.shape[1], 1))
    ax.set_xticklabels(['LosAngeles', 'BombAttacks', 'Whitehouse', 'CIA',
                        'Hostage', 'Handcuff', 'Jack', 'Chloe', 'Bill',
                        'A.Fayed', 'A.Amar', 'President'], rotation=80)
    
    ax.set_ylabel('Time (s)')
    ax.set_xlabel('Concept')
    plt.title(f'{patient} {phase} predictions')
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', model_name, 'memory', 'control')
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f'epoch{epoch}_free_recall_activations_{phase}.png')
    plt.savefig(file_path)
    plt.cla()
    plt.clf()
    plt.close()

def draw_pvalue_curve(config, phase='recall1', alongwith=[]):
    model_architecture = config['model_architecture']
    use_clusterless = config['use_clusterless']
    use_lfp = config['use_lfp']
    model_name = config['model_name']
    patient = config['patient']
    epoch = config['epoch']

    args = initialize_configs(architecture=model_architecture)
    args['seed'] = 42
    args['patient'] = patient
    args['device'] = 'cuda:2'
    args['use_spike'] = use_clusterless
    args['use_lfp'] = use_lfp
    args['use_spontaneous'] = False
    if use_clusterless:
        args['use_shuffle'] = True
    elif use_lfp:
        args['use_shuffle'] = False

    args['use_bipolar'] = False
    args['use_sleep'] = False
    args['free_recall_phase'] = phase
    args['model_architecture'] = model_architecture
    root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', model_name, 'memory')

    dataloaders = initialize_inference_dataloaders(args)
    model = initialize_model(args)
    # model = torch.compile(model)
    model = model.to(args['device'])

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # load the model with best F1-score
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', model_name, 'train', f'model_weights_epoch{epoch}.tar')
    model.load_state_dict(torch.load(model_dir)['model_state_dict'])
    print('Resume model: %s' % model_dir)
    model.eval()
    
    predictions_all = np.empty((0, args['num_labels']))
    predictions_length = {}
    with torch.no_grad():
        # load the best epoch number from the saved "model_results" structure
        predictions = np.empty((0, args['num_labels']))
        # y_true = np.empty((0, self.config.num_labels))
        for i, (feature, index) in enumerate(dataloaders['inference']):
            # target = target.to(self.device)
            if not args['use_lfp'] and args['use_spike']:
                spike = feature.to(args['device'])
                lfp = None
            elif args['use_lfp'] and not args['use_spike']:
                lfp = {key: value.to(args['device']) for key, value in feature.items()}
                version = args['lfp_data_mode']
                lfp = lfp[version]
                spike = None
            else:
                assert isinstance(feature, list) or isinstance(feature, tuple), "Tensor must be a list or tuple"
                spike = feature[1].to(args['device'])
                lfp = feature[0].to(args['device'])
            # forward pass

            # start_time = time.time()
            spike_emb, lfp_emb, output = model(lfp, spike)
            # end_time = time.time()
            # print('inference time: ', end_time - start_time)
            output = torch.sigmoid(output)
            pred = output.cpu().detach().numpy()
            predictions = np.concatenate([predictions, pred], axis=0)
    predictions_length[phase] = len(predictions)
    predictions_all = np.concatenate([predictions_all, predictions], axis=0)

    for ph in alongwith:
        args['free_recall_phase'] = ph
        dataloaders = initialize_inference_dataloaders(args)
        with torch.no_grad():
            # load the best epoch number from the saved "model_results" structure
            predictions = np.empty((0, args['num_labels']))
            # y_true = np.empty((0, self.config.num_labels))
            for i, (feature, index) in enumerate(dataloaders['inference']):
                # target = target.to(self.device)
                if not args['use_lfp'] and args['use_spike']:
                    spike = feature.to(args['device'])
                    lfp = None
                elif args['use_lfp'] and not args['use_spike']:
                    lfp = {key: value.to(args['device']) for key, value in feature.items()}
                    version = args['lfp_data_mode']
                    lfp = lfp[version]
                    spike = None
                else:
                    assert isinstance(feature, list) or isinstance(feature, tuple), "Tensor must be a list or tuple"
                    spike = feature[1].to(args['device'])
                    lfp = feature[0].to(args['device'])
                # forward pass

                # start_time = time.time()
                spike_emb, lfp_emb, output = model(lfp, spike)
                # end_time = time.time()
                # print('inference time: ', end_time - start_time)
                output = torch.sigmoid(output)
                pred = output.cpu().detach().numpy()
                predictions = np.concatenate([predictions, pred], axis=0)
        predictions_length[ph] = len(predictions)
        predictions_all = np.concatenate([predictions_all, predictions], axis=0)

    activations = predictions_all
    bins_back = np.arange(-16, 1)
    activations_width = [4, 6, 8]

    start_time = time.time()
    CR_bins = []
    if phase == 'FR1' and 'CR1' in alongwith:
        free_recall_windows1 = eval('free_recall_windows' + '_' + patient + '_FR1')
        offset = int(predictions_length[phase] * 0.25) * 1000
        CR_bins = [predictions_length['FR1'], predictions_length['FR1'] + predictions_length['CR1']]
        free_recall_windows2 = eval('free_recall_windows' + '_' + patient + '_CR1')
        free_recall_windows = [fr1 + [cr1_item + offset for cr1_item in cr1]  for fr1, cr1 in zip(free_recall_windows1, free_recall_windows2)]
    elif phase == 'FR1a' and 'FR1b' in alongwith and 'CR1' not in alongwith:
        free_recall_windows1 = eval('free_recall_windows' + '_' + patient + '_FR1a')
        offset = int(predictions_length[phase] * 0.25) * 1000
        free_recall_windows2 = eval('free_recall_windows' + '_' + patient + '_FR1b')
        free_recall_windows = [fr1 + [cr1_item + offset for cr1_item in cr1]  for fr1, cr1 in zip(free_recall_windows1, free_recall_windows2)]
    elif phase == 'FR1a' and 'FR1b' in alongwith and 'CR1' in alongwith:
        free_recall_windows1 = eval('free_recall_windows' + '_' + patient + '_FR1a')
        offset = eval('offset_{}'.format(patient))
        offset = int(predictions_length['FR1a'] * 0.25) * 1000
        free_recall_windows2 = eval('free_recall_windows' + '_' + patient + '_FR1b')
        free_recall_windows = [fr1 + [cr1_item + offset for cr1_item in cr1]  for fr1, cr1 in zip(free_recall_windows1, free_recall_windows2)]
        offset += int(predictions_length['FR1b'] * 0.25) * 1000
        free_recall_windows3 = eval('free_recall_windows' + '_' + patient + '_CR1')
        free_recall_windows = [fr1 + [cr1_item + offset for cr1_item in cr1]  for fr1, cr1 in zip(free_recall_windows, free_recall_windows3)]
        CR_bins = [predictions_length['FR1a'] + predictions_length['FR1b'], predictions_length['FR1a'] + predictions_length['FR1b'] + predictions_length['CR1']]
    else:
        free_recall_windows = eval('free_recall_windows' + '_' + patient + f'_{phase}')
    

    # results = getEmpiricalConceptPs_yyding2(activations, free_recall_windows, -4, 4)
    with multiprocessing.Pool(processes=4) as pool:
        args_list = [(activations, free_recall_windows, bb, aw, CR_bins) for aw in activations_width for bb in bins_back]
        results = pool.starmap(getEmpiricalConceptPs_yyding, args_list)
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
                axs[i].axhline(y=0.05, color=(0.7,0.7,0.7),linestyle='dashed')
                axs[i].set_title(concept)
                axs[i].legend()
            else:
                axs[i].plot([])
                axs[i].set_title(concept)
            
            axs[i].set_xlim(-17, 1)
            axs[i].set_ylim(0, 1)

            axs[i].set_xticks(np.arange(-16, 1, 4))
    plt.tight_layout()
    if len(alongwith) > 0:
        file_path = model_dir.replace('train', 'memory').replace(f'model_weights_epoch{epoch}.tar', f'epoch{epoch}_free_recall_pcurve_{phase}_{"_".join(alongwith)}.png')
    else:
        file_path = model_dir.replace('train', 'memory').replace(f'model_weights_epoch{epoch}.tar', f'epoch{epoch}_free_recall_pcurve_{phase}.png')
    plt.savefig(file_path)
    plt.cla()
    plt.clf()
    plt.close()

def check_avg_score(config, phase='recall1'):
    model_architecture = config['model_architecture']
    use_clusterless = config['use_clusterless']
    use_lfp = config['use_lfp']
    model_name = config['model_name']
    patient = config['patient']
    epoch = config['epoch']

    args = initialize_configs(architecture=model_architecture)
    args['seed'] = 42
    args['patient'] = patient
    args['device'] = 'cuda:2'
    args['use_spike'] = use_clusterless
    args['use_lfp'] = use_lfp
    args['use_spontaneous'] = False
    if use_clusterless:
        args['use_shuffle'] = True
    elif use_lfp:
        args['use_shuffle'] = False

    args['use_bipolar'] = False
    args['use_sleep'] = False
    args['free_recall_phase'] = phase
    args['model_architecture'] = model_architecture


    root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', model_name, 'memory')

    dataloaders = initialize_inference_dataloaders(args)
    model = initialize_model(args)
    # model = torch.compile(model)
    model = model.to(args['device'])

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # load the model with best F1-score
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', model_name, 'train', f'model_weights_epoch{epoch}.tar')
    model.load_state_dict(torch.load(model_dir)['model_state_dict'])
    print('Resume model: %s' % model_dir)
    model.eval()

    with torch.no_grad():
        # load the best epoch number from the saved "model_results" structure
        predictions = np.empty((0, args['num_labels']))
        # y_true = np.empty((0, self.config.num_labels))
        for i, (feature, index) in enumerate(dataloaders['inference']):
            # target = target.to(self.device)
            if not args['use_lfp'] and args['use_spike']:
                spike = feature.to(args['device'])
                lfp = None
            elif args['use_lfp'] and not args['use_spike']:
                lfp = {key: value.to(args['device']) for key, value in feature.items()}
                version = args['lfp_data_mode']
                lfp = lfp[version]
                spike = None
            else:
                assert isinstance(feature, list) or isinstance(feature, tuple), "Tensor must be a list or tuple"
                spike = feature[1].to(args['device'])
                lfp = feature[0].to(args['device'])
            # forward pass

            # start_time = time.time()
            spike_emb, lfp_emb, output = model(lfp, spike)
            # end_time = time.time()
            # print('inference time: ', end_time - start_time)
            output = torch.sigmoid(output)
            pred = output.cpu().detach().numpy()
            predictions = np.concatenate([predictions, pred], axis=0)

    # smoothed_data = np.zeros_like(predictions)
    print(f'AVG Activation Score is: {np.mean(predictions, axis=0)}')

if __name__ == '__main__':
    # '562', '563', '566', '567', '572', 'i728'
    patient_list = ['i728', '572', '567', '566']
    # sd_list = [4.5, 3.5, 4.5, 4, 3, 3.5]
    sd_list = [3.5, 3.5, 3.5, 3.5]
    # data_list = ['notch CAR4.5', 'notch CAR3.5', 'notch CAR4.5', 'notch CAR4', 'notch CAR3.5', 'notch CAR3.5']
    data_list = ['notch CAR-quant-neg', 'notch CAR-quant-neg', 'notch CAR-quant-neg', 'notch CAR-quant-neg', 'notch CAR-quant-neg', 'notch CAR-quant-neg']
    early_stop = [100, 100, 100, 50, 50, 75]
    for p, sd, dd in zip(patient_list, sd_list, data_list):
        config = {
            'model_architecture': 'multi-vit',
            'use_clusterless': True,
            'use_lfp': False,
            'use_combined':False,
            'use_overlap': False,
            'model_name': f'8concepts/{p}_clusterless_multi-vit_test53_optimalX_CARX_6', # '8concepts/i728_lfp_multi-vit_test53_optimal_1',
            'patient': p,
            'epoch': 50,
            'spike_data_mode': dd,
            'spike_data_mode_inference': dd,
            'spike_data_sd': sd, 
            'spike_data_sd_inference': sd, 
            'model_aggregate_type': 'mean',
            'use_augment': False,
            'use_long_input': False,
        }

        if p in ['562', '563']:
            perform_memory_test(config, phase='FR1', alongwith=[])
            perform_memory_test(config, phase='FR2', alongwith=[])
        else:
            perform_memory_test(config, phase='FR1', alongwith=['CR1'])
            perform_memory_test(config, phase='FR2', alongwith=['CR2'])
            # perform_memory_test(config, phase='FR1', alongwith=[])
            # perform_memory_test(config, phase='FR2', alongwith=[])

