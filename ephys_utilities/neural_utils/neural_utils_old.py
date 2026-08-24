#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: brain_wide_analysis
@file: neural_utils_old.py
@time: 2/11/2024 9:41 PM
"""
# Imports
import sys
import warnings

import numpy as np
import json
import pandas as pd
import os
import scipy.ndimage
from multiprocessing import Pool
import matplotlib

import neural_utils_old

matplotlib.use('Agg')  # 'TkAgg' 'Agg' 'Qt5Agg'
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Custom imports
sys.path.insert(0, r"M:\analysis\Axel_Bisi\NWB_reader")
sys.path.insert(0, "/home/bisi/code/NWB_reader")

import NWB_reader_functions as nwb_reader
from ephys_utilities import allen_utils as allen

TRIAL_MAP = {
    0: 'whisker_miss',
    1: 'auditory_miss',
    2: 'whisker_hit',
    3: 'auditory_hit',
    4: 'correct_rejection',
    5: 'false_alarm',
    6: 'association',
}


def process_single_nwb(nwb, day_to_analyze = 0):

    try:
        beh_type, day = nwb_reader.get_bhv_type_and_training_day_index(nwb)
        if day_to_analyze == 'learning' and day !=0:
            return None
        elif day_to_analyze == 'expert' and day == 0:
            return None
        elif day_to_analyze == 'all' and day <0:
            return None

        # if day_to_analyze != 'all':
        #    if day_to_analyze == 0 and day !=0:
        #        return None
        #    elif day_to_analyze > 0 and day == 0:
        #        return None

        unit_table = nwb_reader.get_unit_table(nwb)
        if unit_table is None or 'bc_label' not in unit_table.columns:
            return None

        trial_table = nwb_reader.get_trial_table(nwb)

        mouse_id = nwb_reader.get_mouse_id(nwb)
        session_id = nwb_reader.get_session_id(nwb)
        sess_metadata = nwb_reader.get_session_metadata(nwb)
        reward_group = sess_metadata['wh_reward']

        trial_table['mouse_id'] = mouse_id
        trial_table['session_id'] = session_id
        trial_table['reward_group'] = reward_group
        trial_table['context'] = trial_table['context'].astype(str)
        trial_table['day'] = day
        trial_table['behaviour'] = beh_type

        if trial_table['context'].str.contains('nan').all():
            trial_table['context'] = 'active'
        else:
            trial_table['context'] = trial_table['context'].fillna('active')
            trial_table['context'] = trial_table['context'].replace('nan', 'active')

        unit_table['mouse_id'] = mouse_id
        unit_table['session_id'] = session_id
        unit_table['reward_group'] = reward_group
        unit_table['day'] = day
        unit_table['behaviour'] = beh_type

        #print('Warning: number of root neurons :', mouse_id, len(unit_table[unit_table.ccf_acronym=='root']))

        unit_table = convert_electrode_group_object_to_columns(unit_table)

        return {
            'nwb': nwb,
            'trial_table': trial_table,
            'unit_table': unit_table
        }

    except Exception as e:
        print(f"Error processing {nwb}: {e}")
        return None


def combine_ephys_nwb(nwb_list, day_to_analyze=0, max_workers=24):
    """
    Combine neural and behavioural data from multiple NWB files using multiprocessing and tqdm.
    :param nwb_list: list of NWB file paths.
    :param max_workers: number of parallel processes.
    :return: (trial_table, unit_table, ephys_nwb_list)
    """
    ephys_nwb_list = []
    trial_table_list = []
    unit_table_list = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_nwb, nwb, day_to_analyze=day_to_analyze): nwb for nwb in nwb_list}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading NWB files"):
            result = future.result()
            if result is None:
                continue
            ephys_nwb_list.append(result['nwb'])
            trial_table_list.append(result['trial_table'])
            unit_table_list.append(result['unit_table'])

    print(f"Found {len(ephys_nwb_list)} NWB files with ephys data.")
    print(f"Available NWB files {len(ephys_nwb_list)}:", sorted([os.path.basename(nwb) for nwb in ephys_nwb_list]))

    trial_table = pd.concat(trial_table_list, ignore_index=True) if trial_table_list else pd.DataFrame()
    unit_table = pd.concat(unit_table_list, ignore_index=True) if unit_table_list else pd.DataFrame()

    if not unit_table.empty:
        print('Removing excluded areas from unit table and creating global unit IDs...')
        #unit_table = unit_table[~unit_table['ccf_atlas_acronym'].isin(allen.get_excluded_areas())]
        unit_table = unit_table.reset_index(drop=True)
        unit_table['unit_id'] = unit_table.index  # global unit identifier

    return trial_table, unit_table, ephys_nwb_list


def convert_electrode_group_object_to_columns(data):
    """
    Convert electrode group object to dictionary.
    Creates a new column in the dataframe.
    :param data: pd.DataFrame containing the NWB electrode group field.
    :return:
    """
    elec_group_list = data['electrode_group'].values
    elec_group_name = [e.name for e in elec_group_list]
    # ata['electrode_group'] = elec_group_name
    data["electrode_group"] = [getattr(e, "name", e) for e in elec_group_list]

    elec_group_location = [e.location.replace('nan', 'None') for e in elec_group_list]
    elec_group_location_dict = [eval(e) for e in elec_group_location]
    data['location'] = elec_group_location_dict
    data['target_region'] = [e.get('area') for e in elec_group_location_dict]

    return data

def convert_electrode_group_object_to_columns(data):
    """
    Convert electrode group object to dictionary.
    Creates a new column in the dataframe.
    :param data: pd.DataFrame containing the NWB electrode group field.
    :return: 
    """
    elec_group_list = data['electrode_group'].values
    elec_group_name = [e.name for e in elec_group_list]
    data['electrode_group'] = elec_group_name

    elec_group_location = [e.location.replace('nan', 'None') for e in elec_group_list]
    elec_group_location_dict = [eval(e) for e in elec_group_location]
    data['location'] = elec_group_location_dict
    data['target_region'] = [e.get('area') for e in elec_group_location_dict]

    return data

def compute_fano_factor_from_spike_train(spike_times, event_times, bin_size, time_start, time_stop):
    """
    Computes Fano factor for a single unit over trials.
    :param spike_times:  Spike times in seconds.
    :param event_times:  Stimulus times in seconds.
    :param bin_size: Bin size in seconds.
    :param time_start: Start of peri-stimulus time window.
    :param time_stop: End of peri-stimulus time window.
    :return: Fano factor of spike counts.
    """

    # Initialize Fano factor
    n_bins = int((time_stop - time_start) / bin_size)
    spike_counts = np.zeros((len(event_times), n_bins))

    # Compute spike counts
    for i, stim_time in enumerate(event_times):
        spike_times_in_window = spike_times[
            (spike_times >= stim_time + time_start) & (spike_times < stim_time + time_stop)]
        spike_times_in_window -= stim_time  # align
        spike_counts[i, :] = \
        np.histogram(spike_times_in_window, bins=np.arange(time_start, time_stop + bin_size, bin_size), density=False)[
            0]

    # Compute Fano factor
    fano_factor = np.var(spike_counts, axis=0) / np.mean(spike_counts, axis=0)

    return fano_factor


def compute_trial_variance_for_unit(args):
    """
    Compute across-trial variance of neural activity (PETH variance)
    for a single unit across all trial types and lick flags.
    This function mirrors `compute_peth_for_unit` but replaces the
    trial average with a variance across trials.
    """
    cluster, trial_table_mouse, params = args
    if trial_table_mouse.empty:
        return []

    bin_size = params['bin_size']
    time_start = params['time_start']
    time_stop = params['time_stop']
    baseline_correction = False
    n_max_trials = params['n_max_trials']
    artifact_correction = params['artifact_correction']
    zscore_unit = params['zscore_unit']

    spike_times = np.array(cluster['spike_times'])

    if 'jaw_dlc_onset' in trial_table_mouse.columns:
        align_times = ['start_time', 'jaw_dlc_onset']
    else:
        align_times = ['start_time']

    var_list = []

    for align_event in align_times:
        for context in ['active', 'passive']:
            for trial_type in ['whisker_trial', 'auditory_trial', 'no_stim_trial']:
                lick_flags = [0] if context == 'passive' else [0, 1]

                for lick_flag in lick_flags:
                    event_times_dict = {}
                    if context == 'active':
                        trial_table_sub = trial_table_mouse[
                            (trial_table_mouse['trial_type'] == trial_type)
                            & (trial_table_mouse['lick_flag'] == lick_flag)
                            & (trial_table_mouse['context'] == context)
                            ]
                    elif context == 'passive':
                        trial_table_sub = trial_table_mouse[
                            (trial_table_mouse['trial_type'] == trial_type)
                            & (trial_table_mouse['context'] == context)
                            ]
                    else:
                        continue

                    event_times = trial_table_sub[align_event].values
                    if len(event_times) == 0:
                        continue

                    if len(trial_table_sub['outcome'].unique()) > 1:
                        assert False, "Multiple outcomes found"
                    outcome = trial_table_sub['outcome'].unique()[0]

                    if n_max_trials is not None and len(event_times) > n_max_trials:
                        event_times = event_times[:n_max_trials]

                    # Split passive into pre/post
                    if context == 'passive':
                        idx_mid = len(trial_table_mouse) // 2
                        idx_mid_time = trial_table_mouse['start_time'].values[idx_mid]
                        event_times_dict['passive_pre'] = event_times[event_times < idx_mid_time]
                        event_times_dict['passive_post'] = event_times[event_times >= idx_mid_time]
                    else:
                        event_times_dict['active'] = event_times

                    for key, ev_times in event_times_dict.items():
                        if len(ev_times) == 0:
                            continue

                        # Compute trial-wise PETHs
                        peth = compute_unit_peri_event_histogram(
                            spike_times=spike_times,
                            event_times=ev_times,
                            bin_size=bin_size,
                            time_start=time_start,
                            time_stop=time_stop,
                            artifact_correction=artifact_correction,
                        )

                        # Baseline correction per trial
                        if baseline_correction:
                            bas_stop = int(abs(time_start) / bin_size)
                            baseline = compute_baseline_from_peth(peth, bas_start=0, bas_stop=bas_stop)
                            peth = peth - baseline

                        # Variance across trials (instead of mean)
                        peth_var = np.nanvar(peth, axis=0)

                        cluster_info = cluster.to_dict()
                        cluster_info.update({
                            'align_event': align_event,
                            'trial_var': np.array(peth_var),
                            'reward_group': trial_table_mouse['reward_group'].unique()[0],
                            'session_id': trial_table_mouse['session_id'].unique()[0],
                            'trial_type': trial_type,
                            'lick_flag': lick_flag if context == 'active' else 0,
                            'outcome': outcome if context == 'active' else 'passive',
                            'artifact_correction': artifact_correction,
                            'context': key,
                        })
                        var_list.append(cluster_info)

        # Combined whisker condition (active hit/miss)
        whisker_trials = trial_table_mouse[
            (trial_table_mouse['trial_type'] == 'whisker_trial')
            & (~trial_table_mouse['context'].isin(['passive']))
            ]
        if not whisker_trials.empty:
            event_times = whisker_trials[align_event].values
            if n_max_trials is not None and len(event_times) > n_max_trials:
                event_times = event_times[:n_max_trials]

            peth = compute_unit_peri_event_histogram(
                spike_times=spike_times,
                event_times=event_times,
                bin_size=bin_size,
                time_start=time_start,
                time_stop=time_stop,
                artifact_correction=artifact_correction,
            )

            if baseline_correction:
                bas_stop = int(abs(time_start) / bin_size)
                baseline = compute_baseline_from_peth(peth, bas_start=0, bas_stop=bas_stop)
                peth = peth - baseline

            peth_var = np.nanvar(peth, axis=0)

            cluster_info = cluster.to_dict()
            cluster_info.update({
                'align_event': align_event,
                'trial_var': np.array(peth_var),
                'reward_group': trial_table_mouse['reward_group'].unique()[0],
                'session_id': trial_table_mouse['session_id'].unique()[0],
                'trial_type': 'whisker_trial',
                'lick_flag': -1,
                'outcome': 'whisker_stim',
                'artifact_correction': artifact_correction,
                'context': 'active',
            })
            var_list.append(cluster_info)

    # Optionally z-score the variance across conditions for the unit
    if zscore_unit and len(var_list) > 1:
        vars_all = np.stack([p['trial_var'] for p in var_list])
        mean = np.nanmean(vars_all)
        std = np.nanstd(vars_all)
        if std == 0 or np.isnan(std):
            std = 1.0
        vars_all_z = (vars_all - mean) / std
        for i, v in enumerate(var_list):
            v['trial_var'] = vars_all_z[i]

    return var_list


def compute_unit_peri_event_histogram(spike_times, event_times, bin_size, time_start, time_stop,
                                      artifact_correction=True):
    """
    Computes peri-stimulus time histogram for a single unit.
    :param spike_times:  Spike times in seconds.
    :param event_times:  Stimulus times in seconds.
    :param bin_size: Bin size in seconds.
    :param time_start: Start of peri-stimulus time window.
    :param time_stop: End of peri-stimulus time window.
    :param artifact_correction: Boolean to apply artifact correction.
    :return: Peri-stimulus time histogram of spike counts.
    """

    # Initialize histogram
    if artifact_correction:
        bin_size_hist = 0.001
        n_bins = int((time_stop - time_start) / 0.001)
    else:
        bin_size_hist = bin_size
        n_bins = int((time_stop - time_start) / bin_size)

    peri_stim_hist = np.zeros((len(event_times), n_bins))
    if len(event_times) == 0:
        return np.full((1, n_bins), np.nan)

    # Compute histogram
    for i, event in enumerate(event_times):
        spike_times_in_window = spike_times[(spike_times >= event + time_start) & (spike_times < event + time_stop)]
        spike_times_in_window -= event  # align
        spike_counts = \
        np.histogram(spike_times_in_window, bins=np.arange(time_start, time_stop + bin_size_hist, bin_size_hist),
                     density=False)[0]
        peri_stim_hist[i, :] = spike_counts[:n_bins]  # add counts
        # peri_stim_hist[i,:] = spike_counts # add counts

    if artifact_correction:
        # Bin to correct for artifact
        if bin_size_hist == 0.001:
            stim_dur = 3  # stimulus duration in msec
            art_start = -2  # ms before stim
            art_stop = stim_dur + 2  # ms after stim
            art_start_bin = int(abs(time_start) / bin_size_hist) + art_start
            art_stop_bin = int(abs(time_start) / bin_size_hist) + art_stop

        # Get baseline firing rate from PETH
        bas_stop = int(abs(time_start) / bin_size_hist) - 5  # 5 time bins before stim
        trial_baselines = compute_trial_baseline_from_peth(peri_stim_hist,
                                                           bas_start=0,
                                                           bas_stop=bas_stop)
        trial_baselines = np.nan_to_num(trial_baselines, nan=0)
        # Make Poisson noise based on baseline firing rate
        rng = np.random.default_rng(seed=None)  # no seed for variability
        poisson_noise = [rng.poisson(lam=trial_baselines[i], size=n_bins) for i in range(len(event_times))]
        poisson_noise = np.array(poisson_noise)

        # Replace spike counts with Poisson noise in artifact window
        try:
            # print('shape', peri_stim_hist.shape, 'art_start_bin:', art_start_bin, 'art_stop_bin:', art_stop_bin)
            peri_stim_hist[:, art_start_bin:art_stop_bin] = poisson_noise[:, art_start_bin:art_stop_bin]
        except IndexError:
            print('Index error in artifact correction. Skipping correction because no trials.')
            print('shape', peri_stim_hist.shape, 'art_start_bin:', art_start_bin, 'art_stop_bin:', art_stop_bin,
                  'events', len(event_times))
            print('art_start_bin:', art_start_bin)
            return peri_stim_hist

        # Rebin to desired bin size if there was artifact correction
        if artifact_correction and bin_size != 0.001:
            # Aggregate spike counts in bin_size in ms time bins using the sum over the bins
            current_bin_size_ms = int(bin_size_hist * 1000)
            new_bin_size_ms = int(bin_size * 1000)
            n_trials = peri_stim_hist.shape[0]

            peri_stim_hist_original = peri_stim_hist.copy()
            peri_stim_hist = peri_stim_hist.reshape(n_trials, -1, new_bin_size_ms // current_bin_size_ms).sum(axis=2)

            debug = False
            if debug:
                fig, ax = plt.subplots(1, 1)
                time = np.linspace(time_start, time_stop, peri_stim_hist.shape[1])
                ax.plot(time, np.nanmean(peri_stim_hist_original, axis=0), c='k')
                ax.plot(time, np.nanmean(peri_stim_hist, axis=0), c='r')
                ax.axvline(0, c='k', linestyle='--')
                plt.show()

    return peri_stim_hist


def compute_fano_factor_from_peth(peth, time_start, time_stop):
    """
    Computes Fano factor from peri-event time histogram of a single unit.
    :param peth: Peri-event time histogram.
    :return: Fano factor of spike counts.
    """
    # Get window of interest
    peth = peth[:, int(time_start):int(time_stop)]
    # Compute Fano factor
    fano_factor = np.var(peth, axis=0) / np.mean(peth, axis=0)

    return fano_factor


def compute_baseline_from_spike_train(spike_times, event_times, bas_start, bas_stop):
    """
    Computes baseline firing rate for a unitspike train.
    :param spike_times:  Spike times in seconds.
    :param event_times:  Stimulus times in seconds.
    :param bas_start: Start of baseline time window.
    :param bas_stop: End of baseline time window.
    :return: Baseline firing rate, in Hz.
    """

    # Get all spikes in baseline window
    baseline_spikes = []
    for event in event_times:
        bas_spikes = spike_times[(spike_times >= event + bas_start) & (spike_times < event + bas_stop)]
        baseline_spikes.append(bas_spikes)
    # Calculate baseline firing rate
    baseline_rate = len(baseline_spikes) / (bas_stop - bas_start)

    return baseline_rate


def compute_baseline_from_peth(peth, bas_start, bas_stop):
    """
    Computes baseline firing rate from peri-event time histogram.
    :param peth: Peri-event time histogram.
    :param bas_start: Start of baseline time window.
    :param bas_stop: End of baseline time window.
    :return: Baseline firing rate, in Hz.
    """
    baseline_rate = np.mean(peth[:, bas_start:bas_stop])
    return baseline_rate


def compute_trial_baseline_from_peth(peth, bas_start, bas_stop):
    """
    Computes baseline firing rate from peri-event time histogram, for each trials.
    :param peth: Peri-event time histogram.
    :param bas_start: Start of baseline time window.
    :param bas_stop: End of baseline time window.
    :return: Baseline firing rate, in Hz.
    """
    baseline_rate = np.mean(peth[:, bas_start:bas_stop], axis=1)
    return baseline_rate


def compute_zscored_all(peths_array):
    """
    Z-score peri-event time histograms of a population of units.
    :param peths_array:
    :return:
    """
    mean_firing_rates = np.mean(peths_array)
    std_firing_rates = np.std(peths_array)
    return (peths_array - mean_firing_rates) / std_firing_rates


def normalize_by_std(peths_array):
    """
    Divide peri-event time histograms of a population of units by the standard deviation.
    PETHs are assumed already centered around zero.
    :param peths_array:
    :return:
    """
    std_firing_rates = np.std(peths_array, axis=1)
    return peths_array / std_firing_rates[:, np.newaxis]


def compute_zscored_peths(peths_array):
    """
    Z-score peri-event time histograms of a population of units.
    :param peths_array: Array of peri-event time histograms.
    :return: Z-scored peri-event time histograms.
    """
    mean_firing_rates = np.mean(peths_array, axis=1)
    std_firing_rates = np.std(peths_array, axis=1)
    return (peths_array - mean_firing_rates[:, np.newaxis]) / std_firing_rates[:, np.newaxis]


def compute_zscored_peths_per_unit(peths_array):
    """
    Z-score peri-event time histograms of a population of units, for each unit/population.
    :param peths_array:
    :return:
    """
    mean_firing_rates = np.mean(peths_array, axis=1)
    std_firing_rates = np.std(peths_array, axis=1)
    return (peths_array - mean_firing_rates[np.newaxis, :, :]) / std_firing_rates[np.newaxis, :, :]


def apply_moving_average(data, window_size):
    """ Computes a moving average of a 1-D array, with a window size centered on each data point.
    The size adapts at the edges of the array.
    """
    smoothed_data = []
    for i in range(len(data)):
        start_index = max(0, i - window_size // 2)
        end_index = min(len(data), i + window_size // 2 + 1)
        window = data[start_index:end_index]
        smoothed_data.append(sum(window) / len(window))
    return smoothed_data


def halfgaussian_kernel1d(sigma, radius):
    """
    Computes a 1-D Half-Gaussian convolution kernel.
    """
    sigma2 = sigma * sigma
    x = np.arange(0, radius + 1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    return phi_x


def halfgaussian_filter1d(input, sigma, axis=-1, output=None,
                          mode="nearest", cval=0.0, truncate=4.0):
    """
    Convolves a 1-D Half-Gaussian convolution kernel.
    """
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    weights = halfgaussian_kernel1d(sigma, lw)
    origin = -lw // 2
    return scipy.ndimage.convolve1d(input, weights, axis, output, mode, cval, origin)


def half_gaussian_kernel(size, sigma):
    t = np.arange(0, size)
    kernel = np.exp(-t ** 2 / (2 * sigma ** 2))
    kernel = np.exp(-t ** 2 / (2 * sigma ** 2))
    return kernel / np.sum(kernel)  # Normalize the kernel


def causal_gaussian_filter(spike_train, sigma):
    size = int(3 * sigma)  # Choose the size of the kernel
    kernel = half_gaussian_kernel(size, sigma)
    smoothed_train = np.convolve(spike_train, kernel, mode='full')[:len(spike_train)]
    return smoothed_train


def subtract_baseline_from_peth(peth, bin_size, align_event, time_start, per_trial=False):
    """
    Subtract baseline from a PETH, optionally per-trial.

    :param peth: array of shape (trials, bins)
    :param bin_size: size of bins in seconds
    :param align_event: string indicating event alignment type
    :param time_start: start time of PETH relative to event
    :param per_trial: bool, if True compute baseline separately for each trial
    :return: baseline-corrected PETH
    """
    if align_event == 'start_time':
        # baseline from time_start back to 0
        baseline_duration = abs(time_start)
        bas_stop = int(baseline_duration / bin_size)
        bas_start = 0

    elif align_event == 'jaw_dlc_onset':  # TODO: update to align at stim time
        # baseline from -1 to -0.5 sec before jaw onset
        baseline_duration = 0.5  # 0.5 sec window
        bas_stop = int(baseline_duration / bin_size)
        bas_start = 0

    else:
        raise ValueError(f"Unknown align_event: {align_event}")

    if per_trial:
        # Subtract baseline per trial
        baseline = np.nanmean(peth[:, bas_start:bas_stop], axis=1, keepdims=True)  # shape (trials, 1)
        peth_corrected = peth - baseline
    else:
        # Subtract baseline averaged across all trials
        baseline = np.nanmean(peth[:, bas_start:bas_stop])
        peth_corrected = peth - baseline

    return peth_corrected


def compute_peth_for_unit(args):
    """
    Compute PETH for a single unit (cluster) across all trial types and lick flags.
    This function is designed to run in parallel processes.
    :param args: A tuple containing (cluster, trial_table_mouse, params).
    :return: List of dictionaries containing PETH information for the unit.
    """
    cluster, trial_table_mouse, params = args
    if trial_table_mouse.empty:
        return []

    bin_size = params['bin_size']
    time_start = params['time_start']
    time_stop = params['time_stop']
    baseline_correction = params['baseline_correction']
    n_max_trials = params['n_max_trials']
    artifact_correction = params['artifact_correction']

    # Ensure spike_times is a numpy array (if it was an h5py object)
    spike_times = np.array(cluster['spike_times'])

    # Check if jaw onset times are available
    if 'jaw_dlc_onset' in trial_table_mouse.columns:
        align_times = ['jaw_dlc_onset']
    else:
        align_times = ['start_time']

    peth_list = []

    # Iterate over all combinations of trial type and lick flag
    for align_event in align_times:
        for context in ['active', 'passive']:
            for trial_type in ['whisker_trial', 'auditory_trial', 'no_stim_trial']:
                if context == 'passive':
                    lick_flags = [0]
                else:
                    lick_flags = [0, 1]

                for lick_flag in lick_flags:
                    event_times_dict = {}

                    if context == 'active':
                        # Filter trials based on trial type and lick flag
                        trial_table_sub = trial_table_mouse[
                            (trial_table_mouse['trial_type'] == trial_type)
                            & (trial_table_mouse['lick_flag'] == lick_flag)
                            & (trial_table_mouse['context'] == context)
                            ]

                    elif context == 'passive':
                        # Filter trials based on trial type and lick flag
                        trial_table_sub = trial_table_mouse[
                            (trial_table_mouse['trial_type'] == trial_type)
                            & (trial_table_mouse['lick_flag'] == lick_flag)
                            & (trial_table_mouse['context'] == context)
                            ]
                    else:
                        print('Unknown context:', context)
                        continue

                    # Get event times and outcome
                    event_times = trial_table_sub[align_event].values

                    if len(event_times) == 0:
                        print('No event times for', len(trial_table_sub), context, trial_type, lick_flag,
                              cluster['unit_id'], trial_table_mouse['mouse_id'].unique()[0], flush=True)
                        continue
                    if len(trial_table_sub['outcome'].unique()) > 1:
                        print(f'Warning: multiple outcomes found at selection for {trial_type}, lick={lick_flag}',
                              trial_table_sub['outcome'].unique())

                    outcome = trial_table_sub['outcome'].unique()[0]

                    # For passive, get pre and post sessions
                    if context == 'passive':
                        # Get trial index midway in session
                        idx_mid = len(trial_table_mouse) // 2
                        idx_mid_time = trial_table_mouse['start_time'].values[idx_mid]

                        # Get pre and post passive trials
                        event_times_pre = event_times[event_times < idx_mid_time]
                        event_times_post = event_times[event_times >= idx_mid_time]
                        event_times_dict['passive_pre'] = event_times_pre
                        event_times_dict['passive_post'] = event_times_post

                    elif context == 'active':
                        # Optionally, limit number of trials for active
                        if n_max_trials is not None and len(event_times) > n_max_trials:
                            event_times = event_times[:n_max_trials]

                        event_times_dict['active'] = event_times

                    # Iterate over event times
                    for key, event_times in event_times_dict.items():
                        # Compute PETH
                        # if len(event_times) == 0:
                        #    continue
                        peth = compute_unit_peri_event_histogram(spike_times=spike_times,
                                                                 event_times=event_times,
                                                                 bin_size=bin_size,
                                                                 time_start=time_start,
                                                                 time_stop=time_stop,
                                                                 artifact_correction=artifact_correction)

                        # Apply baseline correction if enabled
                        if baseline_correction:
                            peth = subtract_baseline_from_peth(
                                peth,
                                bin_size,
                                align_event,
                                time_start,
                                per_trial=True  # set True for per-trial baseline
                            )

                        # if baseline_correction:
                        #    if align_event == 'start_time': # from time-start back to baseline duration
                        #        baseline_duration = abs(time_start)
                        #        bas_stop = int(baseline_duration / bin_size)
                        #        baseline = compute_baseline_from_peth(peth, bas_start=0, bas_stop=bas_stop)
                        #
                        #    elif align_event == 'jaw_dlc_onset': # from -1 to -0.5 sec before jaw onset
                        #        baseline_duration = 0.1
                        #        bas_stop = int(baseline_duration / bin_size)
                        #        baseline = compute_baseline_from_peth(peth, bas_start=0, bas_stop=bas_stop)
                        #
                        #    # Remove baseline from PETH
                        #    peth = peth - baseline

                        # Average across trials
                        peth = np.nanmean(peth, axis=0)

                        # Add PETH info to cluster dict
                        cluster_info = cluster.to_dict()
                        cluster_info['align_event'] = align_event
                        cluster_info['peth'] = np.array(peth)
                        cluster_info['reward_group'] = trial_table_mouse['reward_group'].unique()[0]
                        cluster_info['session_id'] = trial_table_mouse['session_id'].unique()[0]
                        cluster_info['trial_type'] = trial_type
                        cluster_info['lick_flag'] = lick_flag if context == 'active' else 0
                        cluster_info['outcome'] = outcome if context == 'active' else 'passive'
                        cluster_info['artifact_correction'] = artifact_correction
                        cluster_info['context'] = key

                        # Append to list
                        peth_list.append(cluster_info)

        # PETH for whisker stimulation (active hit and miss)
        # -------------------------------------------
        whisker_trials = trial_table_mouse[
            (trial_table_mouse['trial_type'] == 'whisker_trial')
            & (~trial_table_mouse['context'].isin(['passive']))
            ]

        if not whisker_trials.empty:
            event_times = whisker_trials[align_event].values
            if n_max_trials is not None and len(event_times) > n_max_trials:
                event_times = event_times[:n_max_trials]

            peth = compute_unit_peri_event_histogram(
                spike_times=spike_times,
                event_times=event_times,
                bin_size=bin_size,
                time_start=time_start,
                time_stop=time_stop,
                artifact_correction=artifact_correction,
            )

            if baseline_correction:
                if align_event == 'start_time':  # from time-start back to baseline duration
                    baseline_duration = abs(time_start)
                    bas_stop = int(baseline_duration / bin_size)
                    baseline = compute_baseline_from_peth(peth, bas_start=0, bas_stop=bas_stop)

                elif align_event == 'jaw_dlc_onset':  # from -1 to -0.5 sec before jaw onset
                    baseline_duration = 0.1
                    bas_stop = int(baseline_duration / bin_size)
                    baseline = compute_baseline_from_peth(peth, bas_start=0, bas_stop=bas_stop)

            peth = np.nanmean(peth, axis=0)

            cluster_info = cluster.to_dict()
            cluster_info.update({
                'align_event': align_event,
                'peth': np.array(peth),
                'reward_group': trial_table_mouse['reward_group'].unique()[0],
                'session_id': trial_table_mouse['session_id'].unique()[0],
                'trial_type': 'whisker_trial',
                'lick_flag': 1,  # combined case
                'outcome': 'whisker_stim',
                'artifact_correction': artifact_correction,
                'context': 'active',
            })

            peth_list.append(cluster_info)

    # Normalization across all PETHs for the unit
    scaling_method = params.get('unit_scaling', None)  # 'zscore' or 'minmax'

    if scaling_method and len(peth_list) > 1:
        peths_all = np.stack([p['peth'] for p in peth_list])

        if scaling_method == 'zscore':
            print('Unit normalization: z-score')
            # Z-score normalization
            subtract_mean = not baseline_correction
            mean = np.nanmean(peths_all) if subtract_mean else 0
            std = np.nanstd(peths_all)
            if std == 0 or np.isnan(std):
                std = 1.0
            peths_all_scaled = (peths_all - mean) / std

        elif scaling_method == 'minmax':
            print('Unit normalization: min-max')
            # Min-max normalization
            min_val = np.nanmin(peths_all)
            max_val = np.nanmax(peths_all)
            if np.isnan(min_val) or np.isnan(max_val) or max_val == min_val:
                # Avoid division by zero
                peths_all_scaled = np.zeros_like(peths_all)
            else:
                peths_all_scaled = (peths_all - min_val) / (max_val - min_val)

        else:
            raise ValueError(f"Unknown scaling_method: {scaling_method}")

        # Update back into list
        for i, p in enumerate(peth_list):
            p['peth'] = peths_all_scaled[i]

    return peth_list


def compute_peth_for_unit_block(args):
    """
    Compute PETH for a single unit (cluster) across all trial types and lick flags.
    This function is designed to run in parallel processes.
    :param args: A tuple containing (cluster, trial_table_mouse, params).
    :return: List of dictionaries containing PETH information for the unit.
    """
    cluster, trial_table_mouse, params = args

    bin_size = params['bin_size']
    time_start = params['time_start']
    time_stop = params['time_stop']
    baseline_correction = params['baseline_correction']
    n_max_trials = params['n_max_trials']
    artifact_correction = params['artifact_correction']
    align_event = 'start_time'

    # Ensure spike_times is a numpy array (if it was an h5py object)
    spike_times = np.array(cluster['spike_times'])

    peth_list = []

    # Iterate over all combinations of trial type, lick flag, block
    for trial_type in ['whisker_trial', 'auditory_trial', 'no_stim_trial']:

        trial_table_type = trial_table_mouse[trial_table_mouse['trial_type'] == trial_type]
        if params['all_blocks']:
            block_id_list = trial_table_type['block_id'].unique()
        else:  # early vs. late (take only these two)
            block_id_list = trial_table_type['block_id'].unique()
            block_id_list = [block_id_list[0], block_id_list[-1]]

        for block_id in block_id_list:
            event_times_dict = {}

            # Filter trials
            trial_table_sub = trial_table_type[
                # (trial_table_type['context'] == 'active') #combine both passive anda ctive (if early learning trial)
                # &
                (trial_table_type['block_id'] == block_id)
            ]

            if trial_table_sub.empty:
                print('Missing condition for', trial_type, block_id, cluster, trial_table_mouse.mouse_id)
                continue

            # Get event times and outcome
            event_times = trial_table_sub['start_time'].values
            outcome = trial_table_sub['outcome'].unique()[0]

            # Optionally, limit number of trials
            if n_max_trials is not None and len(event_times) > n_max_trials:
                event_times = event_times[:n_max_trials]

            event_times_dict['active'] = event_times

            # Iterate over event times
            for key, event_times in event_times_dict.items():
                # Compute PETH
                peth = compute_unit_peri_event_histogram(spike_times=spike_times,
                                                         event_times=event_times,
                                                         bin_size=bin_size,
                                                         time_start=time_start,
                                                         time_stop=time_stop,
                                                         artifact_correction=artifact_correction)

                # Apply baseline correction if enabled
                if baseline_correction:
                    peth = subtract_baseline_from_peth(
                        peth,
                        bin_size,
                        align_event,
                        time_start,
                        per_trial=True  # set True for per-trial baseline
                    )

                # Average across trials
                peth = np.mean(peth, axis=0)

                # Add PETH info to cluster dict
                cluster_info = cluster.to_dict()
                cluster_info['peth'] = np.array(peth)
                cluster_info['reward_group'] = trial_table_mouse['reward_group'].unique()[0]
                cluster_info['session_id'] = trial_table_mouse['session_id'].unique()[0]
                cluster_info['trial_type'] = trial_type
                cluster_info['lick_flag'] = 1
                cluster_info['outcome'] = outcome
                cluster_info['artifact_correction'] = artifact_correction
                cluster_info['context'] = key

                if params['all_blocks']:  # at inflection trial
                    cluster_info['block'] = block_id
                else:  # early vs late.
                    if block_id == 0:
                        cluster_info['block'] = 'early'
                    else:
                        cluster_info['block'] = 'late'

            # Append to list
            peth_list.append(cluster_info)

    # PETH for whisker stimulation (active hit and miss)
    # -------------------------------------------
    whisker_trials = trial_table_mouse[
        (trial_table_mouse['trial_type'] == 'whisker_trial')
        & (~trial_table_mouse['context'].isin(['passive']))
        ]

    if not whisker_trials.empty:
        event_times = whisker_trials[align_event].values
        if n_max_trials is not None and len(event_times) > n_max_trials:
            event_times = event_times[:n_max_trials]

        peth = compute_unit_peri_event_histogram(
            spike_times=spike_times,
            event_times=event_times,
            bin_size=bin_size,
            time_start=time_start,
            time_stop=time_stop,
            artifact_correction=artifact_correction,
        )

        if baseline_correction:
            bas_stop = int(abs(time_start) / bin_size)
            baseline = compute_baseline_from_peth(peth, bas_start=0, bas_stop=bas_stop)
            peth = peth - baseline

        peth = np.nanmean(peth, axis=0)

        cluster_info = cluster.to_dict()
        cluster_info.update({
            'align_event': align_event,
            'peth': np.array(peth),
            'reward_group': trial_table_mouse['reward_group'].unique()[0],
            'session_id': trial_table_mouse['session_id'].unique()[0],
            'trial_type': 'whisker_trial',
            'lick_flag': -1,  # combined case
            'outcome': 'whisker_stim',
            'artifact_correction': artifact_correction,
            'context': 'active',
        })

        peth_list.append(cluster_info)

    # Normalization across all PETHs for the unit
    scaling_method = params.get('unit_scaling', None)  # 'zscore' or 'minmax'

    if scaling_method and len(peth_list) > 1:
        peths_all = np.stack([p['peth'] for p in peth_list])

        if scaling_method == 'zscore':
            # Z-score normalization
            subtract_mean = not baseline_correction
            mean = np.nanmean(peths_all) if subtract_mean else 0
            std = np.nanstd(peths_all)
            if std == 0 or np.isnan(std):
                std = 1.0
            peths_all_scaled = (peths_all - mean) / std

        elif scaling_method == 'minmax':
            # Min-max normalization
            min_val = np.nanmin(peths_all)
            max_val = np.nanmax(peths_all)
            if np.isnan(min_val) or np.isnan(max_val) or max_val == min_val:
                # Avoid division by zero
                peths_all_scaled = np.zeros_like(peths_all)
            else:
                peths_all_scaled = (peths_all - min_val) / (max_val - min_val)

        else:
            raise ValueError(f"Unknown scaling_method: {scaling_method}")

        # Update back into list
        for i, p in enumerate(peth_list):
            p['peth'] = peths_all_scaled[i]

    return peth_list


def build_peth_table_parallel(trial_table, unit_table, params, proc_data_path, file_name):
    """
    Build peri-event time histogram dataframe from trial and unit tables using multiprocessing for speed.
    :param trial_table: Trial table.
    :param unit_table: Unit table.
    :param params: Dictionary of peri-event time histogram parameters.
    :param proc_data_path: Path to save processed data table.
    :param file_name: Name of the output file (without extension).
    :return: Peri-event time histogram table.
    """

    print('Building session-wide PETH table with multiprocessing...')

    # Map performance to outcome in trial_table
    trial_table['outcome'] = trial_table['perf'].astype('int32').map(TRIAL_MAP)

    # Group trial table by mouse_id and pass it along with the cluster for each unit
    trial_by_mouse = {m: df for m, df in trial_table.groupby('mouse_id')}
    unit_args = [
        (cluster, trial_by_mouse.get(cluster['mouse_id'], pd.DataFrame()), params)
        for _, cluster in unit_table.iterrows()
    ]

    # Ensure that all spike_times are numpy arrays (required by multiprocessing)
    for cluster in unit_args:
        cluster[0]['spike_times'] = np.array(cluster[0]['spike_times'])

    # Set up multiprocessing pool
    with Pool(os.cpu_count() - 2) as pool:
        results = pool.map(compute_peth_for_unit, unit_args)

    # Flatten list of lists into a single list
    peth_list = [item for sublist in results for item in sublist]

    # Convert list to DataFrame
    peth_df = pd.DataFrame(peth_list)

    # Add metadata columns
    peth_df['bin_size'] = params['bin_size']
    peth_df['time_start'] = params['time_start']
    peth_df['time_stop'] = params['time_stop']

    # Save separately for start_time and jaw_onset
    # peth_df_start = peth_df[peth_df['align_event']=='start_time']
    # peth_df_jaw = peth_df[peth_df['align_event']=='jaw_onset_time']

    # Save the PETH table to HDF5 format
    if proc_data_path is not None:
        if not os.path.exists(proc_data_path):
            os.makedirs(proc_data_path)

        if params['align_event'] == 'start_time':
            fname = file_name + f'_start_time'
        elif params['align_event'] == 'jaw_onset_time':
            fname = file_name + f'_jaw_onset'
        else:
            print(f'Warning: unknown align_event {params["align_event"]}.')

        # for df, suffix in zip([peth_df_start, peth_df_jaw], ['start_time', 'jaw_onset']):
        #    fname = file_name + f'_{suffix}'
        try:
            output_file = os.path.join(proc_data_path, fname + '.h5')
            peth_df.to_hdf(output_file, key='df', mode='w')
            print(f'Saved multi-mouse PETH dataframe to {output_file}')

        except Exception as e:
            print('Could not save as HDF5:', e)
            output_file = os.path.join(proc_data_path, fname + '.h5')
            peth_df.to_pickle(output_file)
            print(f'Saved multi-mouse PETH dataframe to {output_file}')

        with open(os.path.join(proc_data_path, fname + '_params.json'), 'w') as f:
            json.dump(params, f)

    return peth_df


def build_trial_variance_table_parallel(trial_table, unit_table, params, proc_data_path=None, file_name=None):
    """
    Compute across-trial variance peri-event for each unit in parallel.
    """
    print("Building trial-variance table with multiprocessing...")

    # Prepare arguments per unit
    trial_by_mouse = {m: df for m, df in trial_table.groupby('mouse_id')}
    unit_args = [
        (row, trial_by_mouse.get(row['mouse_id'], pd.DataFrame()), params)
        for _, row in unit_table.iterrows()
    ]

    # Multiprocessing
    with Pool(os.cpu_count() - 2) as pool:
        results = pool.map(compute_trial_variance_for_unit, unit_args)

    # Flatten and create DataFrame
    var_list = [item for sublist in results for item in sublist]
    var_df = pd.DataFrame(var_list)

    # Add metadata
    var_df['bin_size'] = params['bin_size']
    var_df['time_start'] = params['time_start']
    var_df['time_stop'] = params['time_stop']

    # Save
    if proc_data_path is not None and file_name is not None:
        os.makedirs(proc_data_path, exist_ok=True)
        output_file = os.path.join(proc_data_path, file_name + '_trial_variance.h5')
        var_df.to_hdf(output_file, key='df', mode='w')
        with open(os.path.join(proc_data_path, file_name + '_params.json'), 'w') as f:
            json.dump(params, f)
        print(f"Saved trial-variance table to {output_file}")

    return var_df


def build_peth_table_parallel_block(trial_table, unit_table, params, proc_data_path):
    """
    Build peri-event time histogram dataframe from trial and unit tables using multiprocessing for speed,
    for different trial blocks.
    :param trial_table: Trial table.
    :param unit_table: Unit table.
    :param params: Dictionary of peri-event time histogram parameters.
    :param proc_data_path: Path to save processed data table.
    :return: Peri-event time histogram table.
    """

    print('Building block-wise PETH table with multiprocessing...')

    # Remove passive data, early licks, association trials
    trial_table = trial_table[trial_table['context'] != 'passive']
    trial_table = trial_table[trial_table['perf'] != 6]

    # Map performance to outcome in trial_table
    trial_table['outcome'] = trial_table['perf'].astype(int).map(TRIAL_MAP)
    trial_table = trial_table[trial_table['lick_flag'] == 1]
    trial_table['trial_id_type'] = trial_table.groupby(['mouse_id', 'context', 'trial_type'])['trial_id'].cumcount()

    # Create trial blocks
    params['all_blocks'] = False
    trial_table['block_id'] = trial_table['trial_id_type'] // params['block_size']

    # Convert info from electrode group object into columns (group name, probe location, target area)
    unit_table = convert_electrode_group_object_to_columns(unit_table)

    # Group trial table by mouse_id and pass it along with the cluster for each unit
    unit_args = [(cluster, trial_table[trial_table['mouse_id'] == cluster['mouse_id']], params)
                 for idx, cluster in unit_table.iterrows()]

    # Ensure that all spike_times are numpy arrays (required by multiprocessing)
    for cluster in unit_args:
        cluster[0]['spike_times'] = np.array(cluster[0]['spike_times'])

    # Set up multiprocessing pool
    with Pool(os.cpu_count() - 5) as pool:
        # Distribute computation of PETHs across processes
        results = pool.map(compute_peth_for_unit_block, unit_args)

    # Flatten list of lists into a single list
    peth_list = [item for sublist in results for item in sublist]

    # Convert list to DataFrame
    peth_df = pd.DataFrame(peth_list)

    # Add metadata columns
    peth_df['bin_size'] = params['bin_size']
    peth_df['time_start'] = params['time_start']
    peth_df['time_stop'] = params['time_stop']
    peth_df['baseline_correction'] = params['baseline_correction']

    # Save the PETH table to feather format
    if proc_data_path is not None:
        file_name = 'peth_table_block'
        if not os.path.exists(proc_data_path):
            os.makedirs(proc_data_path)

        try:
            output_file = os.path.join(proc_data_path, file_name + '.h5')
            peth_df.to_hdf(output_file, key='df', mode='w')
            print(f'Saved multi-mouse PETH dataframe to {output_file}')

        except Exception as e:
            print('Could not save as HDF5:', e)
            output_file = os.path.join(proc_data_path, file_name + '.pkl')
            peth_df.to_pickle(output_file)
            print(f'Saved multi-mouse PETH dataframe to {output_file}')

        with open(os.path.join(proc_data_path, file_name + '_params.json'), 'w') as f:
            json.dump(params, f)

    return peth_df


def build_peth_table_parallel_inflection(trial_table, unit_table, params, proc_data_path):
    """
    Build peri-event time histogram dataframe from trial and unit tables using multiprocessing for speed,
    before and after inflection points.
    :param trial_table: Trial table.
    :param unit_table: Unit table.
    :param params: Dictionary of peri-event time histogram parameters.
    :param proc_data_path: Path to save processed data table.
    :return: Peri-event time histogram table.
    """
    print('Building inflection PETH table with multiprocessing...')

    # Select for passive, optionally
    if not params['include_passive']:
        trial_table = behavior_analysis_utils.keep_active_trials(trial_table)

    # Map performance to outcome in trial_table
    trial_table['outcome'] = trial_table['perf'].astype(int).map(TRIAL_MAP)
    trial_table['trial_id_type'] = trial_table.groupby(['mouse_id', 'trial_type'])['trial_id'].cumcount()
    trial_table['trial_id_type_context'] = trial_table.groupby(['mouse_id', 'trial_type', 'context'])[
        'trial_id'].cumcount()

    # Step 1: Get first_active_id per mouse_id
    first_active = trial_table[trial_table["context"] == "active"].groupby("mouse_id")["trial_id"].min().rename(
        "first_active_id")
    trial_table = trial_table.merge(first_active, on="mouse_id", how="left")

    # Step 2: Get last_passive_id (trial_id just before first_active_id)
    def find_last_passive_id(sub_df):
        threshold = sub_df["first_active_id"].iloc[0]
        return sub_df[sub_df["trial_id"] < threshold]["trial_id"].max()

    last_passive = trial_table.groupby("mouse_id").apply(find_last_passive_id).rename("last_passive_id")
    trial_table = trial_table.merge(last_passive, on="mouse_id", how="left")

    # Step 3: convert learning trial index (whisker-based) to session_trial_id
    whisker_df = trial_table[(trial_table["trial_type"] == "whisker_trial") & (
                trial_table["context"] == "active")]  # get trials where learning trial was computed with
    matched_trials = whisker_df[whisker_df["learning_trial"] == whisker_df[
        "trial_id_type_context"]]  # find match trial index matching learning trial (whisker only)
    learning_trial_id = matched_trials.groupby("mouse_id")["trial_id"].first().rename(
        "learning_trial_id")  # get session-wide trial id
    trial_table = trial_table.merge(learning_trial_id, on="mouse_id", how="left")

    # Check if trial is before inflection or after inflection row-wise
    trial_table['pre_inflection'] = trial_table.apply(lambda row: row['trial_id'] < row['learning_trial_id'], axis=1)
    trial_table['post_inflection'] = trial_table.apply(lambda row: row['trial_id'] >= row['learning_trial_id'], axis=1)

    # Step 4: Compute trial_id column relative to learning_trial_id, for each trial_type
    def compute_relative_trial_index(group, learning_trial_id):
        """Compute trial index relative to the learning trial (always a whisker trial)."""
        group = group.sort_values('trial_id')  # Ensure trials are sorted in order

        # Assign negative indices before, 0 at learning trial, and positive indices after
        group['trial_id_from_inflection'] = (
            group['trial_id'].apply(lambda x:
                                    -sum(group['trial_id'] < learning_trial_id) +
                                    sum(group['trial_id'] <= x) - 1
                                    )
        )

        return group

    # Apply function separately for each (mouse_id, trial_type)
    def process_mouse_group(mouse_group):
        try:
            learning_trial_id = \
            mouse_group.loc[mouse_group['trial_id'] == mouse_group['learning_trial_id'].iloc[0], 'trial_id'].values[0]
        except IndexError as err:
            print('Error finding learning trial id for mouse:', mouse_group['mouse_id'].iloc[0], err)
            learning_trial_id = None
        return mouse_group.groupby("trial_type", group_keys=False).apply(compute_relative_trial_index,
                                                                         learning_trial_id)

    trial_table = trial_table.groupby("mouse_id", group_keys=False).apply(process_mouse_group)

    # Step 5: define custom trial ranges to compute PETHs over
    block_mapping = {
        (-10, -6): 'preinflection',
        (-5, -1): 'during',
        (0, 10): 'postinflection',
    }

    # Function to assign block name based on trial_id_from_inflection
    def assign_block(trial_id_from_inflection, min_trial, max_trial):
        # Adjust the ranges to fit within the available trials, if necessary
        for (start, end), block_name in block_mapping.items():
            # Ensure that ranges are within the possible trial range
            adjusted_start = max(start, min_trial)
            adjusted_end = min(end, max_trial)

            if adjusted_start <= trial_id_from_inflection <= adjusted_end:
                return block_name

        # If no block is found, assign 'Uncategorized' for trial_id_from_inflection outside defined ranges
        return 'Uncategorized'

    # Apply function to create block_id column
    trial_table['block_id'] = trial_table.groupby(['mouse_id', 'trial_type'])['trial_id_from_inflection'] \
        .transform(lambda x: x.apply(lambda y: assign_block(y, x.min(), x.max())))

    # Keep only specific blocks for PETH calculation
    params['all_blocks'] = True
    trial_table = trial_table[trial_table['block_id'].isin(list(block_mapping.values()))]

    # -------------
    # Compute PETHs
    # -------------

    # Group trial table by mouse_id and pass it along with the cluster for each unit
    unit_args = [(cluster, trial_table[trial_table['mouse_id'] == cluster['mouse_id']], params)
                 for idx, cluster in unit_table.iterrows()]

    # Ensure that all spike_times are numpy arrays (required by multiprocessing)
    for cluster in unit_args:
        cluster[0]['spike_times'] = np.array(cluster[0]['spike_times'])

    # Set up multiprocessing pool
    with Pool(os.cpu_count() - 5) as pool:
        # Distribute computation of PETHs across processes
        results = pool.map(compute_peth_for_unit_block, unit_args)

    # Flatten list of lists into a single list
    peth_list = [item for sublist in results for item in sublist]

    # Convert list to DataFrame
    peth_df = pd.DataFrame(peth_list)

    # Add metadata columns
    peth_df['bin_size'] = params['bin_size']
    peth_df['time_start'] = params['time_start']
    peth_df['time_stop'] = params['time_stop']
    peth_df['baseline_correction'] = params['baseline_correction']

    # Save the PETH table to feather format
    if proc_data_path is not None:
        file_name = 'peth_table_inflection'
        if not os.path.exists(proc_data_path):
            os.makedirs(proc_data_path)

        try:
            output_file = os.path.join(proc_data_path, file_name + '.h5')
            peth_df.to_hdf(output_file, key='df', mode='w')
            print(f'Saved multi-mouse PETH dataframe to {output_file}')


        except Exception as e:
            print('Could not save as HDF5:', e)
            output_file = os.path.join(proc_data_path, file_name + '.pkl')
            peth_df.to_pickle(output_file)
            print(f'Saved multi-mouse PETH dataframe to {output_file}')

        with open(os.path.join(proc_data_path, file_name + '_params.json'), 'w') as f:
            json.dump(params, f)

    return peth_df


def build_session_dynamics_table(trial_table, unit_table, params, proc_data_path):  # TODO: make it parallel
    """
    Build session trial-by-trial dynamics dataframe from trial and unit tables.
    :param trial_table:
    :param unit_table:
    :param params:
    :param proc_data_path:
    :return:
    """
    print('Building session dynamics PETH table...')

    trial_table = pd.DataFrame(trial_table)
    unit_table = pd.DataFrame(unit_table)

    # Remove passive data, early licks, association trials
    trial_table = trial_table[trial_table['context'] != 'passive']
    trial_table = trial_table[trial_table['perf'] != 6]

    # Get spike train data parameters
    bin_size = params['bin_size']
    time_start = params['time_start']
    time_stop = params['time_stop']
    baseline_correction = params['baseline_correction']

    # Initialize table
    sess_dyn_df = []

    # Map perf to outcome
    trial_table['outcome'] = trial_table['perf'].astype(int).map(TRIAL_MAP)

    for trial_type in ['whisker_trial', 'auditory_trial', 'no_stim_trial']:

        # Get trial subsets, skip if no such trial type
        trial_table_sub = trial_table[(trial_table['trial_type'] == trial_type)]
        if len(trial_table_sub.index) == 0:
            continue

        event_times = trial_table_sub['start_time'].values
        n_trials = len(event_times)

        if trial_type == 'whisker_trial':
            artifact_correction = params['artifact_correction']
        else:
            artifact_correction = False

        # Iterate over clusters to get spike times aligned
        for idx, cluster in unit_table.iterrows():
            spike_times = cluster['spike_times']

            peth = compute_unit_peri_event_histogram(spike_times=spike_times,
                                                     event_times=event_times,
                                                     bin_size=bin_size,
                                                     time_start=time_start,
                                                     time_stop=time_stop,
                                                     artifact_correction=artifact_correction)

            if baseline_correction:
                # Get baseline indices before alignment index
                bas_start = 0
                baseline_duration = abs(time_start)  # seconds
                bas_stop = int(baseline_duration / bin_size)  # this works if symmetrical window
                peth = peth - compute_baseline_from_peth(peth, bas_start=bas_start, bas_stop=bas_stop)

            cluster_dict = cluster.to_dict()
            cluster_info = {k: [v] * n_trials for k, v in cluster_dict.items()}
            cluster_info['spike_counts'] = list(peth)  # splits along trials
            cluster_info['bin_size'] = [bin_size] * n_trials
            cluster_info['time_start'] = [time_start] * n_trials
            cluster_info['time_stop'] = [time_stop] * n_trials
            cluster_info['artifact_correction'] = [artifact_correction] * n_trials
            cluster_info['baseline_correction'] = [baseline_correction] * n_trials
            cluster_info['trial_type'] = [trial_type] * n_trials
            cluster_info['lick_flag'] = trial_table_sub['lick_flag'].values
            cluster_info['reward_group'] = trial_table['reward_group'].unique()[0]
            cluster_info['session_id'] = trial_table['session_id'].unique()[0]
            cluster_info['outcome'] = trial_table_sub['outcome'].values
            cluster_info['trial_index'] = trial_table_sub.index
            cluster_info['trial_type_index'] = np.arange(n_trials)
            cluster_info_df = pd.DataFrame.from_dict(cluster_info)
            sess_dyn_df.append(cluster_info_df)

            # for trial_type_index, event_time in enumerate(event_times):
    #
    #
    #    spike_times_in_window = spike_times[(spike_times >= event_time + time_start) & (spike_times < event_time + time_stop)]
    #    spike_times_in_window -= event_time
    #    spike_counts = np.histogram(spike_times_in_window, bins=np.arange(time_start, time_stop + bin_size, bin_size), density=False)[0]
    #
    #    # Fill cluster info
    #    cluster_info = cluster.to_dict()
    #    cluster_info['spike_counts'] = spike_counts
    #    cluster_info['bin_size'] = bin_size
    #    cluster_info['time_start'] = time_start
    #    cluster_info['time_stop'] = time_stop
    #    cluster_info['artifact_correction'] = artifact_correction
    #    cluster_info['baseline_correction'] = baseline_correction
    #    cluster_info['trial_type'] = trial_type
    #    cluster_info['lick_flag'] = trial_table_sub['lick_flag'].values[trial_type_index]
    #    cluster_info['reward_group'] = trial_table['reward_group'].unique()[0]
    #    cluster_info['session_id'] = trial_table['session_id'].unique()[0]
    #    cluster_info['outcome'] = trial_table_sub['outcome'].values[trial_type_index]
    #    cluster_info['trial_type_index'] = trial_type_index
    #
    #    # Append to list
    #    sess_dyn_df.append(cluster_info)

    # Concatenate all trial-type spikes
    sess_dyn_df = pd.concat(sess_dyn_df)

    # Save PETH table as processed data file
    if proc_data_path is not None:
        file_name = 'peth_table_trial_dynamics'
        if not os.path.exists(proc_data_path):
            os.makedirs(proc_data_path)

        try:
            output_file = os.path.join(proc_data_path, file_name + '.h5')
            sess_dyn_df.to_hdf(output_file, key='df', mode='w')
            print(f'Saved multi-mouse PETH dataframe to {output_file}')


        except Exception as e:
            print('Could not save as HDF5:', e)
            output_file = os.path.join(proc_data_path, file_name + '.pkl')
            sess_dyn_df.to_pickle(output_file)
            print(f'Saved multi-mouse PETH dataframe to {output_file}')

        with open(os.path.join(proc_data_path, file_name + '_params.json'), 'w') as f:
            json.dump(params, f)

    return sess_dyn_df




def _extract_imec_id(df):
    """
    Return a copy of the dataframe with an integer imec_id column.

    If imec_id already exists, it is converted to an integer.
    Otherwise it is extracted from electrode_group strings such as:
        imec0_shank0 -> 0
        imec1        -> 1
    """
    df = df.copy()

    if "imec_id" in df.columns:
        df["imec_id"] = (
            df["imec_id"]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .astype(int)
        )

    elif "electrode_group" in df.columns:
        df["imec_id"] = (
            df["electrode_group"]
            .astype(str)
            .str.extract(r"imec(\d+)", expand=False)
            .astype(int)
        )

    else:
        raise ValueError(
            "DataFrame contains neither 'imec_id' nor 'electrode_group'."
        )

    return df


def merge_unit_quantifications(unit_table, *dfs, verbose=True):
    """
    Merge one or more unit-level dataframes onto unit_table.

    Merge keys:
        mouse_id
        session_id
        cluster_id
        imec_id

    imec_id is extracted from electrode_group when necessary.

    Parameters
    ----------
    unit_table : pd.DataFrame
        Primary dataframe.

    *dfs : pd.DataFrame
        Additional dataframes to merge.

    verbose : bool
        Print alignment diagnostics.

    Returns
    -------
    pd.DataFrame
    """

    base = _extract_imec_id(unit_table)

    keys = [
        "mouse_id",
        "session_id",
        "imec_id",
        "cluster_id",
    ]

    for i, df in enumerate(dfs):

        other = _extract_imec_id(df)

        # Ensure merge keys exist
        missing = [k for k in keys if k not in other.columns]
        if missing:
            raise ValueError(
                f"DataFrame {i} is missing merge keys: {missing}"
            )

        # Check duplicate merge keys
        dup = other.duplicated(keys)
        if dup.any():
            print(
                f"\nWARNING: DataFrame {i} contains "
                f"{dup.sum()} duplicated merge keys."
            )
            print(
                other.loc[dup, keys]
                .sort_values(["mouse_id", "session_id"])
            )

        # Compare keys
        compare = base[keys].merge(
            other[keys],
            how="outer",
            indicator=True,
        )

        missing_in_other = compare["_merge"] == "left_only"
        missing_in_base = compare["_merge"] == "right_only"

        if verbose:

            if missing_in_other.any():
                print(
                    f"\nDataFrame {i}: "
                    f"{missing_in_other.sum()} unit(s) from unit_table "
                    "were not found."
                )

                summary = (
                    compare.loc[missing_in_other]
                    .groupby(["mouse_id", "session_id"])
                    .size()
                    .rename("n_missing")
                    .reset_index()
                    .sort_values(["mouse_id", "session_id"])
                )

                print(summary.to_string(index=False))

            if missing_in_base.any():
                print(
                    f"\nDataFrame {i}: "
                    f"{missing_in_base.sum()} unit(s) are not present "
                    "in unit_table."
                )

                summary = (
                    compare.loc[missing_in_base]
                    .groupby(["mouse_id", "session_id"])
                    .size()
                    .rename("n_extra")
                    .reset_index()
                    .sort_values(["mouse_id", "session_id"])
                )

                print(summary.to_string(index=False))

        # Merge only new columns
        cols_to_merge = [
            c for c in other.columns
            if c not in keys and c not in base.columns
        ]

        base = base.merge(
            other[keys + cols_to_merge],
            on=keys,
            how="left",
            validate="one_to_one",
        )

    return base

def compute_presence_ratio(unit_df, spike_times_col='spike_times', bin_size=60.0,
                            session_col='session_id'):
    """Compute the fraction of 60-s recording bins containing >=1 spike, per unit.

    Recording boundaries (earliest -> latest spike) are computed PER SESSION
    (grouped by session_col), not pooled across the whole unit_df - so units
    from different sessions are scored against their own session's duration.

    Returns
    -------
    pd.Series
        Presence ratio for each unit, aligned with unit_df.index.
    """
    out = pd.Series(0.0, index=unit_df.index)

    for _, session_df in unit_df.groupby(session_col):
        all_spikes = [np.asarray(s) for s in session_df[spike_times_col]]
        non_empty = [s for s in all_spikes if len(s) > 0]
        if not non_empty:
            continue

        rec_start = min(np.min(s) for s in non_empty)
        rec_end = max(np.max(s) for s in non_empty)
        n_bins = int(np.ceil((rec_end - rec_start) / bin_size))
        if n_bins == 0:
            continue

        for idx, s in zip(session_df.index, all_spikes):
            if len(s) == 0:
                continue
            bin_indices = ((s - rec_start) // bin_size).astype(int)
            n_present_bins = len(np.unique(bin_indices))
            out.loc[idx] = n_present_bins / n_bins

    return out


def compute_coverage_ratio(unit_df, spike_times_col='spike_times', session_col='session_id'):
    """Fraction of the recording duration spanned by each unit's own spike train,
    reimplemented from baseline_analysis.py's filter_units_by_quality
    (cicada_analysis/templates/baseline_analysis.py) - same formula, applied
    here directly rather than importing that function (which expects to run
    as part of its own quality-filtering pipeline).

    Recording duration is derived PER SESSION (grouped by session_col): earliest
    first-spike and latest last-spike across ALL units within that session's
    unit_df (should be the full, unfiltered per-session unit table - not already
    subset to e.g. bc_label=='good' - since the population span is what
    "coverage" is relative to).

    Returns a pd.Series aligned to unit_df's index; 1.0 = spikes span the full
    recording, 0.0 for units with <2 spikes or an unusable duration.
    """
    out = pd.Series(0.0, index=unit_df.index)

    for _, session_df in unit_df.groupby(session_col):
        all_spikes = [np.asarray(s) for s in session_df[spike_times_col]]
        firsts = [s[0] for s in all_spikes if len(s) > 0]
        lasts = [s[-1] for s in all_spikes if len(s) > 0]
        if firsts and lasts:
            rec_start = min(firsts)
            rec_dur = max(lasts) - rec_start
        else:
            rec_start, rec_dur = 0.0, 0.0

        for idx, s in zip(session_df.index, all_spikes):
            if len(s) > 1 and rec_dur > 0:
                out.loc[idx] = (s[-1] - s[0]) / rec_dur

    return out



def compute_presence_coverage_metrics(unit_table):
    """ Computes presence and coverage ratio to units."""
    unit_table['coverage_ratio'] = compute_coverage_ratio(unit_table)
    unit_table['presence_ratio'] = compute_presence_ratio(unit_table)
    return unit_table

DEFAULT_METRIC_THRESHOLDS = { #min/max thresholds for good unit classification, based on Bombcell et al. 2023
    # Bombcell metrics (min val, max val)
    "nSpikes":                           (300,   None),
    "percentageSpikesMissing_gaussian":  (None,  20),
    "fractionRPVs_estimatedTauR":        (None,  0.1),
    "maxDriftEstimate":                  (100,   1000),
    "presenceRatio":                     (0.7,    None), #bombcell implementation
    "isolationDistance" :                (20,    None),
    "Lratio":                            (None,  0.3),
    # Our metrics
    "presence_ratio": (0.5, None),
    "coverage_ratio": (0.9, None),
    "drift_shift_test_pval": (0.01, None),
    "drift_abs_r": (0.5, None),
}
ROUTE_THRESHOLDS = {
    "nSpikes": "mua",
    "percentageSpikesMissing_gaussian": "mua",
    "percentageSpikesMissing_symmetric": "mua",
    "fractionRPVs_estimatedTauR": "mua",
    "presenceRatio": "mua",
    "isolationDistance": "mua",
    "Lratio": "mua",
    "driftIndependence": "mua",
    "presence_ratio": "mua",
    "coverage_ratio": "mua",
}

# metrics folded into a single joint criterion - see docstring below
JOINT_METRICS = ("drift_abs_r", "drift_shift_test_pval")


def _metric_ok(vals, lo, hi):
    """NaN -> ignored (True); else in-range check against (lo, hi), inf-padded."""
    lo = -np.inf if lo is None else lo
    hi = np.inf if hi is None else hi
    return np.where(np.isnan(vals), True, (vals >= lo) & (vals <= hi))


def classify_units_quality(unit_table: pd.DataFrame,
                            thresholds: dict = None,
                            exclude: list = ['Lratio', 'isolationDistance', 'presenceRatio', 'maxDriftEstimate'],
                            label_col: str = "quality_label") -> pd.DataFrame:
    """
    Vectorized bombcell-style good/mua classification. A NaN value for a metric is
    IGNORED for that unit (doesn't count as a fail), not treated as an automatic fail.
    exclude: metric names to skip entirely for ALL units (e.g. ['isolationDistance']).
    label_col: new quality metric categorical.

    drift_abs_r and drift_shift_test_pval are evaluated JOINTLY as a single combined
    criterion rather than two independent metrics: a unit only fails the drift check
    if BOTH are simultaneously out of range. Passing either one on its own is enough
    to pass the combined check (protects against either metric alone being noisy).
    Excluding either name in `exclude` drops the whole joint check.
    """
    thresholds = thresholds or DEFAULT_METRIC_THRESHOLDS
    exclude = set(exclude or [])
    nonsoma_mask = unit_table["bc_label"].eq("non-soma")
    out = unit_table.copy()
    n = len(unit_table)
    pass_mask = np.ones(n, dtype=bool)

    cols = [m for m in thresholds if m not in exclude and m not in JOINT_METRICS
            and m in unit_table.columns]
    if cols:
        lo = np.array([thresholds[m][0] if thresholds[m][0] is not None else -np.inf for m in cols])
        hi = np.array([thresholds[m][1] if thresholds[m][1] is not None else np.inf for m in cols])
        vals = unit_table[cols].to_numpy(dtype=float, copy=False)   # (n, k)
        in_range = (vals >= lo) & (vals <= hi)
        ok = np.where(np.isnan(vals), True, in_range)
        pass_mask &= ok.all(axis=1)

    if not exclude.intersection(JOINT_METRICS):
        present = [m for m in JOINT_METRICS if m in unit_table.columns]
        if len(present) == 2:
            r_ok = _metric_ok(unit_table["drift_abs_r"].to_numpy(dtype=float, copy=False),
                               *thresholds.get("drift_abs_r", (None, None)))
            p_ok = _metric_ok(unit_table["drift_shift_test_pval"].to_numpy(dtype=float, copy=False),
                               *thresholds.get("drift_shift_test_pval", (None, None)))
            pass_mask &= (r_ok | p_ok)          # fail only if BOTH fail together
        elif present:
            missing = [m for m in JOINT_METRICS if m not in unit_table.columns]
            warnings.warn(f"joint drift check needs both {JOINT_METRICS}; "
                           f"missing {missing} - skipping drift check entirely")

    out[label_col] = np.where(pass_mask, "good", "mua")
    out[label_col] = np.where(nonsoma_mask, "non-soma", out[label_col])
    return out


def keep_active_from_whisker_onset(trial_df):
    """
    Remove auditory blocks at onset of session, where mice were not yet engaged in the task, before whisker introduction
    :param trial_df: trial table dataframe with active trials only
    :return:
    """
    print('Keeping active trials and removing auditory onset blocks...')

    # Keep active trials
    trial_df = trial_df[
        (~trial_df['context'].isin(['passive']))
        # & (trial_df['perf'] != 6)
        # & (trial_df['early_lick'] == 0)
    ]
    print(f'Number of active trials: {len(trial_df)}')
    df = trial_df.copy()
    print('Getting whisker trial indices...')

    # Find first whisker trial per mouse
    first_whisker_id = (
        df[df['trial_type'] == 'whisker_trial']
        .groupby('mouse_id')['trial_id']
        .min()
        .rename('first_whisker_id')
    )

    # Merge to get first whisker trial per mouse
    df = df.merge(first_whisker_id, on='mouse_id', how='left')

    # Keep only trials >= first whisker trial
    df = df[df['trial_id'] >= df['first_whisker_id']].copy()
    # Reindex trial_id to start at 0 from first whisker trial
    df['trial_id'] = df['trial_id'] - df['first_whisker_id']

    # Define also a whisker_trial_id, just for whisker trials
    df['whisker_trial_id'] = np.nan
    whisker_mask = df['trial_type'] == 'whisker_trial'
    df.loc[whisker_mask, 'whisker_trial_id'] = df.loc[whisker_mask].groupby('mouse_id').cumcount()
    df['whisker_trial_id'] = df['whisker_trial_id'].astype('Int64')  # keep as nullable integer

    # Drop helper column
    df.drop(columns='first_whisker_id', inplace=True)

    return df


def keep_passive_mice(data_df):
    print('Filtering for mice with passive pre/post data...')
    mouse_ids = data_df['mouse_id'].unique()
    passive_mouse_ids = []
    for m in mouse_ids:
        if m.startswith('AB'):
            try:
                if int(m[2:]) >= 116:
                    if m not in ['AB144', 'AB155']:
                        passive_mouse_ids.append(m)
            except ValueError:
                continue  # skip if name is wrong
        elif m.startswith('MH'):
            if m not in ['MH013', 'MH062']:  # no passive post for MH013
                passive_mouse_ids.append(m)
    return data_df[data_df['mouse_id'].isin(passive_mouse_ids)]


def keep_shared_areas(data_df, nomenclature, n_min_units=10, n_min_mice=3):
    print(f'Filtering for R+/R- shared areas in {nomenclature} with at least {n_min_units} units '
          f'and {n_min_mice} mice in each reward group...')

    # Intersect areas across reward groups
    areas_rplus = data_df[data_df['reward_group'] == 1][nomenclature].unique()
    areas_rminus = data_df[data_df['reward_group'] == 0][nomenclature].unique()
    areas_intersect = set(areas_rplus).intersection(set(areas_rminus))
    print(f'Found {len(areas_intersect)} R+/R- shared areas: {areas_intersect}')

    # Count unique elements
    if n_min_units > 0 or n_min_mice > 0:
        bc_mask = data_df['bc_label'].isin(['good', 'mua'])

        # Count unique units per area and reward group
        n_units_rplus = (
            data_df[(data_df['reward_group'] == 1) & bc_mask]
            .groupby(nomenclature)['unit_id']
            .nunique()
        )
        n_units_rminus = (
            data_df[(data_df['reward_group'] == 0) & bc_mask]
            .groupby(nomenclature)['unit_id']
            .nunique()
        )

        # Count unique mice per area and reward group
        n_mice_rplus = (
            data_df[(data_df['reward_group'] == 1) & bc_mask]
            .groupby(nomenclature)['mouse_id']
            .nunique()
        )
        n_mice_rminus = (
            data_df[(data_df['reward_group'] == 0) & bc_mask]
            .groupby(nomenclature)['mouse_id']
            .nunique()
        )

        # Filter shared areas based on both unit and mouse counts
        shared_areas = []
        for area in areas_intersect:
            units_ok = (
                    (area in n_units_rplus.index and n_units_rplus[area] >= n_min_units)
                    and (area in n_units_rminus.index and n_units_rminus[area] >= n_min_units)
            )
            mice_ok = (
                    (area in n_mice_rplus.index and n_mice_rplus[area] >= n_min_mice)
                    and (area in n_mice_rminus.index and n_mice_rminus[area] >= n_min_mice)
            )
            if units_ok and mice_ok:
                shared_areas.append(area)

        removed_areas = areas_intersect - set(shared_areas)
        if len(removed_areas) > 0:
            print(f'Removed {len(removed_areas)} areas with insufficient counts:')
            for area in removed_areas:
                print(f"  {area}: "
                      f"R+ {n_units_rplus.get(area, 0)}u/{n_mice_rplus.get(area, 0)}m, "
                      f"R- {n_units_rminus.get(area, 0)}u/{n_mice_rminus.get(area, 0)}m")
    else:
        shared_areas = list(areas_intersect)

    print(f'Keeping {len(shared_areas)} shared areas meeting both unit and subject thresholds.')
    if len(shared_areas) > 0:
        print("Shared areas:", shared_areas)

    # Filter dataset
    data_df = data_df[data_df[nomenclature].isin(shared_areas)]
    return data_df, shared_areas


def keep_units_params(unit_df, filter_params):
    """
    Keep only units matching filter parameters.
    :param unit_df: pd.DataFrame , unit table.
    :param filter_params: dict, keys are column names, values are values to keep (or list of values).
    :return:
    """
    print('Filtering units...')
    for key, val in filter_params.items():
        if key not in unit_df.columns:
            print(f'Warning: {key} not in unit table columns, skipping this filter.')
        else:
            if key == 'bc_label':
                if isinstance(val, str):
                    val = [val]
                unit_df = unit_df[unit_df[key].isin(val)]
            if key == 'firing_rate':
                unit_df = unit_df[unit_df[key].astype(float) >= val]

                # ADD filters as needed

    return unit_df


def keep_mouse_neuron_pairs(target_df, pairs_df):
    """
    Keep only specified mouse/neuron pairs in the PETH dataframe.
    :param target_df: pd.DataFrame , target dataframe to filter.
    :param pairs_df: pd.DataFrame , dataframe with 'mouse_id' and 'neuron_id' columns specifying pairs to keep.
    :return:
    """
    # Ensure proper columns
    if not {'mouse_id', 'neuron_id'}.issubset(pairs_df.columns):
        raise ValueError("`pairs_df` must contain 'mouse_id' and 'neuron_id' columns.")

    # Merge to select matching rows
    matched = target_df.merge(pairs_df[['mouse_id', 'neuron_id']],
                              on=['mouse_id', 'neuron_id'],
                              how='inner')  # inner join keeps only matches
    print(f'Filtered to {len(matched)} mouse/neuron pairs from {len(target_df)} total.')
    return matched


def keep_units_with_conditions(peth_df, conditions):
    # Keep units/mice that contain all conditions to be compared
    conditions = list(conditions)
    # Ignore the conditions that include diff as I don't always plot them but they are computed
    # conditions = [c for c in conditions if 'diff' not in c]

    # Keep only units that have PETHs for exactly all of those outcomes
    units_with_all_conditions = (
        peth_df.groupby('unit_id')['outcome']
        .apply(lambda x: set(conditions).issubset(set(x)))
        # .apply(lambda x: set(conditions) == set(x))
    )

    # Filter the table to only include those units
    units_to_keep = units_with_all_conditions[units_with_all_conditions].index
    peth_df_filtered = peth_df[peth_df['unit_id'].isin(units_to_keep)]

    # Number of units excluded
    n_excluded = len(units_with_all_conditions) - len(units_to_keep)

    # Excluded mice
    excluded_mice = peth_df[~peth_df['unit_id'].isin(units_to_keep)]['mouse_id'].unique()
    print(f'Filtered out {n_excluded} units missing some of the required conditions: {conditions}.')
    print(f'Excluded mice: {excluded_mice}')

    return peth_df_filtered


def get_passive_conditions_fast(peth_df):
    print('Computing passive conditions (pre-post diffs and diff-of-diffs) fast...')

    # Keep metadata
    metadata_cols = [c for c in peth_df.columns if c not in ['unit_id', 'peth', 'outcome']]
    metadata_df = peth_df.groupby('unit_id')[metadata_cols].first().reset_index()

    # Filter unit_id that do not have both passive pre and passive post outcomes
    # Step 1: Find all units and the contexts they have
    contexts_per_unit = peth_df.groupby('unit_id')['context'].unique()

    # Step 2: Identify units missing one or both of the desired contexts
    required = {'passive_pre', 'passive_post'}
    bad_units = [u for u, ctxs in contexts_per_unit.items() if not required.issubset(set(ctxs))]
    bad_mice = peth_df[peth_df['unit_id'].isin(bad_units)]['mouse_id'].unique()

    # Step 3: Count and print them
    # print(f"Units missing 'passive_pre' or 'passive_post': {len(bad_units)}")
    # print("Excluded units:", bad_units)
    # print(f"Affected mice: {bad_mice}")

    # Step 4: Exclude them from the dataframe
    df_filtered = peth_df[~peth_df['unit_id'].isin(bad_units)].copy()

    # Optional: check result
    print(f"Remaining units: {df_filtered['unit_id'].nunique()}")
    # -------------

    # Keep only passive, no-lick trials
    df = peth_df.query("context in ['passive_pre','passive_post'] and lick_flag == 0").copy()

    # Make a key for each condition
    df["cond_key"] = df["trial_type"] + "_" + df["context"]

    # Pivot into wide format: one row per unit_id, one column per cond_key
    wide = (
        df.pivot(index="unit_id", columns="cond_key", values="peth")
        .reindex(columns=[
            "whisker_trial_passive_pre", "whisker_trial_passive_post",
            "auditory_trial_passive_pre", "auditory_trial_passive_post"
        ])
    )

    # Compute diffs vectorized
    wide["whisker_diff"] = wide["whisker_trial_passive_post"] - wide["whisker_trial_passive_pre"]
    wide["auditory_diff"] = wide["auditory_trial_passive_post"] - wide["auditory_trial_passive_pre"]
    wide["auditory_diff_whisker_diff"] = wide["auditory_diff"] - wide["whisker_diff"]

    # Rename for output consistency
    wide = wide.rename(columns={
        "whisker_trial_passive_pre": "whisker_passive_pre",
        "whisker_trial_passive_post": "whisker_passive_post",
        "auditory_trial_passive_pre": "auditory_passive_pre",
        "auditory_trial_passive_post": "auditory_passive_post",
    }).reset_index()

    # Melt to long format: one row per unit_id × outcome
    df_long = wide.melt(
        id_vars=["unit_id"],
        value_vars=[
            "whisker_passive_pre", "whisker_passive_post",
            "auditory_passive_pre", "auditory_passive_post",
            "whisker_diff", "auditory_diff", "auditory_diff_whisker_diff"
        ],
        var_name="outcome",
        value_name="peth"
    )

    # Merge back metadata
    df_long = df_long.merge(metadata_df, on="unit_id", how="left")

    return df_long


def get_block_perf_conditions_fast(peth_df):
    """
    Compute per-unit differences of neural activity for block_perf_type (high vs low),
    trial_type (whisker vs auditory), and outcome (hit vs miss).

    Returns a long-format DataFrame with columns: unit_id, outcome, peth, plus metadata.
    """
    print("Computing high/low state differences for each trial type...")

    var_to_plot = 'peth' if 'peth' in peth_df.columns else 'trial_var'

    # Keep metadata
    metadata_cols = [c for c in peth_df.columns if
                     c not in ['unit_id', var_to_plot, 'outcome', 'block_perf_type', 'trial_type']]
    metadata_df = peth_df.groupby('unit_id')[metadata_cols].first().reset_index()

    # Step 1: keep only relevant trials
    df = peth_df.copy()

    # Step 2: make a condition key: trial_type + block_perf_type + outcome
    df['cond_key'] = df['outcome'] + "_" + df['block_perf_type']

    # Expected condition keys
    keys = [
        "whisker_hit_high", "whisker_miss_high",
        "whisker_hit_low", "whisker_miss_low",
        "auditory_hit_high",  # "auditory_miss_high",
        "auditory_hit_low",  # "auditory_miss_low"
        "false_alarm_high", "false_alarm_low",
        "correct_reject_high", "correct_reject_low",
        "whisker_stim_high", "whisker_stim_low",
    ]

    # Pivot into wide format: one row per unit_id, one column per cond_key
    wide = (
        df.pivot(index="unit_id", columns="cond_key", values=var_to_plot)
        .reindex(columns=keys)
    )

    # Compute differences
    wide["whisker_hit_high_low_diff"] = wide["whisker_hit_high"] - wide["whisker_hit_low"]
    wide["whisker_miss_high_low_diff"] = wide["whisker_miss_high"] - wide["whisker_miss_low"]
    wide["whisker_high_hit_miss_diff"] = wide["whisker_hit_high"] - wide["whisker_miss_high"]
    wide["whisker_low_hit_miss_diff"] = wide["whisker_hit_low"] - wide["whisker_miss_low"]
    wide["auditory_hit_high_low_diff"] = wide["auditory_hit_high"] - wide["auditory_hit_low"]
    # wide["auditory_miss_high_low_diff"] = wide["auditory_miss_high"] - wide["auditory_miss_low"]
    # wide["auditory_high_hit_miss_diff"] = wide["auditory_hit_high"] - wide["auditory_miss_high"]
    # wide["auditory_low_hit_miss_diff"] = wide["auditory_hit_low"] - wide["auditory_miss_low"]
    wide["false_alarm_high_low_diff"] = wide["false_alarm_high"] - wide["false_alarm_low"]
    wide["correct_reject_high_low_diff"] = wide["correct_reject_high"] - wide["correct_reject_low"]
    wide["whisker_stim_high_low_diff"] = wide["whisker_stim_high"] - wide["whisker_stim_low"]

    # Define keys for melting
    diff_keys = [
        "whisker_hit_high_low_diff", "whisker_miss_high_low_diff",
        "whisker_high_hit_miss_diff", "whisker_low_hit_miss_diff",
        "auditory_hit_high_low_diff",  # "auditory_miss_high_low_diff",
        # "auditory_high_hit_miss_diff", "auditory_low_hit_miss_diff",
        "false_alarm_high_low_diff", "correct_reject_high_low_diff",
        "whisker_stim_high_low_diff"
    ]

    all_keys = keys + diff_keys

    df_long = wide[all_keys].reset_index().melt(
        id_vars="unit_id",
        value_vars=all_keys,
        var_name="outcome",
        value_name=var_to_plot
    )

    # Merge back metadata
    df_long = df_long.merge(metadata_df, on="unit_id", how="left")

    return df_long


def get_inflection_conditions_fast(peth_df):
    """
    Compute per-unit differences of neural activity for block_id (pre-learning, during, post-learning),
    trial_type (whisker vs auditory), and outcome (hit vs miss).

    Returns a long-format DataFrame with columns: unit_id, outcome, peth, plus metadata.
    """
    print("Computing pre/during/post inflection blocks differences for each trial type...")

    # Keep metadata
    var_to_plot = 'peth' if 'peth' in peth_df.columns else 'trial_var'
    metadata_cols = [c for c in peth_df.columns if c not in ['unit_id', var_to_plot, 'outcome', 'block', 'trial_type']]
    metadata_df = peth_df.groupby('unit_id')[metadata_cols].first().reset_index()

    # Step 1: keep only relevant trials
    df = peth_df.copy()

    # Step 2: make a condition key: trial_type + block_id + outcome
    df['cond_key'] = df['trial_type'] + "_" + df['block']

    # Expected condition keys
    keys = [
        "whisker_trial_preinflection", "whisker_trial_during", "whisker_trial_postinflection",
        "auditory_trial_preinflection", "auditory_trial_during", "auditory_trial_postinflection",
    ]

    # Pivot into wide format: one row per unit_id, one column per cond_key
    wide = (
        df.pivot(index="unit_id", columns="cond_key", values=var_to_plot)
        .reindex(columns=keys)
    )

    # Compute differences
    wide["whisker_pre_post_diff"] = wide["whisker_trial_postinflection"] - wide["whisker_trial_preinflection"]
    wide["auditory_pre_post_diff"] = wide["auditory_trial_postinflection"] - wide["auditory_trial_preinflection"]

    diff_keys = ["whisker_pre_post_diff", "auditory_pre_post_diff"]
    all_keys = keys + diff_keys

    # Rename
    wide = wide.rename(columns={
        "whisker_trial_preinflection": "whisker_preinflection",
        "whisker_trial_during": "whisker_during",
        "whisker_trial_postinflection": "whisker_postinflection",
        "auditory_trial_preinflection": "auditory_preinflection",
        "auditory_trial_during": "auditory_during",
        "auditory_trial_postinflection": "auditory_postinflection",
    }).reset_index()

    df_long = wide.melt(
        id_vars="unit_id",
        value_vars=[
            "whisker_preinflection", "whisker_during", "whisker_postinflection",
            "auditory_preinflection", "auditory_during", "auditory_postinflection",
            "whisker_pre_post_diff", "auditory_pre_post_diff"
        ],
        var_name="outcome",
        value_name=var_to_plot
    )

    # Merge back metadata
    df_long = df_long.merge(metadata_df, on="unit_id", how="left")

    return df_long



