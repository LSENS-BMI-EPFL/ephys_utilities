#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: brain_wide_analysis
@file: load_helpers.py
@time: 7/10/2025 12:58 PM
"""

# Imports
import os
import sys
import socket
import numpy as np
import pandas as pd
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing as mp


hostname = socket.gethostname()
if 'haas' in hostname:
    MAX_WORKERS = 100
    ROOT_PATH_AXEL = pathlib.Path('/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4')
    ROOT_PATH_MYRIAM = ROOT_PATH_AXEL

    sys.path.append("/home/bisi/code/NWB_reader")
    import NWB_reader_functions as nwb_reader

else:
    MAX_WORKERS = 20
    ROOT_PATH_AXEL = pathlib.Path(r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Axel_Bisi\combined_results')
    ROOT_PATH_MYRIAM = pathlib.Path(r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Myriam_Hamon\combined_results')
    sys.path.insert(0, r"M:\analysis\Axel_Bisi\NWB_reader")
    import NWB_reader_functions as nwb_reader

# ---------------------------------
# Generic parallel loading helper
# ---------------------------------

def _run_parallel_load(items, worker_fn, max_workers=20, desc="Loading files", show_progress=True):
    """
    Generic threaded loader shared by every load_* function in this module.

    Applies `worker_fn` to each element of `items` using a ThreadPoolExecutor
    (these loaders are I/O-bound: NWB metadata lookups + file existence
    checks + CSV/HDF5/pickle reads, so threads are the right tool, not
    processes).

    Contract for `worker_fn(item)`
    -------------------------------
    Must return a 3-tuple ``(status, data, message)``:
        status  : one of "ok", "missing", "skipped", "failed"
        data    : the loaded payload (DataFrame, dict, ...) when status=="ok",
                  otherwise None
        message : human-readable string to print for "missing"/"failed"
                  (e.g. "[WARN] ... not found for AB123 (path)"), or None
                  for "ok"/"skipped"

    Any exception raised inside `worker_fn` is caught here and converted
    into a "failed" result automatically, so individual worker functions
    don't need a blanket try/except just for that.

    Returns
    -------
    data_list : list
        The `data` payloads for every "ok" item (empty DataFrames are
        dropped, matching the original per-function behaviour), in
        completion order (not necessarily input order).
    counts : dict
        Per-status counts, e.g. {"ok": 8, "missing": 2, "failed": 0, "skipped": 1}.
    """
    data_list = []
    messages = []
    counts = {"ok": 0, "missing": 0, "skipped": 0, "failed": 0}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker_fn, item): item for item in items}
        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(futures), desc=desc, ncols=100)

        for future in iterator:
            item = futures[future]
            try:
                status, data, message = future.result()
            except Exception as e:
                status, data, message = "failed", None, f"[ERROR] Failed to load {item}: {e}"

            counts[status] = counts.get(status, 0) + 1
            if message:
                messages.append(message)
            if status == "ok" and data is not None:
                if hasattr(data, "empty") and data.empty:
                    pass  # keep counted as "ok" but don't concat an empty frame
                else:
                    data_list.append(data)

    if messages:
        print("\n".join(messages))

    return data_list, counts


# ---------------
# Loading helpers
# ---------------

def load_learning_curves_data(path_to_data, subject_ids, max_workers=20):
    """Load whisker-trial learning-curve data for a list of mice, in parallel."""
    print('Loading learning curve data...')

    def load_single(m):
        file_name = f'{m}_whisker_0_whisker_trial_learning_curve_interp.h5'
        file_path = os.path.join(path_to_data, m, 'whisker_0', 'learning_curve', file_name)
        if not os.path.exists(file_path):
            return "missing", None, f"[WARN] No whisker curve for: {m} ({file_path})"
        df_w = pd.read_hdf(file_path)
        return "ok", df_w, None

    data_list, counts = _run_parallel_load(
        subject_ids, load_single, max_workers=max_workers, desc="Loading learning curves"
    )

    if not data_list:
        raise FileNotFoundError("No learning curve files were loaded successfully.")

    data_df = pd.concat(data_list, ignore_index=True)
    return data_df


def load_jaw_onset_data(nwb_files, day_to_analyze, experimenter='AB', max_workers=12):
    """
    Load jaw onset data from NWB files in parallel.
    :param nwb_files: List of NWB file paths.
    :param experimenter: Experimenter identifier ('AB' or 'MH').
    :param max_workers: Number of threads for parallel loading.
    :return: Combined DataFrame of jaw onset times.
    """
    print('Loading jaw onset data...')

    def load_single(nwb_file):
        mouse_id = nwb_reader.get_mouse_id(nwb_file)
        beh_type, day = nwb_reader.get_bhv_type_and_training_day_index(nwb_file)
        if day_to_analyze == 'learning' and day != 0:
            return "skipped", None, None
        elif day_to_analyze == 'expert' and day == 0:
            return "skipped", None, None
        elif day_to_analyze == 'all' and day < 0:
            return "skipped", None, None

        if mouse_id.startswith('AB'):
            data_path = ROOT_PATH_AXEL
        elif mouse_id.startswith('MH'):
            data_path = ROOT_PATH_AXEL
        else:
            return "failed", None, f"[WARN] Unknown experimenter: {experimenter} for {mouse_id}"

        data_path = pathlib.Path(str(data_path).replace('_ks4', ''))
        file_path = os.path.join(data_path, mouse_id, 'dlc_jaw_onset_times.pkl')
        if not os.path.exists(file_path):
            return "missing", None, f"[WARN] Jaw onset data not found for {mouse_id} ({file_path})"

        df = pd.read_pickle(file_path)
        df['mouse_id'] = mouse_id
        return "ok", df, None

    data_list, counts = _run_parallel_load(
        nwb_files, load_single, max_workers=max_workers, desc="Loading jaw onset files"
    )

    if not data_list:
        print("No jaw onset .pkl files were loaded successfully.")
        return None

    jaw_onset_table = pd.concat(data_list, ignore_index=True)
    return jaw_onset_table


def load_task_modulation_data(nwb_files, experimenter, max_workers=20):
    """
    Load task modulation data from NWB files in parallel.
    :param nwb_files: List of NWB file paths.
    :param experimenter: Experimenter identifier ('AB' or 'MH') for path selection.
    :param max_workers:
    :return:
    """
    print('Loading task modulation data...')

    def load_single(nwb_file):
        mouse_id = nwb_reader.get_mouse_id(nwb_file)
        session_id = nwb_reader.get_session_id(nwb_file)

        if experimenter == 'AB':
            data_path = ROOT_PATH_AXEL
        elif experimenter == 'MH':
            data_path = ROOT_PATH_MYRIAM
        else:
            return "failed", None, f"[WARN] Unknown experimenter: {experimenter} for {mouse_id}"

        file_path = os.path.join(data_path, mouse_id, 'whisker_0', 'task_modulation',
                                 f'{mouse_id}_task_modulation_results.csv')

        if not os.path.exists(file_path):
            return "missing", None, f"[WARN] Missing task modulation: {mouse_id} ({file_path})"

        df = pd.read_csv(file_path)
        df['mouse_id'] = mouse_id
        df['session_id'] = session_id
        return "ok", df, None

    data_list, counts = _run_parallel_load(
        nwb_files, load_single, max_workers=max_workers, desc="Loading task-mod. files"
    )

    if not data_list:
        raise FileNotFoundError("No task modulation analysis CSV files were loaded successfully.")

    data_table = pd.concat(data_list, ignore_index=True)

    # Normalize neuron_id column
    #if 'neuron_id' not in data_table.columns and 'neuron_id' in data_table.columns:
    #    data_table['neuron_id'] = data_table['neuron_id']

    data_table = data_table[['mouse_id', 'neuron_id', 'task_modulated']]

    return data_table


def load_roc_analysis_data(nwb_files, experimenter, max_workers=20):
    """
    Load ROC analysis data from NWB files in parallel.
    :param nwb_files: List of NWB file paths.
    :param experimenter: Experimenter identifier ('AB' or 'MH') for path selection.
    :param max_workers:
    :return:
    """
    print('Loading ROC analysis data...')

    def load_single(nwb_file):
        mouse_id = nwb_reader.get_mouse_id(nwb_file)
        session_id = nwb_reader.get_session_id(nwb_file)

        if experimenter == 'AB':
            data_path = ROOT_PATH_AXEL
        elif experimenter == 'MH':
            data_path = ROOT_PATH_MYRIAM
        else:
            return "failed", None, f"[WARN] Unknown experimenter: {experimenter} for {mouse_id}"

        file_path = os.path.join(data_path, mouse_id, 'whisker_0', 'roc_analysis',
                                 f'{mouse_id}_roc_results_new.csv')

        if not os.path.exists(file_path):
            return "missing", None, f"[WARN] Missing ROC: {mouse_id} ({file_path})"

        df = pd.read_csv(file_path)
        df['mouse_id'] = mouse_id
        df['session_id'] = session_id
        return "ok", df, None

    data_list, counts = _run_parallel_load(
        nwb_files, load_single, max_workers=max_workers, desc="Loading ROC files"
    )

    if not data_list:
        raise FileNotFoundError("No ROC analysis CSV files were loaded successfully.")

    data_table = pd.concat(data_list, ignore_index=True)

    if 'neuron_id' not in data_table.columns and 'neuron_id' in data_table.columns:
        data_table['neuron_id'] = data_table['neuron_id']

    print(f"ROC types available: {data_table['analysis_type'].unique().tolist()}")
    data_table['si_sign'] = np.sign(data_table['selectivity'])

    return data_table


def load_wf_analysis_data(nwb_files, experimenter, max_workers=12):
    """
    Load waveform classification analysis data (cortical + striatal) from
    NWB files, in parallel.

    Each mouse contributes up to two per-mouse CSVs from the same
    'whisker_0/waveform_analysis' folder, each already one row per cluster
    (no pivoting needed -- just add each CSV's classification column as-is):
        {mouse_id}_cortical_wf_type.csv  -- required; contributes 'waveform_type'
        {mouse_id}_striatal_type.csv     -- optional; contributes 'striatal_type'

    :param nwb_files: List of NWB file paths.
    :param experimenter: Experimenter identifier ('AB' or 'MH').
    :param max_workers:
    :return: DataFrame indexed by (mouse_id, session_id, electrode_group, cluster_id)
             with 'waveform_type' and 'striatal_type' columns.
    """
    print('Loading waveform classification analysis data...')

    join_cols = ['mouse_id', 'session_id', 'electrode_group', 'cluster_id']

    def load_single(nwb_file):
        mouse_id = nwb_reader.get_mouse_id(nwb_file)
        if experimenter == 'AB':
            data_path = ROOT_PATH_AXEL
        elif experimenter == 'MH':
            data_path = ROOT_PATH_MYRIAM
        else:
            return "failed", None, f"[WARN] Unknown experimenter: {experimenter} for {mouse_id}"

        wf_dir = os.path.join(data_path, mouse_id, 'whisker_0', 'waveform_analysis')
        cortical_path = os.path.join(wf_dir, f'{mouse_id}_cortical_wf_type.csv')
        striatal_path = os.path.join(wf_dir, f'{mouse_id}_striatal_wf_type.csv')

        if not os.path.exists(cortical_path):
            return "missing", None, (f"[WARN] Cortical waveform type file not found for "
                                     f"{mouse_id} at {cortical_path}. Skipping.")

        cortical_df = pd.read_csv(cortical_path)
        if 'waveform_type' not in cortical_df.columns:
            return "failed", None, (f"[WARN] Expected column 'waveform_type' not found in "
                                    f"{cortical_path} (columns: {list(cortical_df.columns)})")
        cortical_keys = [c for c in join_cols if c in cortical_df.columns]
        cortical_df = cortical_df[cortical_keys + ['waveform_type']]

        striatal_df = None
        if os.path.exists(striatal_path):
            striatal_df = pd.read_csv(striatal_path)
            if 'striatal_type' not in striatal_df.columns:
                print(f"  [waveform classification] Expected column 'striatal_type' not found "
                     f"in {striatal_path} (columns: {list(striatal_df.columns)}) -- "
                     f"'striatal_type' will be NaN for this mouse.")
                striatal_df = None
            else:
                striatal_keys = [c for c in join_cols if c in striatal_df.columns]
                striatal_df = striatal_df[striatal_keys + ['striatal_type']]
        else:
            print(f"  [waveform classification] No striatal type file for {mouse_id} "
                 f"({striatal_path}) -- 'striatal_type' will be NaN for this mouse.")

        return "ok", (cortical_df, striatal_df), None

    results, counts = _run_parallel_load(
        nwb_files, load_single, max_workers=max_workers, desc="Loading waveform analysis files"
    )

    if not results:
        raise FileNotFoundError("No cortical waveform type CSV files were loaded successfully.")

    cortical_list, striatal_list = zip(*results)

    # Each CSV is already one row per cluster -- just concatenate, no pivot needed.
    data_table = pd.concat(cortical_list, ignore_index=True)

    # Add the striatal classification as a second column (left join: mice/units
    # without a striatal file just get NaN there, they aren't dropped)
    striatal_frames = [df for df in striatal_list if df is not None]
    if striatal_frames:
        striatal_table = pd.concat(striatal_frames, ignore_index=True)
        data_table = data_table.merge(striatal_table, on=join_cols, how='left')
    else:
        data_table['striatal_type'] = np.nan

    data_table['cluster_id'] = data_table['cluster_id'].astype(str)
    return data_table


def load_motion_dredge_shift_test_results(nwb_files, day_to_analyze, experimenter='AB', max_workers=12):
    """
    Load per-session time-binned drift (motion) shift-test results
    (single_neuron_shift_test_figs.py output) from per-session CSVs, in parallel.

    Parameters
    ----------
    nwb_files : list
        List of NWB file paths.
    experimenter : str, optional
        'AB' or 'MH', selects the results root.
    max_workers : int, optional
        Maximum number of worker threads.

    Returns
    -------
    pandas.DataFrame
        Concatenated motion-drift shift-test results from all successfully
        loaded sessions.
    """
    print('Loading motion-drift (DREDge) shift test results ...')

    if experimenter == 'AB':
        data_path = ROOT_PATH_AXEL
    elif experimenter == 'MH':
        data_path = ROOT_PATH_MYRIAM
    else:
        raise ValueError(f"Unknown experimenter '{experimenter}'. Expected 'AB' or 'MH'.")

    def load_single(nwb_file):
        mouse_id = nwb_reader.get_mouse_id(nwb_file)
        beh, day = nwb_reader.get_bhv_type_and_training_day_index(nwb_file)
        if day_to_analyze == 'learning' and day != 0:
            return "skipped", None, None
        elif day_to_analyze == 'expert' and day == 0:
            return "skipped", None, None
        elif day_to_analyze == 'all' and day < 0:
            return "skipped", None, None

        session_day = f"{beh}_{day}"
        file_path = os.path.join(
            data_path, mouse_id, session_day, 'single_neuron_motion_shift_test',
            f'{mouse_id}_{session_day}_motion_shift_test_results.csv'
        )

        if not os.path.exists(file_path):
            return "missing", None, (f"[WARN] Motion-drift shift test file not found for "
                                     f"{mouse_id} at {file_path}. Skipping.")

        df = pd.read_csv(file_path)
        return "ok", df, None

    data_list, counts = _run_parallel_load(
        nwb_files, load_single, max_workers=max_workers, desc="Loading motion-drift files"
    )

    print(f"\nLoaded {counts['ok']}/{len(nwb_files)} sessions "
         f"({counts['missing']} missing, {counts['failed']} failed, {counts['skipped']} skipped)")

    if not data_list:
        print("[WARN] No motion-drift shift test files loaded.")
        return pd.DataFrame(columns=['mouse_id', 'session_id', 'neuron_id'])

    data_table = pd.concat(data_list, ignore_index=True)
    return data_table


def load_spontaneous_reward_lick_times(nwb_files, day_to_analyze, max_workers=12, load_summary=False):
    """
    Load per-session spontaneous-lick CSVs for a list of NWB files, in
    parallel via ThreadPoolExecutor (I/O-bound: file existence checks + CSV
    reads, so threads are appropriate here, unlike the NWB-processing case).
    Returns
    -------
    pandas.DataFrame indexed by session_id, with columns:
        mouse_id, session_id, spontaneous_licks, reward_times
    """
    print('Loading spontaneous/reward licks data...')

    summary_path = pathlib.Path(ROOT_PATH_AXEL) / "spontaneous_licks"
    summary_path.mkdir(parents=True, exist_ok=True)
    file_path = summary_path / "spontaneous_licks.csv"

    if load_summary:
        print('Loading combined spontaneous/reward licks data...')
        return pd.read_csv(file_path)

    def load_single(nwb_file):
        mouse_id = nwb_reader.get_mouse_id(nwb_file)
        session_id = nwb_reader.get_session_id(nwb_file)  # adjust if named differently
        beh, day = nwb_reader.get_bhv_type_and_training_day_index(nwb_file)
        if day_to_analyze == 'learning' and day != 0:
            return "skipped", None, None
        elif day_to_analyze == 'expert' and day == 0:
            return "skipped", None, None
        elif day_to_analyze == 'all' and day < 0:
            return "skipped", None, None
        if 'whisker' not in beh:
            return "skipped", None, None

        # experimenter/data_path both currently resolve to ROOT_PATH_AXEL (kept as in the original)
        data_path = ROOT_PATH_AXEL
        session_file = os.path.join(data_path, mouse_id, f'{beh}_{day}', 'spontaneous_licks',
                                    f'{session_id}_spontaneous_licks.csv')
        if not os.path.exists(session_file):
            return "missing", None, (f"[WARN] Spontaneous licks data file not found for {mouse_id}, "
                                     f"session {session_id} at {session_file}. Skipping.")

        df = pd.read_csv(session_file)
        record = {
            "mouse_id": mouse_id,
            "session_id": session_id,
            "spontaneous_licks": df.loc[df["event_type"].isin(['single', 'short_cluster', 'bout']), "lick_time"].to_numpy(),
            "reward_times": df.loc[df["event_type"] == "reward_time", "lick_time"].to_numpy(),
        }
        return "ok", record, None

    data_list, counts = _run_parallel_load(
        nwb_files, load_single, max_workers=max_workers, desc="Loading spontaneous licks files"
    )

    print(f"\nLoaded {counts['ok']}/{len(nwb_files)} sessions "
         f"({counts['missing']} missing, {counts['failed']} failed, {counts['skipped']} non-whisker skipped)")

    results = {rec["session_id"]: rec for rec in data_list}
    df = pd.DataFrame.from_dict(results, orient="index").reset_index(drop=True)

    # Save global
    df.to_csv(file_path, index=False)

    return df


# -------------------------------------
# Helpers to merge unit quantifications
# -------------------------------------

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