#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: brain_wide_analysis
@file: data_utils.py
"""
# Imports
import sys
import warnings

import numpy as np
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Custom imports
sys.path.insert(0, r"M:\analysis\Axel_Bisi\NWB_reader")
sys.path.insert(0, "/home/bisi/code/NWB_reader")
import NWB_reader_functions as nwb_reader

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
    data['electrode_group'] = elec_group_name

    elec_group_location = [e.location.replace('nan', 'None') for e in elec_group_list]
    elec_group_location_dict = [eval(e) for e in elec_group_location]
    data['location'] = elec_group_location_dict
    data['target_region'] = [e.get('area') for e in elec_group_location_dict]

    return data


def keep_active_trials(data_df):
    """
    Keep active trials, filter out passive trials and excluded trials (early licks, trials without particles, etc.).
    :param data_df: pd.DataFrame with trial information
    :return:
    """
    print('Keeping active trials...')

    data_df = data_df[
        (~data_df.context.isin(['passive']))
    ]
    if 'perf' in data_df.columns:
        data_df = data_df[data_df.perf!=6]
    if 'early_lick' in data_df.columns:
        data_df = data_df[data_df.early_lick==0]
    return data_df


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