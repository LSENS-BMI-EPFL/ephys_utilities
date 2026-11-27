#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: ephys_utilities
@file: reference_sheets.py

Single entry point for the dataset reference sheets (Axel 2026-11-27: all analyses load and select data through
ephys_utilities, never with their own read_excel / filters).

Canonical location: share_internal/Axel_Bisi_Share/dataset_info (the copy in analysis/Axel_Bisi/dataset_info is stale).
  - joint_mouse_reference_weight.xlsx: one row per mouse (cohort, exclusions, learning category, video availability)
  - joint_probe_insertion_info.xlsx: one row per probe insertion (probe type, coordinates, setup, target area)

Mouse selection rules:
  - exclude == 1: mouse excluded from everything
  - exclude_ephys == 1: mouse excluded from ephys analyses only (behaviour kept)
  - reward_group: cohort per mouse, R+ (whisker hits rewarded) or R- ('R+proba' mice are not part of the R+ / R- cohorts)
  - learning_category: good / moderate = learner, bad = non-learner (empty = not classified)
"""
import os
from pathlib import Path

import pandas as pd

MOUSE_SHEET = 'joint_mouse_reference_weight.xlsx'
PROBE_SHEET = 'joint_probe_insertion_info.xlsx'
SHARE_CANDIDATES = (Path(r'M:\share_internal'), Path('/mnt/share_internal'))
LEARNER_CATEGORIES = {'good': True, 'moderate': True, 'bad': False}
COHORTS = ('R+', 'R-')


def dataset_info_dir():
    """share_internal/Axel_Bisi_Share/dataset_info on this machine (env SSL_DATASET_INFO overrides)."""
    if os.environ.get('SSL_DATASET_INFO'):
        return Path(os.environ['SSL_DATASET_INFO'])
    for root in SHARE_CANDIDATES:
        if root.exists():
            return root / 'Axel_Bisi_Share' / 'dataset_info'
    raise FileNotFoundError(f'share_internal not found in {SHARE_CANDIDATES}; set SSL_DATASET_INFO')


def load_mouse_reference(path=None):
    """
    Mouse reference sheet, all rows, with derived columns:
      cohort (= reward_group), learner (True / False / NaN from learning_category).
    :param path: optional sheet path (default: canonical share_internal sheet)
    :return: pd.DataFrame, one row per mouse (mouse_id)
    """
    path = Path(path) if path else dataset_info_dir() / MOUSE_SHEET
    df = pd.read_excel(path)
    if 'mouse_id' not in df.columns and 'mouse_name' in df.columns:
        df = df.rename(columns={'mouse_name': 'mouse_id'})
    df['cohort'] = df['reward_group']
    df['learner'] = df['learning_category'].map(LEARNER_CATEGORIES)
    df.attrs['source'] = str(path)
    df.attrs['modified'] = pd.Timestamp(os.path.getmtime(path), unit='s').isoformat()
    return df


def select_mice(scope='behaviour', learners=None, ref=None):
    """
    Included mice of the R+ / R- cohorts.
    :param scope: 'behaviour' (exclude == 0) or 'ephys' (exclude == 0 and exclude_ephys == 0)
    :param learners: None (all), True (learners only) or False (non-learners only)
    :param ref: optional output of load_mouse_reference (default: loaded here)
    :return: pd.DataFrame of the selected mice (mouse_id, cohort, learner, ... all sheet columns)
    """
    if scope not in ('behaviour', 'ephys'):
        raise ValueError("scope must be 'behaviour' or 'ephys'")
    ref = load_mouse_reference() if ref is None else ref
    keep = (ref['exclude'].fillna(0) == 0) & ref['cohort'].isin(COHORTS)
    if scope == 'ephys':
        keep &= ref['exclude_ephys'].fillna(0) == 0
    if learners is not None:
        keep &= ref['learner'] == bool(learners)
    out = ref[keep].reset_index(drop=True)
    out.attrs = dict(ref.attrs, scope=scope, learners=learners)
    return out


def keep_selected_mice(data_df, scope='behaviour', learners=None, ref=None):
    """Rows of data_df (any table with mouse_id) whose mouse passes select_mice; adds cohort and learner columns."""
    mice = select_mice(scope, learners, ref)[['mouse_id', 'cohort', 'learner']]
    data_df = data_df.drop(columns=[c for c in ('cohort', 'learner') if c in data_df.columns])
    return data_df.merge(mice, on='mouse_id', how='inner')


def load_probe_insertion_info(path=None):
    """
    Probe insertion sheet (one row per insertion), empty spill-over columns ('Colonne*', 'Unnamed*') dropped.
    :param path: optional sheet path (default: canonical share_internal sheet)
    :return: pd.DataFrame with mouse_id (renamed from mouse_name)
    """
    path = Path(path) if path else dataset_info_dir() / PROBE_SHEET
    header = pd.read_excel(path, nrows=0).columns
    cols = [c for c in header if not str(c).startswith(('Colonne', 'Unnamed'))]
    df = pd.read_excel(path, usecols=cols).rename(columns={'mouse_name': 'mouse_id'})
    df = df.dropna(how='all')
    df.attrs['source'] = str(path)
    return df
