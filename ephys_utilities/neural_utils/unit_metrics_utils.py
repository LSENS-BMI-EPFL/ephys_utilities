#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: brain_wide_analysis
@file: unit_metrics_utils.py
"""
# Imports
import numpy as np
import pandas as pd
import warnings

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