#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: brain_wide_analysis
@file: dlc_helpers.py

Single entry point for DeepLabCut (DLC) data stored in the NWB files (NWB_converter, DLC_to_nwb.py and
utils/dlc_utils.compute_kinematics_alt for AB / MH sessions): loading, units, alignment checks, likelihood masking,
cleaning, DLC lick times, top + side combination, event-locked traces, window metrics and movement onsets.

How the NWB stores DLC (AB / MH, compute_kinematics_alt), which this module relies on:
- processing/behavior/BehavioralTimeSeries/<series>: data in pixels (positions, distances; pupil_area in px^2),
  degrees (angles) or likelihood; light smoothing only (dlc2kinematics, window 3), no likelihood masking.
- every series has a `conversion` attribute = mm per pixel of its camera (side ~0.030, top ~0.033), also set on angles
  and likelihoods (where it must not be applied).
- `*_velocity` = frame-to-frame difference x nominal frame rate (per second, signed); whisker_velocity is angular
  (deg/s). The loader converts units only (x mm per pixel): it never multiplies by the frame rate again.
- timestamps = camera pulses on the session clock, one per frame. Both cameras are triggered on the same clock, but
  each camera can miss its own frames (and the saved video can start / end on a different pulse), so the side and top
  timestamp series are never assumed identical: each series is used with its own camera's timestamps.
- tongue_dlc_licks / jaw_dlc_licks (BehavioralEvents) are kept on the video clock in the NWB (side frame index /
  frame rate, 0 = first side frame). get_dlc_lick_times corrects the session offset at loading.

Velocity vs speed (rules used throughout):
- velocity = signed time derivative of a scalar series (distance, angle, area): sign = direction (e.g. jaw opening
  vs closing, whisker protraction vs retraction). Stored *_velocity series are velocities.
- speed = |velocity| for a scalar series (e.g. whisker speed = |whisker_velocity|, deg/s), or the 2-D speed of a
  keypoint, sqrt(dx^2 + dy^2) / dt (mm/s), which is NOT |d distance / dt| (a keypoint moving around its reference
  point has speed but no distance change). derivative() recomputes either from the timestamps (NaN across gaps).
- known converter issue handled here: jaw / nose reference point at the image origin when > 5% of frames are below
  the likelihood cutoff (fix_zero_reference; NWB_ks4 2026-11-03: jaw in 96 of 244 DLC sessions, side nose 3, top
  nose 14; list in combined_results_ks4/ssl-dlc-utils-validation/tables/dlc_reference_at_origin.csv). tongue_distance is the distance from the image origin by design
  (compute_kinematics_alt), not from a reference point.
- per-session exceptions (end-aligned DLC, camera pulse runs, camera timestamps from the log): ibl-ai-agent
  skills/ssl-valid-data/references/nwb_session_exceptions.md.

Typical use:
    views = load_dlc(nwb_file)                       # {'side': DLCView, 'top': DLCView}, physical units, masked
    checks = check_dlc(nwb_file)                     # one row per camera, with flags
    jaw = clean_trace(views['side'].data['jaw_distance'], views['side'].timestamps, max_gap_s=0.05)
    t, m = peth(jaw, views['side'].timestamps, trial_starts, window=(-1, 2))
    ok, verdict = pupil_session_ok(session_id)       # pupil QC verdict; 'fail' sessions excluded from pupil analyses
    pupil, _ = clean_series(views['side'], 'pupil_area')  # pupil QC frame states (blink / squint / failed -> NaN)
    pupil_pct = pupil_relative(pupil)                # % change from the session median
"""

# Imports
import glob
import json
import os
from dataclasses import dataclass, field

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, median_filter

TS_PATH = 'processing/behavior/BehavioralTimeSeries'
EV_PATH = 'processing/behavior/BehavioralEvents'

# Likelihood series that decide whether a frame of a derived series is valid (keypoints used to compute it in
# compute_kinematics_alt). Positions <kp>_x / <kp>_y use <kp>_likelihood (see likelihood_series).
# Jaw: never gated by likelihood (user, 2026-11-03: the jaw is visible and well placed in frames with low jaw
# likelihood, e.g. AB157 / AB158 / MH062 with > 75% of frames < 0.5); occasional wrong predictions are removed with a
# median filter instead (CLEAN_DEFAULTS).
NO_LIKELIHOOD_KEYPOINTS = ('jaw',)
SERIES_LIKELIHOOD = {
    'tongue_angle': ['tongue_likelihood'], 'tongue_distance': ['tongue_likelihood'],
    'tongue_velocity': ['tongue_likelihood'],
    'nose_angle': ['nose_tip_likelihood'], 'nose_distance': ['nose_tip_likelihood'],
    'nose_velocity': ['nose_tip_likelihood'],
    # pupil: each of the four points (the stored pupil_likelihood is already thresholded at 0.5 on all four)
    'pupil_area': ['pupil_top_likelihood', 'pupil_right_likelihood', 'pupil_bottom_likelihood', 'pupil_left_likelihood'],
    'pupil_area_velocity': ['pupil_top_likelihood', 'pupil_right_likelihood', 'pupil_bottom_likelihood',
                            'pupil_left_likelihood'],
    'whisker_angle': ['top_particle_likelihood', 'whisker_base_likelihood'],
    'whisker_velocity': ['top_particle_likelihood', 'whisker_base_likelihood'],
    'top_nose_angle': ['top_nose_tip_likelihood'], 'top_nose_distance': ['top_nose_tip_likelihood'],
    'top_nose_velocity': ['top_nose_tip_likelihood'],
}
DEFAULT_CUTOFF = 0.5  # converter pcutoff for AB / MH (compute_kinematics_alt); kept for the tongue (presence)
# Per likelihood series (user, 2026-11-03: 0.5 is too strict; validation 007: pupil, whisker and nose points are
# well placed from likelihood ~0.1, below it the eye is closed / the points are thrown off or occluded by the spout)
LIKELIHOOD_CUTOFF = {
    'pupil_top_likelihood': 0.1, 'pupil_right_likelihood': 0.1, 'pupil_bottom_likelihood': 0.1,
    'pupil_left_likelihood': 0.1,
    'top_particle_likelihood': 0.1, 'whisker_base_likelihood': 0.1,
    'nose_tip_likelihood': 0.1, 'top_nose_tip_likelihood': 0.1,
}

# Expected DLC frame -> camera pulse offset (frames) in correctly paired sessions (ssl-dlc-alignment-debug 005-007):
# tongue_likelihood peaks a few frames before the piezo lick
SIDE_LICK_OFFSET_OK = (0, 12)


# ---------------------------------
# Units
# ---------------------------------

def series_unit(name):
    """(unit, power of the mm-per-pixel conversion) for a DLC series."""
    if name.endswith('likelihood'):
        return '', 0
    if name == 'pupil_area':
        return 'mm^2', 2
    if name == 'pupil_area_velocity':
        return 'mm^2/s', 2
    if name.endswith('_angle'):
        return 'deg', 0
    if name == 'whisker_velocity':
        return 'deg/s', 0
    if name.endswith('_velocity'):
        return 'mm/s', 1
    if name.endswith(('_distance', '_x', '_y')):
        return 'mm', 1
    return '', 0


def likelihood_series(name):
    """Likelihood series names that gate a DLC series (empty list: no gating)."""
    if name in SERIES_LIKELIHOOD:
        return SERIES_LIKELIHOOD[name]
    for suffix in ('_x', '_y'):
        if name.endswith(suffix) and name[:-len(suffix)] not in NO_LIKELIHOOD_KEYPOINTS:
            return [name[:-len(suffix)] + '_likelihood']
    return []


def cutoff_for(likelihood_name, cutoff=None):
    if cutoff is not None:
        return cutoff
    return LIKELIHOOD_CUTOFF.get(likelihood_name, DEFAULT_CUTOFF)


# ---------------------------------
# Loading
# ---------------------------------

@dataclass
class DLCView:
    """DLC series of one camera. data: frames x series (physical units, NaN where masked); likelihood: frames x
    likelihood series; timestamps: session time of each frame."""
    view: str
    timestamps: np.ndarray
    data: pd.DataFrame
    likelihood: pd.DataFrame
    units: dict
    rate: float
    n_frames_raw: int
    n_timestamps_raw: int
    masked: dict = field(default_factory=dict)  # series -> fraction of frames masked
    session_id: str = ''  # NWB file name; used to find the pupil QC of the session
    fixed_reference: list = field(default_factory=list)  # keypoint groups recomputed by fix_zero_reference

    @property
    def time(self):
        return self.timestamps


def _camera_groups(ts_group):
    """Series grouped by camera: key = (n timestamps, first timestamp); side = group of jaw_angle, top = group of
    whisker_angle."""
    groups = {}
    for name in ts_group:
        g = ts_group[name]
        if 'timestamps' not in g or name == 'lick_trace':
            continue
        ts = g['timestamps']
        if ts.shape[0] == 0:
            continue
        groups.setdefault((ts.shape[0], float(ts[0])), []).append(name)
    views = {}
    for view, marker in (('side', 'jaw_angle'), ('top', 'whisker_angle')):
        for key, names in groups.items():
            if marker in names:
                views[view] = names
    # series of the same camera as a marker but listed under both (identical trains): keep each view's own names
    if 'side' in views and 'top' in views and views['side'] is views['top']:
        names = views['side']
        views['top'] = [n for n in names if n.startswith(('whisker', 'top_'))]
        views['side'] = [n for n in names if n not in views['top']]
    return views


def load_dlc(nwb_file, views=('side', 'top'), series=None, physical_units=True, mask_likelihood=False, cutoff=None,
             fix_reference=True, jaw_reference='nwb'):
    """Load the DLC series of an NWB file per camera.

    :param nwb_file: path to the NWB file.
    :param views: cameras to load ('side', 'top').
    :param series: optional list of series names to keep (likelihood series are always loaded).
    :param physical_units: convert pixels to mm (mm^2 for pupil area) with the NWB conversion factor.
    :param mask_likelihood: NaN where any gating likelihood is below its cutoff (velocities: current or previous
        frame).
    :param cutoff: single cutoff for all likelihoods (default: LIKELIHOOD_CUTOFF / DEFAULT_CUTOFF).
    :param fix_reference: recompute jaw / nose series whose converter reference point fell to the image origin
        (fix_zero_reference; DLCView.fixed_reference lists them).
    :param jaw_reference: 'nwb' (default, user 2026-11-03): jaw_distance as stored in the NWB; 'jaw_ref': distance of
        the jaw keypoint from the median position of the tracked jaw_ref keypoint (jaw_from_jaw_ref, optional).
    :return: {view: DLCView}; a camera without DLC is absent.
    """
    out = {}
    with h5py.File(nwb_file, 'r') as f:
        if TS_PATH not in f:
            return out
        grp = f[TS_PATH]
        cams = _camera_groups(grp)
        for view in views:
            if view not in cams:
                continue
            names = sorted(cams[view])
            ts = grp[names[0]]['timestamps'][:]
            n_data = min(grp[n]['data'].shape[0] for n in names)
            n = min(n_data, len(ts))  # frames paired from the first timestamp (converter convention)
            raw, lik, conv = {}, {}, {}
            for name in names:
                d = grp[name]['data']
                if name.endswith('likelihood'):
                    lik[name] = d[:n].astype(float)
                else:
                    raw[name] = d[:n].astype(float)
                    conv[name] = float(d.attrs.get('conversion', 1.0))
            rate200 = 1 / np.median(np.diff(ts[0:200])) if len(ts) > 1 else np.nan  # as DLC_to_nwb
            fixed = fix_zero_reference(raw, lik, view, rate200) if fix_reference else []
            if view == 'side' and jaw_reference == 'jaw_ref' and jaw_from_jaw_ref(raw, rate200):
                fixed.append('jaw_distance_from_jaw_ref')
            data, units = {}, {}
            for name, x in raw.items():
                if series is not None and name not in series:
                    continue
                unit, power = series_unit(name)
                if physical_units and power:
                    x = x * conv[name] ** power
                else:
                    unit = 'px' if power == 1 else 'px^2' if power == 2 else unit
                data[name] = x
                units[name] = unit
            lik = pd.DataFrame(lik)
            data = pd.DataFrame(data)
            masked = {}
            if mask_likelihood:
                for name in data.columns:
                    gate = [l for l in likelihood_series(name) if l in lik]
                    if not gate:
                        continue
                    bad = np.zeros(n, dtype=bool)
                    for l in gate:
                        bad |= lik[l].to_numpy() < cutoff_for(l, cutoff)
                    if name.endswith('_velocity'):
                        bad[1:] |= bad[:-1]
                    data.loc[bad, name] = np.nan
                    masked[name] = float(bad.mean())
            ts = ts[:n]
            out[view] = DLCView(view=view, timestamps=ts, data=data, likelihood=lik, units=units,
                                rate=frame_rate(ts), n_frames_raw=n_data, n_timestamps_raw=len(grp[names[0]]['timestamps']),
                                masked=masked, fixed_reference=fixed,
                                session_id=os.path.splitext(os.path.basename(str(nwb_file)))[0])
    return out


def jaw_from_jaw_ref(raw, rate):
    """Recompute, in place (pixels), jaw_distance and jaw_velocity relative to the tracked jaw_ref keypoint (user,
    2026-11-03): distance of the jaw keypoint from the median jaw_ref position over all frames (likelihood not used
    for the jaw); velocity = frame difference x rate as in the converter. jaw_angle is left as is.
    :return: True if recomputed (jaw and jaw_ref positions present)."""
    if not all(k in raw for k in ('jaw_x', 'jaw_y', 'jaw_ref_x', 'jaw_ref_y')):
        return False
    rx, ry = np.median(raw['jaw_ref_x']), np.median(raw['jaw_ref_y'])
    dist = np.hypot(raw['jaw_x'] - rx, raw['jaw_y'] - ry)
    raw['jaw_distance'] = dist
    vel = np.zeros_like(dist)
    vel[1:] = np.diff(dist) * np.round(rate, 2)
    raw['jaw_velocity'] = vel
    return True


def fix_zero_reference(raw, lik, view, rate, cutoff=DEFAULT_CUTOFF):
    """Recompute, in place (pixels), the derived series whose reference point fell to the image origin in
    compute_kinematics_alt: the reference there is a percentile (side: 5th) or median (top) of
    np.where(likelihood > cutoff, x, 0), which is 0 as soon as more than 5% (side) / 50% (top) of the frames are below
    the cutoff. Recomputed with the same formulas, the reference taken over the valid frames only. Series of sessions
    below these fractions are left unchanged (identical to the NWB).
    Side: jaw_distance / jaw_angle / jaw_velocity and tongue_angle (jaw reference), nose_distance / nose_angle /
    nose_velocity. Top: top_nose_distance / top_nose_angle / top_nose_velocity.
    :return: list of keypoint groups recomputed ('jaw', 'nose', 'top_nose')."""
    fixed = []
    if view == 'side':
        groups = (('jaw', 'jaw', 'jaw_ref_x', 5, 0.05), ('nose', 'nose_tip', 'nose_base_x', 5, 0.05))  # (name, keypoint, base, percentile, max fraction low)
    else:
        groups = (('top_nose', 'top_nose_tip', 'top_nose_base_y', 50, 0.5),)
    for name, kp, base, pct, max_low in groups:
        need = [f'{kp}_x', f'{kp}_y', base]
        lname = f'{kp}_likelihood'
        if lname not in lik or any(k not in raw for k in need):
            continue
        ok = lik[lname] > cutoff
        if (~ok).mean() <= max_low or ok.sum() < 10:
            continue
        if kp in NO_LIKELIHOOD_KEYPOINTS:
            ok = np.ones(len(ok), dtype=bool)  # jaw: reference over all frames (likelihood not used for the jaw)
        x, y = raw[f'{kp}_x'], raw[f'{kp}_y']
        rx, ry = np.percentile(x[ok], pct), np.percentile(y[ok], pct)
        if view == 'side':
            dy = y - ry
            raw[f'{name}_angle'] = np.degrees(np.arcsin(dy / np.sqrt(dy ** 2 + (x - np.median(raw[base])) ** 2)))
            if name == 'jaw' and 'tongue_y' in raw and 'tongue_x' in raw:
                ty = raw['tongue_y'] - ry
                raw['tongue_angle'] = np.degrees(np.arcsin(ty / np.sqrt(ty ** 2 + (raw['tongue_x'] - np.median(raw[base])) ** 2)))
        else:
            dx = rx - x
            raw[f'{name}_angle'] = np.degrees(np.arcsin(dx / np.sqrt(dx ** 2 + (ry - np.median(raw[base])) ** 2)))
        dist = np.sqrt((y - ry) ** 2 + (x - rx) ** 2)
        raw[f'{name}_distance'] = dist
        vel = np.zeros_like(dist)
        vel[1:] = np.diff(dist) * np.round(rate, 2)
        raw[f'{name}_velocity'] = vel
        fixed.append(name)
    return fixed


def frame_rate(timestamps):
    """Camera frame rate (Hz) from the median inter-frame interval."""
    if len(timestamps) < 2:
        return np.nan
    return float(1 / np.median(np.diff(timestamps)))


# ---------------------------------
# DLC lick times
# ---------------------------------

def get_dlc_lick_times(nwb_file, method='frame'):
    """tongue_dlc_licks / jaw_dlc_licks on the session clock (the NWB keeps them on the video clock: side frame index /
    rate, rate = rounded 1 / median of the first 200 side intervals, as in DLC_to_nwb).

    method='frame' (default): each lick -> its side frame -> that frame's camera timestamp. This is the session offset
        (first side timestamp) plus the true time of each frame, so it also absorbs the camera clock running slightly
        off the nominal rate and any missing pulses.
    method='offset': video time + first side timestamp (nominal rate; for comparison).
    :return: {'tongue': array, 'jaw': array} (missing keys if not in the file).
    """
    out = {}
    with h5py.File(nwb_file, 'r') as f:
        if EV_PATH not in f or TS_PATH not in f or 'jaw_angle' not in f[TS_PATH]:
            return out
        side_ts = f[TS_PATH]['jaw_angle']['timestamps'][:]
        if len(side_ts) < 2:
            return out
        rate = np.round(1 / np.median(np.diff(side_ts[0:200])), 2)
        for key, name in (('tongue', 'tongue_dlc_licks'), ('jaw', 'jaw_dlc_licks')):
            if name not in f[EV_PATH]:
                continue
            t_video = f[EV_PATH][name]['timestamps'][:]
            if method == 'offset':
                out[key] = t_video + side_ts[0]
                continue
            frames = np.rint(t_video * rate).astype(int)
            frames = frames[(frames >= 0) & (frames < len(side_ts))]
            out[key] = side_ts[frames]
    return out


# ---------------------------------
# Derivatives: velocity and speed
# ---------------------------------

def derivative(x, timestamps, kind='velocity', max_gap_frames=1.5):
    """Time derivative of a scalar DLC series from its timestamps (central differences on the true frame times).
    kind='velocity': signed (units / s); kind='speed': |velocity|. NaN where a neighbouring frame is NaN or further
    than max_gap_frames x the median frame interval (missing frames)."""
    x = np.asarray(x, dtype=float)
    t = np.asarray(timestamps, dtype=float)
    dt = np.diff(t)
    med = np.median(dt)
    d = np.diff(x) / dt
    d[dt > max_gap_frames * med] = np.nan
    v = np.full(len(x), np.nan)
    # central difference = mean of the backward and forward differences (one-sided at the ends / next to gaps)
    back, fwd = np.r_[np.nan, d], np.r_[d, np.nan]
    v = np.nanmean(np.vstack([back, fwd]), axis=0) if len(x) > 1 else v
    both_nan = np.isnan(back) & np.isnan(fwd)
    v[both_nan] = np.nan
    return np.abs(v) if kind == 'speed' else v


def keypoint_speed(x, y, timestamps, max_gap_frames=1.5):
    """2-D speed of a keypoint (same units as x, y per s): sqrt(dx^2 + dy^2) / dt between consecutive frames
    (assigned to the later frame; NaN across gaps)."""
    x, y, t = (np.asarray(a, dtype=float) for a in (x, y, timestamps))
    dt = np.diff(t)
    sp = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2) / dt
    sp[dt > max_gap_frames * np.median(dt)] = np.nan
    return np.r_[np.nan, sp]


# ---------------------------------
# Checks
# ---------------------------------

def _event_locked_offset(x, timestamps, events, max_lag=300):
    """Offset L (frames) maximising the mean of x at frame (event frame - L): DLC frame f belongs to camera pulse
    f + L under the stored pairing, plus the physiological lead of the signal."""
    idx = np.searchsorted(timestamps, events)
    idx = idx[(idx > max_lag) & (idx < len(x) - max_lag)]
    if len(idx) < 10:
        return np.nan, np.nan
    lags = np.arange(-max_lag, max_lag + 1)
    xz = np.nan_to_num((x - np.nanmean(x)) / np.nanstd(x))
    m = np.array([xz[idx - L].mean() for L in lags])
    mad = np.median(np.abs(m - np.median(m))) or 1
    k = int(np.argmax(np.abs(m - np.median(m))))
    return int(lags[k]), float((m[k] - np.median(m)) / mad)


def check_dlc(nwb_file, cutoff=None):
    """Per-camera checks of the DLC data of an NWB file. One row per camera with:
    frames vs timestamps (n, difference, s), timestamp regularity (rate, gaps > 1.5 intervals, largest gap, not
    increasing), coverage of the trials (camera start / end vs first trial start / last trial stop), fraction of
    frames masked for the main series, and an alignment check: offset of the side tongue_likelihood around piezo
    licks (expected SIDE_LICK_OFFSET_OK frames) and of the top whisker_angle (reported only).
    Flags (column 'flags'): 'frames!=timestamps' (>= 1 s of frames), 'gaps', 'not_increasing', 'partial_coverage',
    'side_offset' (alignment outside the expected range), 'no_dlc'."""
    rows = []
    views = load_dlc(nwb_file, mask_likelihood=False)
    with h5py.File(nwb_file, 'r') as f:
        licks = f[EV_PATH]['piezo_lick_times']['timestamps'][:] if EV_PATH in f and 'piezo_lick_times' in f[EV_PATH] else np.array([])
        t = f['intervals/trials'] if 'intervals/trials' in f else None
        t_first = float(t['start_time'][0]) if t is not None else np.nan
        t_last = float(t['stop_time'][-1]) if t is not None else np.nan
        session_id = f['general/session_id'][()].decode() if 'general/session_id' in f else ''
    if not views:
        return pd.DataFrame([dict(session_id=session_id, view='', flags='no_dlc')])
    for view, v in views.items():
        ts = v.timestamps
        dt = np.diff(ts)
        med = np.median(dt) if len(dt) else np.nan
        gaps = dt > 1.5 * med
        row = dict(session_id=session_id, view=view, n_frames=v.n_frames_raw, n_timestamps=v.n_timestamps_raw,
                   frames_minus_timestamps=v.n_frames_raw - v.n_timestamps_raw,
                   diff_s=(v.n_frames_raw - v.n_timestamps_raw) / v.rate, rate=v.rate,
                   n_gaps=int(gaps.sum()), max_gap_s=float(dt.max()) if len(dt) else np.nan,
                   not_increasing=int((dt <= 0).sum()), camera_start=float(ts[0]), camera_stop=float(ts[-1]),
                   first_trial_start=t_first, last_trial_stop=t_last)
        main = 'tongue_likelihood' if view == 'side' else 'whisker_angle'
        if view == 'side' and 'tongue_likelihood' in v.likelihood:
            row['lick_offset_frames'], row['lick_offset_z'] = _event_locked_offset(
                v.likelihood['tongue_likelihood'].to_numpy(), ts, licks)
        elif view == 'top' and main in v.data:
            row['lick_offset_frames'], row['lick_offset_z'] = _event_locked_offset(v.data[main].to_numpy(), ts, licks)
        for l in v.likelihood.columns:
            if l in ('jaw_likelihood', 'tongue_likelihood', 'pupil_likelihood', 'top_particle_likelihood',
                     'whisker_base_likelihood', 'top_nose_tip_likelihood', 'nose_tip_likelihood'):
                row[f'frac_below_{l}'] = float((v.likelihood[l] < cutoff_for(l, cutoff)).mean())
        flags = []
        if abs(row['diff_s']) >= 1:
            flags.append('frames!=timestamps')
        if row['n_gaps']:
            flags.append('gaps')
        if row['not_increasing']:
            flags.append('not_increasing')
        if np.isfinite(t_first) and (ts[0] > t_first or ts[-1] < t_last):
            flags.append('partial_coverage')
        if view == 'side' and np.isfinite(row.get('lick_offset_frames', np.nan)):
            lo, hi = SIDE_LICK_OFFSET_OK
            if not lo <= row['lick_offset_frames'] <= hi:
                flags.append('side_offset')
        row['flags'] = ';'.join(flags)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------
# Cleaning
# ---------------------------------

def clean_trace(x, timestamps=None, likelihood=None, cutoff=DEFAULT_CUTOFF, jump_thresh=None, mad_thresh=None,
                max_gap_s=0.05, smooth_sigma_s=None, median_s=None):
    """Clean one DLC trace (frames already on its camera's timestamps).

    Steps (each optional): running median over median_s (odd number of frames; removes isolated wrong predictions,
    e.g. jaw); NaN where likelihood < cutoff; NaN where the frame-to-frame jump > jump_thresh (trace
    units); NaN beyond mad_thresh x MAD from the median; linear interpolation over gaps <= max_gap_s only (longer
    gaps stay NaN; None: no interpolation); Gaussian smoothing (sigma in s) that ignores NaNs.
    :return: cleaned trace, boolean mask of rejected frames.
    """
    x = np.asarray(x, dtype=float).copy()
    if median_s:
        dt0 = np.median(np.diff(timestamps)) if timestamps is not None else 1.0
        k = max(3, int(round(median_s / dt0)) | 1)
        fin = np.isfinite(x)
        xm = median_filter(np.where(fin, x, np.nanmedian(x)), size=k, mode='nearest')
        x = np.where(fin, xm, np.nan)
    bad = ~np.isfinite(x)
    if likelihood is not None:
        bad |= np.asarray(likelihood, dtype=float) < cutoff
    if jump_thresh is not None:
        jump = np.abs(np.diff(x, prepend=x[0]))
        bad |= jump > jump_thresh
    if mad_thresh is not None:
        med = np.nanmedian(x[~bad])
        mad = np.nanmedian(np.abs(x[~bad] - med))
        if mad > 0:
            bad |= np.abs(x - med) > mad_thresh * mad
    x[bad] = np.nan
    t = np.arange(len(x)) if timestamps is None else np.asarray(timestamps, dtype=float)
    if max_gap_s is not None and bad.any() and (~bad).sum() > 1:
        dt = np.median(np.diff(t)) if timestamps is not None else 1.0
        good = np.flatnonzero(~bad)
        filled = np.interp(t, t[good], x[good])
        # gap length of each bad frame: time between the surrounding good frames
        nxt = np.searchsorted(good, np.arange(len(x)))
        prv = nxt - 1
        inside = (prv >= 0) & (nxt < len(good))
        gap = np.full(len(x), np.inf)
        gap[inside] = t[good[nxt[inside]]] - t[good[prv[inside]]] - dt
        fill = bad & (gap <= max_gap_s + 1e-9)
        x[fill] = filled[fill]
    if smooth_sigma_s:
        dt = np.median(np.diff(t)) if timestamps is not None else 1.0
        sigma = smooth_sigma_s / dt
        valid = np.isfinite(x)
        num = gaussian_filter1d(np.where(valid, x, 0.0), sigma, mode='nearest')
        den = gaussian_filter1d(valid.astype(float), sigma, mode='nearest')
        with np.errstate(invalid='ignore', divide='ignore'):
            x = np.where(valid, num / den, np.nan)
    return x, bad


# Default cleaning per series (user decisions 2026-11-03; pupil 2026-11-04): none beyond what the NWB already has,
# except the pupil (running median + smoothing, below). The NWB series come
# from DLC *filtered.h5 (DLC median filter) and dlc2kinematics smoothing (window 3) in the converter, so the default
# returns the stored values: no likelihood mask (no NaN from likelihood thresholding), no extra median (median_s=0.015
# available per call), gap interpolation <= 50 ms only acts on NaN (none without masking).
# A likelihood mask can still be requested per call (cutoff='default': LIKELIHOOD_CUTOFF per keypoint, else
# DEFAULT_CUTOFF). Earlier default (superseded): likelihood mask and linear
# interpolation over gaps <= 50 ms; jaw: no likelihood mask, running median instead. No jump / MAD step (MAD removed
# real whisks), smoothing off.
CLEAN_DEFAULTS = {
    'default': dict(cutoff=None, max_gap_s=0.05, jump_thresh=None, mad_thresh=None, smooth_sigma_s=None,
                    median_s=None),
    'jaw': dict(cutoff=None, max_gap_s=0.05, jump_thresh=None, mad_thresh=None, smooth_sigma_s=None, median_s=None),
    # pupil (user, 2026-11-04): short spikes (blinks: DLC throws the points off the closed eye) removed by a running
    # median over 0.1 s (excursions < ~50 ms disappear, steps stay sharp); no Gaussian (blurs onsets). Without the
    # pupil QC only; with it use clean_pupil (blinks / squints / failed frames NaN first). Does not fix sessions where
    # the eye is closed / occluded for long stretches (validation 012 / 013): pupil QC verdict 'fail'.
    'pupil': dict(cutoff=None, max_gap_s=0.05, jump_thresh=None, mad_thresh=None, smooth_sigma_s=None, median_s=0.1),
}

# Tongue presence: tongue out where tongue_likelihood > this (user, 2026-11-04)
TONGUE_PRESENCE_CUTOFF = 0.6


def clean_defaults(name):
    """Default clean_trace parameters of a series (CLEAN_DEFAULTS)."""
    key = 'jaw' if name.startswith('jaw') else 'pupil' if name.startswith('pupil') else 'default'
    return dict(CLEAN_DEFAULTS[key])


def tongue_presence(view, cutoff=TONGUE_PRESENCE_CUTOFF):
    """Tongue out (1) / in (0) per side frame from tongue_likelihood > cutoff; NaN where the likelihood is missing.
    Averaged over trials it gives the probability of tongue out."""
    lik = view.likelihood['tongue_likelihood'].to_numpy().astype(float)
    return np.where(np.isfinite(lik), (lik > cutoff).astype(float), np.nan)


def clean_series(view, name, pupil_qc=True, **overrides):
    """Clean one series of a DLCView (loaded with mask_likelihood=False) with the default parameters of that series
    (clean_defaults), optionally overridden. pupil_area: with the session's pupil QC when it exists (clean_pupil,
    default, user 2026-11-04); pupil_qc=False or no QC output: running median of the stored trace only (a warning
    is printed). Session-level exclusion is separate: pupil_session_ok. :return: cleaned trace, rejected mask."""
    p = clean_defaults(name)
    p.update(overrides)
    if name == 'pupil_area':
        frames = None
        if pupil_qc:
            frames, _ = load_pupil_qc(view.session_id) if view.session_id else (None, None)
            if frames is None:
                print(f'dlc_helpers: no pupil QC for {view.session_id or "this view"}; pupil_area without QC')
        return clean_pupil(view, frames, median_s=p['median_s'])
    cutoff = p.pop('cutoff')
    gate = [l for l in likelihood_series(name) if l in view.likelihood]
    lik = None
    if cutoff is not None and gate:
        # per gating series: likelihood / its own cutoff >= 1 is valid; combined as the minimum ratio, cutoff 1
        lik = np.min(np.vstack([view.likelihood[l].to_numpy() / cutoff_for(l, None if cutoff == 'default' else cutoff)
                                for l in gate]), 0)
        if name.endswith('_velocity'):
            lik = np.minimum(lik, np.r_[lik[0], lik[:-1]])
    return clean_trace(view.data[name].to_numpy(), view.timestamps, likelihood=lik, cutoff=1.0, **p)


def clean_trace_steps(x, timestamps, likelihood=None, cutoff=DEFAULT_CUTOFF, jump_thresh=None, mad_thresh=None,
                      max_gap_s=0.05, smooth_sigma_s=0.01, median_s=None):
    """The clean_trace pipeline one step at a time, to visualise what each step does.
    :return: dict step name -> (trace after the step, cumulative rejected mask), in order: raw, median, likelihood, jump,
        mad, interpolate, smooth (steps whose parameter is None repeat the previous trace)."""
    steps = {'raw': (np.asarray(x, dtype=float), ~np.isfinite(np.asarray(x, dtype=float)))}
    kw = dict(likelihood=None, jump_thresh=None, mad_thresh=None, max_gap_s=None, smooth_sigma_s=None, median_s=None)
    order = [('median', dict(median_s=median_s)), ('likelihood', dict(likelihood=likelihood, cutoff=cutoff)), ('jump', dict(jump_thresh=jump_thresh)),
             ('mad', dict(mad_thresh=mad_thresh)), ('interpolate', dict(max_gap_s=max_gap_s)),
             ('smooth', dict(smooth_sigma_s=smooth_sigma_s))]
    for name, params in order:
        kw.update(params)
        steps[name] = clean_trace(x, timestamps, **kw)
    return steps


# ---------------------------------
# Pupil: QC frame states (ibl-ai-agent projects/ssl-pupil-qc)
# ---------------------------------
# Per session, the pupil QC writes <results>/<mouse>/<day folder>/pupil_qc/<session>_frames.parquet (one row per side
# frame: t, state 0 valid / 1 blink / 2 squint / 3 failed, interpolated = failed gap <= 0.1 s between valid frames,
# linearly interpolated) and <session>_metrics.json (verdict pass / warn / fail on % valid frames and % valid trials).

PUPIL_QC_ROOTS = ('/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4', r'M:\analysis\Axel_Bisi\combined_results_ks4',
                  '//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Axel_Bisi/combined_results_ks4')
PUPIL_STATE = {'valid': 0, 'blink': 1, 'squint': 2, 'failed': 3}
# Sessions where one pupil point is systematically off the eye with high likelihood: pupil_area is recomputed with that
# point replaced by the mirror image of the opposite point across the axis of the other two (user, 2026-11-04;
# MH029 0502: right point off in 64% of frames; 3-point vs 4-point area on a clean session r = 0.988, -4%). Same list
# in ibl-ai-agent projects/ssl-pupil-qc/pupil_qc.THREE_POINT (the QC states use the same area).
PUPIL_THREE_POINT = {'MH029_20250502_110353': 'right'}
_PUPIL_OPPOSITE = {'right': ('left', 'top', 'bottom'), 'left': ('right', 'top', 'bottom'),
                   'top': ('bottom', 'left', 'right'), 'bottom': ('top', 'left', 'right')}


def pupil_area(view):
    """pupil_area of the side DLCView (units of the view), with the PUPIL_THREE_POINT correction for listed sessions:
    the listed point is mirrored from its opposite point and the polygon area top -> right -> bottom -> left recomputed
    (shoelace, as the converter)."""
    point = PUPIL_THREE_POINT.get(view.session_id)
    if point is None:
        return view.data['pupil_area'].to_numpy().astype(float).copy()
    d = {f'{q}_{c}': view.data[f'pupil_{q}_{c}'].to_numpy().astype(float) for q in ('top', 'right', 'bottom', 'left')
         for c in ('x', 'y')}
    opp, a, b = _PUPIL_OPPOSITE[point]
    dx, dy = d[f'{b}_x'] - d[f'{a}_x'], d[f'{b}_y'] - d[f'{a}_y']
    nrm = np.hypot(dx, dy)
    nrm[nrm == 0] = np.nan
    ux, uy = dx / nrm, dy / nrm
    proj = (d[f'{opp}_x'] - d[f'{a}_x']) * ux + (d[f'{opp}_y'] - d[f'{a}_y']) * uy
    px, py = d[f'{a}_x'] + proj * ux, d[f'{a}_y'] + proj * uy
    d[f'{point}_x'], d[f'{point}_y'] = 2 * px - d[f'{opp}_x'], 2 * py - d[f'{opp}_y']
    x = np.column_stack([d[f'{q}_x'] for q in ('top', 'right', 'bottom', 'left')])
    y = np.column_stack([d[f'{q}_y'] for q in ('top', 'right', 'bottom', 'left')])
    return 0.5 * np.abs(np.sum(x * np.roll(y, -1, 1) - y * np.roll(x, -1, 1), 1))
PUPIL_MEDIAN_S = 0.1  # running median after the QC mask (user, 2026-11-04)


def load_pupil_qc(session_id, roots=PUPIL_QC_ROOTS):
    """Pupil QC of a session: (frames DataFrame, metrics dict), or (None, None) if the session has no QC output."""
    mouse = session_id.split('_')[0]
    for root in roots:
        hits = glob.glob(os.path.join(root, mouse, '*', 'pupil_qc', f'{session_id}_frames.parquet'))
        if hits:
            frames = pd.read_parquet(hits[0])
            mpath = hits[0].replace('_frames.parquet', '_metrics.json')
            metrics = json.load(open(mpath)) if os.path.exists(mpath) else {}
            return frames, metrics
    return None, None


def pupil_qc_valid(view, frames):
    """Per frame of the side DLCView: True where the pupil QC calls the frame valid or interpolated (matched on the
    frame timestamps; frames missing from the QC are not valid)."""
    t = frames['t'].to_numpy()
    ok = ((frames['state'].to_numpy() == PUPIL_STATE['valid']) | frames['interpolated'].to_numpy())
    i = np.clip(np.searchsorted(t, view.timestamps), 0, len(t) - 1)
    j = np.where((i > 0) & (np.abs(t[i - 1] - view.timestamps) < np.abs(t[i] - view.timestamps)), i - 1, i)
    match = np.abs(t[j] - view.timestamps) < 0.25 * np.median(np.diff(view.timestamps))
    return match & ok[j]


def clean_pupil(view, frames=None, median_s=PUPIL_MEDIAN_S):
    """Pupil area of the side DLCView (mm^2 if loaded in physical units) cleaned with the pupil QC frame states:
    NaN on blink, squint and failed frames, failed gaps <= 0.1 s between valid frames linearly interpolated (as in the
    QC), then a NaN-aware running median over median_s (a window needs more than half of its frames valid).
    Without QC frames: running median of the stored trace only (clean_series default).
    :return: cleaned trace, boolean mask of frames not used (QC not valid)."""
    x = pupil_area(view)
    if frames is None:
        return clean_trace(x, view.timestamps, median_s=median_s, **{k: v for k, v in clean_defaults('pupil_area').items()
                                                                     if k not in ('cutoff', 'median_s')})
    valid = pupil_qc_valid(view, frames)
    t, f_interp = frames['t'].to_numpy(), frames['interpolated'].to_numpy()
    k = np.clip(np.searchsorted(t, view.timestamps), 0, len(t) - 1)
    interp = valid & f_interp[k] & (np.abs(t[k] - view.timestamps) < 0.25 * np.median(np.diff(view.timestamps)))
    x[~valid | interp] = np.nan
    keep = valid & ~interp
    if interp.any() and keep.sum() > 1:
        idx = np.arange(len(x))
        x[interp] = np.interp(idx[interp], idx[keep], x[keep])
    if median_s:
        w = max(3, int(round(median_s / np.median(np.diff(view.timestamps)))) | 1)
        x = np.array(pd.Series(x).rolling(w, center=True, min_periods=w // 2 + 1).median(), dtype=float)
        x[~valid] = np.nan
    return x, ~valid


def pupil_session_ok(session_id, allowed=('pass', 'warn')):
    """(usable, verdict) of a session for pupil analyses from its pupil QC verdict (user, 2026-11-04: 'fail'
    sessions excluded; sessions without QC output are not usable until the QC is run)."""
    _, metrics = load_pupil_qc(session_id)
    verdict = metrics.get('verdict', 'no QC') if metrics is not None else 'no QC'
    return verdict in allowed, verdict


def pupil_relative(x, kind='percent'):
    """Pupil area relative to the session: 'percent' = 100 x (x / median - 1) over valid frames (removes pupil size
    differences between mice, cameras and setups), 'z' = (x - mean) / SD."""
    if kind == 'percent':
        return 100 * (x / np.nanmedian(x) - 1)
    return (x - np.nanmean(x)) / np.nanstd(x)


# ---------------------------------
# Combining cameras
# ---------------------------------

def to_timestamps(x, ts_from, ts_to, max_gap_s=0.02):
    """Values of a trace (on ts_from) at times ts_to by linear interpolation; NaN outside the recording, where the
    nearest source frame is further than max_gap_s, or next to NaN source frames."""
    x = np.asarray(x, dtype=float)
    good = np.isfinite(x)
    y = np.interp(ts_to, ts_from[good], x[good], left=np.nan, right=np.nan)
    i = np.clip(np.searchsorted(ts_from, ts_to), 1, len(ts_from) - 1)
    near = np.minimum(np.abs(ts_to - ts_from[i - 1]), np.abs(ts_from[i] - ts_to))
    y[near > max_gap_s] = np.nan
    y[~(good[i - 1] & good[i])] = np.nan
    return y


def nose_norm(views, kind='distance'):
    """Nose displacement combining the side (nose_<kind>) and top (top_nose_<kind>) cameras: the top series is
    interpolated onto the side timestamps, then sqrt(side^2 + top^2). :return: (values, side timestamps)."""
    side, top = views['side'], views['top']
    s = side.data[f'nose_{kind}'].to_numpy()
    tp = to_timestamps(top.data[f'top_nose_{kind}'].to_numpy(), top.timestamps, side.timestamps)
    return np.sqrt(s ** 2 + tp ** 2), side.timestamps


# ---------------------------------
# Event-locked traces and metrics
# ---------------------------------

def peth(x, timestamps, events, window=(-1.0, 2.0), dt=None):
    """Event-locked trace matrix (events x time bins): value of the nearest frame within half a frame interval,
    NaN outside the recording or across gaps. dt: bin size (default: frame interval).
    :return: (bin times relative to the event, matrix)."""
    x = np.asarray(x, dtype=float)
    ts = np.asarray(timestamps, dtype=float)
    frame_dt = np.median(np.diff(ts))
    dt = frame_dt if dt is None else dt
    grid = np.arange(np.round(window[0] / dt), np.round(window[1] / dt) + 1) * dt
    q = np.asarray(events, dtype=float)[:, None] + grid[None, :]
    i = np.clip(np.searchsorted(ts, q), 1, len(ts) - 1)
    j = np.where(np.abs(q - ts[i - 1]) <= np.abs(ts[i] - q), i - 1, i)
    out = x[j]
    out[np.abs(ts[j] - q) > frame_dt / 2 + 1e-6] = np.nan
    return grid, out


def window_metric(x, timestamps, events, window, kind='posture', per_second=False):
    """One value per event over frames with event + window[0] <= t < event + window[1].
    kind='posture': mean of the trace; kind='movement': mean |frame-to-frame difference| (x frame rate if
    per_second; differences across masked frames are NaN and ignored)."""
    x = np.asarray(x, dtype=float)
    ts = np.asarray(timestamps, dtype=float)
    lo = np.searchsorted(ts, np.asarray(events) + window[0])
    hi = np.searchsorted(ts, np.asarray(events) + window[1])
    rate = frame_rate(ts)
    out = np.full(len(lo), np.nan)
    for k, (a, b) in enumerate(zip(lo, hi)):
        seg = x[a:b]
        if kind == 'posture':
            v = seg[np.isfinite(seg)]
            out[k] = v.mean() if len(v) else np.nan
        elif kind == 'movement':
            d = np.abs(np.diff(seg))
            d = d[np.isfinite(d)]
            out[k] = (d.mean() * (rate if per_second else 1)) if len(d) else np.nan
        else:
            raise ValueError(f'unknown kind {kind}')
    return out


def movement_onset_times(x, timestamps, trials, time_start=-1.0, time_stop=2.0, baseline_window=(-1.0, -0.1),
                         thresh_factor=5.0, pooled_baseline_std=True, pre_onset_window=0.5, onset_peak_fraction=0.3):
    """Movement onset per trial (s from start_time), as behavior_analysis_utils.find_dlc_metric_onset_times
    (defaults: beh_plotting_functions DLC onset parameters), with frames located by timestamp.

    Per trial: baseline-subtracted trace (baseline mean over baseline_window), threshold = thresh_factor x baseline
    SD (median SD over trials if pooled_baseline_std). Search from start_time - pre_onset_window to the end of the
    response window (time_stop + artifact window), stopping at the piezo lick; onset = first threshold-crossing bout
    whose first local peak is after start_time, walked back to onset_peak_fraction x peak; fallback = peak time.
    :param trials: DataFrame with start_time, response_window_start_time, lick_time.
    :return: DataFrame (trial index) with onset, threshold, and the trial traces (time_start .. time_stop + artifact).
    """
    rate = frame_rate(timestamps)
    dt = 1 / rate
    artifact = np.round(np.nanmean(trials['response_window_start_time'].to_numpy() - trials['start_time'].to_numpy()), 1)
    grid, tr = peth(x, timestamps, trials['start_time'].to_numpy(), (time_start, time_stop + artifact), dt=dt)
    _, bas = peth(x, timestamps, trials['start_time'].to_numpy(), baseline_window, dt=dt)
    bas_mean, bas_std = np.nanmean(bas, 1), np.nanstd(bas, 1)
    tr = tr - bas_mean[:, None]
    thr = thresh_factor * (np.nanmedian(bas_std) if pooled_baseline_std else bas_std) * np.ones(len(tr))
    thr[np.all(np.isnan(bas), 1)] = np.nan
    t0 = int(np.argmin(np.abs(grid)))
    onsets = np.full(len(tr), np.nan)
    for k, (trace, row) in enumerate(zip(tr, trials.itertuples())):
        if np.all(np.isnan(trace)):
            continue
        rw = row.response_window_start_time - row.start_time
        stop = min(len(trace), t0 + int(round((time_stop + rw) / dt)))
        lick = row.lick_time - row.start_time
        if np.isfinite(lick):
            stop = min(stop, t0 + int(round(lick / dt)))
        start = max(0, t0 - int(round(pre_onset_window / dt)))
        w = np.nan_to_num(trace, nan=-np.inf)
        onset = None
        if np.isfinite(thr[k]):
            cross = np.flatnonzero(w[start:stop] > thr[k]) + start
            i = 0
            while i < len(cross):
                c = cross[i]
                end = c
                while end < stop and w[end] > thr[k]:
                    end += 1
                p = c
                while p + 1 < end and w[p + 1] >= w[p]:
                    p += 1
                if p >= t0:
                    b = p
                    while b > c and w[b] > onset_peak_fraction * w[p]:
                        b -= 1
                    onset = b
                    break
                i = np.searchsorted(cross, end)
        if onset is None:
            seg = trace[t0:stop] if stop > t0 else trace[start:]
            if np.all(np.isnan(seg)):
                continue
            onset = (t0 if stop > t0 else start) + int(np.nanargmax(seg))
        onsets[k] = grid[onset]
    return pd.DataFrame({'onset': onsets, 'threshold': thr, 'trace': list(tr)}, index=trials.index)


# ---------------------------------
# Frame selections (ITI, quiet windows, epochs)
# ---------------------------------

def time_mask(timestamps, starts, stops):
    """Boolean mask of frames with start <= t < stop for any (start, stop) window (frames located by timestamp)."""
    ts = np.asarray(timestamps, dtype=float)
    mask = np.zeros(len(ts), dtype=bool)
    lo = np.searchsorted(ts, np.asarray(starts, dtype=float), side='left')
    hi = np.searchsorted(ts, np.asarray(stops, dtype=float), side='left')
    for a, b in zip(lo, hi):
        mask[a:b] = True
    return mask


def iti_mask(timestamps, trials=None, stim_types=('whisker_trial', 'auditory_trial'), trial_duration=5.0,
             epochs=None):
    """Inter-trial-interval frames: inside the optional epochs ((start, stop) pairs, e.g. one context), excluding
    [start_time, start_time + trial_duration) of trials whose trial_type is in stim_types (as
    behavior_analysis_utils.get_iti_frames, on the camera's own timestamps)."""
    ts = np.asarray(timestamps, dtype=float)
    mask = np.ones(len(ts), dtype=bool)
    if epochs is not None:
        ep = np.asarray(epochs, dtype=float).reshape(-1, 2)
        mask &= time_mask(ts, ep[:, 0], ep[:, 1])
    if trials is not None and stim_types:
        st = trials.loc[trials.trial_type.isin(list(stim_types)), 'start_time'].to_numpy()
        mask &= ~time_mask(ts, st, st + trial_duration)
    return mask


def quiet_window_mask(timestamps, trials, mode='abort_window'):
    """Pre-stimulus quiet-window frames per trial: from abort_window_start_time (mode='abort_window') or a fixed
    duration before start_time (mode=seconds, e.g. 2.0 or 0.5) to start_time.
    :return: list of frame index arrays, one per trial (as get_all_quiet_windows)."""
    ts = np.asarray(timestamps, dtype=float)
    st = trials['start_time'].to_numpy()
    begin = trials['abort_window_start_time'].to_numpy() if mode == 'abort_window' else st - float(mode)
    lo, hi = np.searchsorted(ts, begin), np.searchsorted(ts, st)
    return [np.arange(a, b) for a, b in zip(lo, hi)]
