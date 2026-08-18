#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: brainrender
@file: generate_probe_tracks_visualization_combined.py
@time: 3/3/2025 12:53 PM
"""
import os
import logging
import hashlib
import random
from pathlib import Path
import numpy as np
import pandas as pd
import tempfile
import shutil
import brainrender
from brainrender import Scene
from brainrender.actors import Points
from brainrender.video import VideoMaker

logging.basicConfig(level=logging.WARNING, force=True)
log = logging.getLogger(__name__)

brainrender.settings.SHOW_AXES = False
brainrender.settings.ROOT_ALPHA = 0.1
brainrender.settings.ROOT_COLOR = [0.99, 0.99, 0.99]
brainrender.settings.SHADER_STYLE = 'cartoon'
brainrender.settings.LW = 0.1

TARGET_AREA_CUSTOM_CMAP = {
    'wS1': '#379443',
    'wS2': '#51db64',
    'A1': '#334b82',
    'PPC': '#5a81db',
    'DLS': '#8232ba',
    'wM1': '#fab143',
    'wM2': '#ed753e',
    'ALM': '#a31010',
    'tjM1': '#e3320e',
    'OFC': '#54110c',
    'SC': '#8a6c27'
}

EXPERIMENTER_CMAP = {
    'AB': '#127ac4',
    'MH': '#6915bd'
}


def find_session_folder(base: Path, m_name: str, date_val) -> Path | None:
    """
    Find the session folder for a given mouse and recording date.
    date_val can be:
      - a string 'DD.MM.YYYY'  (as stored in the Excel probe insertion table)
      - a pandas Timestamp / datetime object  (if pandas parsed the column)
    Session folders are named  m_name_YYYYMMDD_hhmmss  and must contain
    an 'Ephys' subfolder to be considered valid.
    Returns the matched session Path, or None if not found.
    """
    mouse_dir = base / m_name
    if not mouse_dir.exists():
        log.warning("  Mouse directory not found: %s", mouse_dir)
        return None

    # Normalise date to YYYYMMDD string regardless of input type
    if hasattr(date_val, 'strftime'):
        date_formatted = date_val.strftime('%Y%m%d')
    else:
        day, month, year = str(date_val).split('.')
        date_formatted = f"{year}{month}{day}"

    prefix = f"{m_name}_{date_formatted}"
    candidates = [
        f for f in sorted(mouse_dir.iterdir())
        if f.is_dir() and f.name.startswith(prefix) and (f / 'Ephys').exists()
    ]

    print(f"  Date {date_val} → prefix '{prefix}' → {len(candidates)} match(es): {[f.name for f in candidates]}")

    if not candidates:
        log.warning("  No session folder found for %s on %s.", m_name, date_val)
        return None
    if len(candidates) > 1:
        log.warning("  Multiple sessions found for %s on %s — using first: %s", m_name, date_val, candidates[0].name)

    return candidates[0]


def generate_brain_animation(scene, output_folder: Path, fig_stem: str, camera_view: str, params: dict, camera: dict) -> None:
    from brainrender.camera import set_camera

    anim_params = params.get('animation', {})
    fps        = anim_params.get('fps', 30)
    duration   = anim_params.get('duration', 8)
    video_zoom = anim_params.get('zoom', 0.8)
    n_frames   = fps * duration
    az_step    = 360.0 / n_frames
    el_step    = -0.1 if camera_view == 'top' else 0

    def make_frame(scene, frame_number, *args, **kwargs):
        if frame_number == 0:
            set_camera(scene, camera)
            scene.plotter.camera.Zoom(video_zoom)
        if camera_view in ['top','angled']:
            scene.plotter.camera.Roll(az_step)
        else:
            scene.plotter.camera.Azimuth(az_step)
        #if el_step:
        #    scene.plotter.camera.Elevation(el_step)

    video_name = f"{fig_stem}_animation"
    print(f"  Rendering animation ({camera_view}): {n_frames} frames @ {fps} fps")

    tmp_dir = tempfile.mkdtemp()
    try:
        vm = VideoMaker(scene, tmp_dir, video_name, make_frame_func=make_frame)
        vm.make_video(duration=duration, fps=fps)

        saved_files = list(Path(tmp_dir).iterdir())
        print(f"  Files in tmp_dir: {saved_files}")

        src  = Path(tmp_dir) / f"{video_name}.mp4"
        dest = output_folder / f"{video_name}.mp4"
        shutil.copy2(src, dest)
        print(f"  Animation saved → {dest}")
    except Exception as e:
        print(f"  Animation error: {e}")
    finally:
        import time
        time.sleep(1)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return


def generate_brain_visualization(params):
    """Plot probe tracks with brainrender using the given parameters."""

    color_by        = params['color_by']
    transparent     = params['transparent']
    dark_background = params['dark_background']
    days            = params['day_of_recording']
    assert isinstance(days, list)

    print(f"color_by={color_by}  camera={params['camera_view']}  format={params['file_format']}  days={days}")

    input_data_path_axel   = Path(params['input_data_path_axel'])
    input_data_path_myriam = Path(params['input_data_path_myriam'])
    output_folder_path     = Path(params['output_folder_path'])

    brainrender.settings.SCREENSHOT_TRANSPARENT_BACKGROUND = transparent
    probe_color = "#ebf7ff" if dark_background else "#262626"
    if dark_background:
        brainrender.settings.BACKGROUND_COLOR = [0, 0, 0]

    camera_view = params['camera_view']
    camera      = params['camera_options'][camera_view]
    zoom        = {'frontal': 1.0, 'sagittal': 0.5, 'top': 1.7, 'angled': 1.3}[camera_view]
    cam_suffix  = camera_view

    areas_to_show = params['areas_to_show']
    str_suffix    = '_'.join(areas_to_show) if areas_to_show and areas_to_show != [''] else 'empty'
    overlay_areas = params['overlay_areas'] if areas_to_show and areas_to_show != [''] else False
    label_areas   = params['label_areas']

    # Output subfolder by color scheme
    subdir = {'reward_group': 'reward_group', 'target_area': 'target_area',
              'experimenter': 'experimenter', 'none': 'tracks'}.get(color_by, 'tracks')
    output_folder_path = output_folder_path / subdir
    output_folder_path.mkdir(parents=True, exist_ok=True)

    scene = Scene(inset=False, title="", screenshots_folder=output_folder_path, title_color='darkgrey',
                  atlas_name='allen_mouse_bluebrain_barrels_10um')

    if overlay_areas:
        for area in areas_to_show:
            actor = scene.add_brain_region(area, alpha=0.1, hemisphere='left', silhouette=True)
            if label_areas:
                scene.add_label(actor, area, size=500, color=None, radius=100, xoffset=0, yoffset=-500, zoffset=100)

    # Probe and mouse metadata
    probe_info_df = pd.read_excel(params['probe_info_path'])
    probe_info_df = probe_info_df[probe_info_df['valid']==1]
    probe_info_df['day_of_recording'] = probe_info_df['day_of_recording'].astype(int)
    probe_info_df = probe_info_df[probe_info_df['day_of_recording'].isin(days)]
    print(f"Probe entries after day filter {days}: {len(probe_info_df)}")

    if color_by == 'target_area':
        target_areas = [a for a in probe_info_df['target_area'].unique() if isinstance(a, str)]
        if params['target_cmap'] == 'auto':
            random.seed(42)
            colors = ['#' + hashlib.md5(f'42-{i}'.encode()).hexdigest()[:6] for i in range(len(target_areas))]
            target_areas_color_dict = dict(zip(target_areas, colors))
        else:
            target_areas_color_dict = TARGET_AREA_CUSTOM_CMAP

    mouse_info_df = pd.read_excel(params['mouse_info_path'])
    mouse_info_df.rename(columns={'mouse_name': 'mouse_id'}, inplace=True)
    mouse_info_df = mouse_info_df[mouse_info_df['exclude'] == 0]
    mouse_info_df = mouse_info_df[mouse_info_df.learning_category.isin(params['learning_category'])]

    mouse_list     = probe_info_df['mouse_name'].unique()
    mouse_list_sub = [m for m in mouse_list if m in mouse_info_df['mouse_id'].unique()]

    probe_arrays_total     = []
    mouse_list_sub_in_plot = []


    for m_name in mouse_list_sub:
        print(f"── Mouse: {m_name}")
        experimenter = m_name[:2]
        base = input_data_path_axel if m_name.startswith('AB') else input_data_path_myriam

        mouse_probes = probe_info_df[probe_info_df['mouse_name'] == m_name]

        if color_by == 'reward_group':
            rg = mouse_probes['reward_group'].values[0]
            probe_color = 'forestgreen' if rg == 'R+' else 'blueviolet'

        for date_val, date_probes in mouse_probes.groupby('date'):

            session_folder = find_session_folder(base, m_name, date_val)
            if session_folder is None:
                continue

            # Resolve atlas-space tracks path from session folder
            if m_name.startswith('AB') and int(m_name[2:]) < 80:
                data_folder = session_folder / 'brainreg' / 'manual_segmentation' / 'standard_space' / 'tracks'
            else:
                data_folder = session_folder / 'Anatomy' / 'fused' / 'registered' / 'segmentation' / 'atlas_space' / 'tracks'

            if not data_folder.exists():
                log.warning("  Track folder not found: %s — skipping.", data_folder)
                continue

            # Check if directory is empty
            if data_folder.exists() and not any(data_folder.iterdir()):
                log.warning("  Track folder is empty: %s — skipping.", data_folder)
                continue

            probe_arrays = sorted(f for f in os.listdir(data_folder) if f.endswith('.npy') and f.startswith('imec'))
            probe_tables = sorted(f for f in os.listdir(data_folder) if f.endswith('.csv') and f.startswith('imec'))
            print(f"  {len(probe_arrays)} probe file(s) found in {data_folder}")
            n_valid_probes_for_session = probe_info_df[(probe_info_df['mouse_name'] == m_name) & (probe_info_df['date'] == date_val) & (probe_info_df['valid'] == 1)].shape[0]
            if len(probe_arrays) != n_valid_probes_for_session:
                # Warn for mismatch
                print(f"  WARNING: {n_valid_probes_for_session} valid probe(s) expected from metadata, but {len(probe_arrays)} .npy files found.")


            get_probe_id = lambda fname: fname.split('_')[0][-1]  # e.g. "imec0_mapped.npy" → "0"

            for probe_array, probe_table in zip(probe_arrays, probe_tables):
                probe_name = probe_array.split('.')[0]
                probe_id   = get_probe_id(probe_name)

                # Match on mouse_name + date + probe_id
                probe_info = date_probes[date_probes['probe_id'] == int(probe_id)]
                if probe_info.empty or not probe_info['valid'].values[0]:
                    print(f"  Skipping {m_name} probe {probe_id} on {date_val} (missing or invalid).")
                    continue

                probe_path       = data_folder / probe_array
                probe_table_path = data_folder / probe_table
                probe_arrays_total.append(str(probe_path))
                print(f"  Loading: {probe_path}")

                #if m_name.startswith('AB') and int(m_name[2:]) < 100 and 'mapped' not in probe_name:
                #    print(f"  Skipping {probe_name} — missing hemisphere correction.")
                #    continue

                rows_selected = None
                if params['filter_shanks']:
                    probe_track_table = pd.read_csv(probe_table_path)
                    probe_track_table.rename(columns={
                        'Region acronym': 'ccf_acronym',
                        'Distance from first position [um]': 'distance_from_surface',
                        'Position': 'distance_from_surface',
                    }, inplace=True)

                    if areas_to_show != ['']:
                        matched = [a for a in probe_track_table['ccf_acronym'].unique() if a in areas_to_show]
                        if not matched:
                            print(f"  No area overlap with areas_to_show — skipping shank.")
                            continue
                        rows_selected = probe_track_table[probe_track_table['ccf_acronym'].isin(matched)].index

                # Load and interpolate track
                probe_track = np.load(probe_path, allow_pickle=True)
                probe_track = np.linspace(probe_track[0], probe_track[-1], num=len(probe_track))
                print(f"  Track shape: {probe_track.shape}")

                probe_target = probe_info['target_area'].values[0]
                if probe_target == 'SC' and probe_info['depth'].values[0] > 4000:
                    probe_track = probe_track[-200:]
                    print(f"  SC deep probe — trimmed to last 200 rows.")

                if rows_selected is not None and len(rows_selected) > 0:
                    probe_track = probe_track[np.array(rows_selected), :]
                    print(f"  After shank filter: {len(probe_track)} rows.")

                if color_by == 'target_area':
                    probe_color = target_areas_color_dict.get(probe_target, '#262626')
                elif color_by == 'experimenter':
                    probe_color = EXPERIMENTER_CMAP.get(experimenter, '#262626')

                print(f"  Adding probe {probe_name} (color={probe_color})")
                scene.add(Points(probe_track, name=probe_name, colors=probe_color, radius=18, alpha=1.0, res=15))
                mouse_list_sub_in_plot.append(m_name)

    scene.render(interactive=False, camera=camera, zoom=zoom)

    # Build figure name stem (shared by screenshot and animation)
    parts = [str_suffix]
    if color_by != 'none':
        parts.append(color_by)
    if params['filter_shanks']:
        parts.append('filtered')
    if dark_background:
        parts.append('dark')
    if days is not None:
        parts.append(f'day{days}')
    parts.append(cam_suffix)
    if transparent:
        parts.append('transparent')
    if 'good' in params['learning_category'] and 'moderate' in params['learning_category']:
        parts.append('learners')
    fig_stem = "all_probes_in_atlas_{}".format('_'.join(parts))
    fig_name = f"{fig_stem}.{params['file_format']}"

    n_mice = len(set(mouse_list_sub_in_plot))
    print(f"Saving {len(probe_arrays_total)} probes from {n_mice} mice → {output_folder_path / fig_name}")
    scene.screenshot(name=fig_name, scale=params['scale'])

    if params.get('animate', False):
        generate_brain_animation(scene, output_folder_path, fig_stem, camera_view, params, camera)

    scene.close()


# Define the dictionary with parameters
params = {
    'input_data_path_axel': r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Axel_Bisi\data',
    'input_data_path_myriam': r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Myriam_Hamon\data',
    'probe_info_path': r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\share_internal\Axel_Bisi_Share\dataset_info\joint_probe_insertion_info.xlsx',
    'mouse_info_path': r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\share_internal\Axel_Bisi_Share\dataset_info\joint_mouse_reference_weight.xlsx',
    'output_folder_path': r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Axel_Bisi\combined_results\anatomy',
    'color_by': 'reward_group',  # 'reward_group', 'target_area', 'none'
    'target_cmap': 'custom_cmap',  # 'auto' or 'custom_cmap'
    'transparent': True,
    'dark_background': False,
    'camera_options': {
        "sagittal": {
            "pos": (5053.37, 2365.62, -19213.1),
            "focal_point": (6587.84, 3849.09, -5688.16),
            "viewup": (-0.0169435, -0.993686, 0.110913),
            "roll": 180,
            "distance": 13692.3,
            "clipping_range": (1055.19, 29632.9),
        },
        "frontal": {
            "pos": (-13024.1, -304.244, -5737.87),
            "focal_point": (6587.84, 3849.09, -5688.16),
            "viewup": (0, -1.00000, 0),
            "roll": 180.000,
            "distance": 20047.0,
            "clipping_range": (5431.04, 38493.8),
        },
        "top": {
            "pos": (6504.45, -39122.2, -5985.28),
            "focal_point": (6587.84, 3849.09, -5688.16),
            "viewup": (-1.00000, 0, 0),
            "roll": 74.3241,
            "distance": 42972.4,
            "clipping_range": (35052.7, 53045.5),
        },
        "angled": {
            "pos": (13425.5, -31640.2, -29477.5),
            "focal_point": (6345.28, 3646.06, -7080.42),
            "viewup": (-0.562399, -0.520597, 0.642406),
            "roll": 159.964,
            "distance": 42389.7,
            "clipping_range": (29001.6, 61298.6),
        },
    },
    'camera_view': 'angled',  # 'sagittal', 'frontal', 'top', 'angled'
    'areas_to_show': [''],  # e.g. ['wS1', 'wM1'] or [''] for none
    'filter_shanks': True,
    'overlay_areas': False,
    'label_areas': False,
    'scale': 3,
    'file_format': 'png',  # 'png', 'svg', 'pdf', 'eps'
    'learning_category': ['moderate','good'],
    'day_of_recording': [0],  # list of day_of_recording values, or [0,1,...] for multiple
    'animate': False,          # set True to also render a rotating video for each sweep iteration
    'animation': {
        'fps':      30,        # frames per second
        'duration': 8,         # seconds for a full 360° rotation (longer = slower)
        'zoom': 0.2,  # adjust per view if needed

    },
}

if __name__ == "__main__":
    color_by_sweep = ['reward_group', 'target_area', 'none']
    camera_views   = ['top', 'angled', 'frontal', 'sagittal']
    file_formats   = ['png', 'svg']
    for color in color_by_sweep:
        params['color_by'] = color
        for camera_view in camera_views:
            params['camera_view'] = camera_view
            for file_format in file_formats:
                params['file_format'] = file_format
                generate_brain_visualization(params)
