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
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
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

DEFAULT_AREA_COLOR = '#888888'


def get_output_session_folder(params: dict, m_name: str, session_folder: Path) -> Path:
    """
    Build the session folder to save figures into, always under the fixed
    analysis/<analyzer>/data/<mouse>/<session> tree determined by params['analyzer']
    (e.g. 'Axel_Bisi'), regardless of which mouse/experimenter the session belongs
    to or which raw-data path it was read from. The <analyzer>/data root is derived
    from input_data_path_axel's own analysis/<name>/data pattern, so it stays
    consistent with wherever that share is mounted. Any missing folders are created
    by the caller via mkdir.
    """
    analysis_root = Path(params['input_data_path_axel']).parent.parent  # .../analysis
    analyzer_data_root = analysis_root / params['analyzer'] / 'data'
    return analyzer_data_root / m_name / session_folder.name


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


def _hex_from_rgb_triplet(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*[int(c) for c in rgb])


def get_allen_color_for_acronym(atlas, acronym, cache, default_color=DEFAULT_AREA_COLOR):
    """
    Look up the official Allen RGB color for a structure acronym via the loaded
    brainrender/BrainGlobe atlas, caching results since many rows share an area.
    """
    if acronym in cache:
        return cache[acronym]
    try:
        rgb = atlas.structures[acronym]['rgb_triplet']
        color = _hex_from_rgb_triplet(rgb)
    except KeyError:
        color = default_color
    cache[acronym] = color
    return color


def get_row_acronyms_from_coords(atlas, probe_track, debug=False):
    """
    Look up the CCF acronym for each point in probe_track directly from atlas
    coordinates, bypassing the per-probe CSV. probe_track rows are assumed to
    already be in atlas space (AP, DV, ML), in microns.
    """
    acronyms = np.empty(len(probe_track), dtype=object)
    for i, p in enumerate(probe_track):
        try:
            acronym = atlas.structure_from_coords(p, microns=True, as_acronym=True)
            acronyms[i] = acronym if acronym else 'root'
        except Exception as e:
            if debug:
                print(f"    [row {i}] coord={p} → EXCEPTION: {e}")
            acronyms[i] = 'root'
    if debug:
        vals, counts = np.unique(acronyms, return_counts=True)
        print(f"    Acronym counts: {dict(zip(vals, counts))}")
    return acronyms


def add_probe_colored_by_allen_area(scene, probe_track, row_acronyms, probe_name, color_cache):
    """
    Split a probe track into contiguous runs sharing the same CCF acronym and add one
    Points actor per run, colored with that structure's official Allen RGB color.
    """
    colors = [get_allen_color_for_acronym(scene.atlas, a, color_cache) for a in row_acronyms]
    n = len(row_acronyms)
    start = 0
    for end in range(1, n + 1):
        if end == n or colors[end] != colors[start]:
            seg = probe_track[start:end]
            if len(seg) > 0:
                scene.add(Points(seg, name=f"{probe_name}_{row_acronyms[start]}",
                                  colors=colors[start], radius=18, alpha=1.0, res=15))
            start = end


def add_single_probe(scene, data_folder, probe_array, probe_table, probe_info, color_by, params,
                      areas_to_show, target_areas_color_dict, experimenter, allen_color_cache, base_probe_color):
    """
    Load one probe's track from data_folder, apply shank filtering / SC trimming /
    coloring per color_by, and add it to scene. Shared by both the combined
    (generate_brain_visualization) and per-session (generate_per_session_visualizations)
    entry points so the loading/coloring logic lives in one place.
    Returns True if a probe actor was added, False if the probe was skipped.
    """
    probe_name        = probe_array.split('.')[0]
    probe_path        = data_folder / probe_array
    probe_table_path  = data_folder / probe_table

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
                return False
            rows_selected = probe_track_table[probe_track_table['ccf_acronym'].isin(matched)].index

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

    probe_color = base_probe_color
    if color_by == 'target_area':
        probe_color = target_areas_color_dict.get(probe_target, '#262626')
    elif color_by == 'experimenter':
        probe_color = EXPERIMENTER_CMAP.get(experimenter, '#262626')

    if color_by == 'area_acronym_custom':
        row_acronyms = get_row_acronyms_from_coords(scene.atlas, probe_track, debug=False)
        print(f"  Adding probe {probe_name}, split by Allen atlas color")
        add_probe_colored_by_allen_area(scene, probe_track, row_acronyms, probe_name, allen_color_cache)
    else:
        print(f"  Adding probe {probe_name} (color={probe_color})")
        scene.add(Points(probe_track, name=probe_name, colors=probe_color, radius=18, alpha=1.0, res=15))

    return True


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
              'experimenter': 'experimenter', 'area_acronym_custom': 'allen_color',
              'none': 'tracks'}.get(color_by, 'tracks')
    output_folder_path = output_folder_path / subdir
    output_folder_path.mkdir(parents=True, exist_ok=True)

    scene = Scene(inset=False, title="", screenshots_folder=output_folder_path, title_color='darkgrey',
                  atlas_name='allen_mouse_bluebrain_barrels_10um')

    # Cache of acronym -> hex color, reused across all probes/mice in this call
    allen_color_cache = {}

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
    else:
        target_areas_color_dict = {}

    mouse_info_df = pd.read_excel(params['mouse_info_path'])
    mouse_info_df.rename(columns={'mouse_name': 'mouse_id'}, inplace=True)
    mouse_info_df = mouse_info_df[
        (mouse_info_df['exclude'] == 0) &
        (mouse_info_df['exclude_ephys'] == 0) &
        (mouse_info_df['reward_group'].isin(['R+', 'R-'])) &
        (mouse_info_df['recording'] == 1) #pertains to day 0
        ]

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

                probe_path = data_folder / probe_array
                probe_arrays_total.append(str(probe_path))
                print(f"  Loading: {probe_path}")

                added = add_single_probe(scene, data_folder, probe_array, probe_table, probe_info, color_by, params,
                                          areas_to_show, target_areas_color_dict, experimenter, allen_color_cache,
                                          probe_color)
                if not added:
                    continue
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


def _run_combined_job(job_params: dict) -> str:
    """
    Worker entry point for one (color_by, camera_view, file_format) combination.
    Must be a module-level function (not a closure) so it can be pickled and sent
    to a separate worker process.
    """
    generate_brain_visualization(job_params)
    return f"{job_params['color_by']}/{job_params['camera_view']}/{job_params['file_format']}"


def run_combined_sweep_parallel(base_params: dict, color_by_sweep: list, camera_views: list,
                                 file_formats: list, max_workers: int = None) -> None:
    """
    Run generate_brain_visualization once per (color_by, camera_view, file_format)
    combination, each in its own worker process. Each combination builds its own
    independent brainrender Scene across all mice/sessions, so this parallelizes
    across the outer sweep only — a single combined figure is still built serially
    within one process.

    max_workers defaults to cpu_count() - 1. Since each worker opens its own
    brainrender/VTK context, keep this modest (e.g. 3-6) if you hit GPU/memory
    issues with many workers running at once.
    """
    combos = list(itertools.product(color_by_sweep, camera_views, file_formats))
    jobs = []
    for color, cam, fmt in combos:
        job_params = dict(base_params)  # shallow copy is enough: nested dicts (camera_options, animation) are read-only per job
        job_params['color_by']    = color
        job_params['camera_view'] = cam
        job_params['file_format'] = fmt
        jobs.append(job_params)

    max_workers = max_workers or max(1, (os.cpu_count() or 4) - 1)
    print(f"Running {len(jobs)} combined-visualization jobs across {max_workers} worker processes...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_combined_job, jp): jp for jp in jobs}
        for future in as_completed(futures):
            jp = futures[future]
            label = f"{jp['color_by']}/{jp['camera_view']}/{jp['file_format']}"
            try:
                future.result()
                print(f"  Done: {label}")
            except Exception as e:
                print(f"  FAILED: {label} — {e}")


def _process_single_mouse_sessions(params: dict, m_name: str, mouse_probes: pd.DataFrame,
                                    target_areas_color_dict: dict) -> tuple:
    """
    Process every valid session for one mouse: find its session folder, build one
    brainrender Scene per session, add all its probes, and save a screenshot.
    Self-contained (re-derives camera/zoom/colors from params rather than sharing
    state with the caller) so it can run standalone in its own worker process for
    parallel execution across mice. allen_color_cache is process-local — each
    worker rebuilds its own, so caching benefit is per-process, not global.
    Returns (m_name, n_figures_saved).
    """
    color_by        = params['color_by']
    transparent     = params['transparent']
    dark_background = params['dark_background']

    input_data_path_axel   = Path(params['input_data_path_axel'])
    input_data_path_myriam = Path(params['input_data_path_myriam'])

    brainrender.settings.SCREENSHOT_TRANSPARENT_BACKGROUND = transparent
    base_probe_color = "#ebf7ff" if dark_background else "#262626"
    if dark_background:
        brainrender.settings.BACKGROUND_COLOR = [0, 0, 0]

    camera_view = params['camera_view']
    camera      = params['camera_options'][camera_view]
    zoom        = {'frontal': 1.0, 'sagittal': 0.5, 'top': 1.7, 'angled': 1.3}[camera_view]

    areas_to_show = params['areas_to_show']
    overlay_areas = params['overlay_areas'] if areas_to_show and areas_to_show != [''] else False
    label_areas   = params['label_areas']

    allen_color_cache = {}

    print(f"── Mouse: {m_name}")
    experimenter = m_name[:2]
    base = input_data_path_axel if m_name.startswith('AB') else input_data_path_myriam

    probe_color = base_probe_color
    if color_by == 'reward_group':
        rg = mouse_probes['reward_group'].values[0]
        probe_color = 'forestgreen' if rg == 'R+' else 'blueviolet'

    n_figures = 0
    for date_val, date_probes in mouse_probes.groupby('date'):

        session_folder = find_session_folder(base, m_name, date_val)
        if session_folder is None:
            continue

        if m_name.startswith('AB') and int(m_name[2:]) < 80:
            data_folder = session_folder / 'brainreg' / 'manual_segmentation' / 'standard_space' / 'tracks'
        else:
            data_folder = session_folder / 'Anatomy' / 'fused' / 'registered' / 'segmentation' / 'atlas_space' / 'tracks'

        if not data_folder.exists() or not any(data_folder.iterdir()):
            log.warning("  Track folder missing/empty: %s — skipping session.", data_folder)
            continue

        probe_arrays = sorted(f for f in os.listdir(data_folder) if f.endswith('.npy') and f.startswith('imec'))
        probe_tables = sorted(f for f in os.listdir(data_folder) if f.endswith('.csv') and f.startswith('imec'))
        get_probe_id = lambda fname: fname.split('_')[0][-1]  # e.g. "imec0_mapped.npy" → "0"

        # Save into the session's own folder, e.g.
        # {analysis_root}/{analyzer}/data/{mouse_id}/{session_folder}/Anatomy/figures/
        # Always determined by params['analyzer'], regardless of which mouse's
        # own raw-data path the session was read from.
        output_session_folder = get_output_session_folder(params, m_name, session_folder)
        output_folder = output_session_folder / 'Anatomy' / 'figures'
        output_folder.mkdir(parents=True, exist_ok=True)

        scene = Scene(inset=False, title="", screenshots_folder=output_folder, title_color='darkgrey',
                      atlas_name='allen_mouse_bluebrain_barrels_10um')

        if overlay_areas:
            for area in areas_to_show:
                actor = scene.add_brain_region(area, alpha=0.1, hemisphere='left', silhouette=True)
                if label_areas:
                    scene.add_label(actor, area, size=500, color=None, radius=100, xoffset=0, yoffset=-500, zoffset=100)

        n_added = 0
        for probe_array, probe_table in zip(probe_arrays, probe_tables):
            probe_id   = get_probe_id(probe_array.split('.')[0])
            probe_info = date_probes[date_probes['probe_id'] == int(probe_id)]
            if probe_info.empty or not probe_info['valid'].values[0]:
                print(f"  Skipping {m_name} probe {probe_id} on {date_val} (missing or invalid).")
                continue

            added = add_single_probe(scene, data_folder, probe_array, probe_table, probe_info, color_by, params,
                                      areas_to_show, target_areas_color_dict, experimenter, allen_color_cache,
                                      probe_color)
            n_added += int(added)

        if n_added == 0:
            print(f"  {m_name} / {session_folder.name}: no valid probes — skipping figure.")
            scene.close()
            continue

        scene.render(interactive=False, camera=camera, zoom=zoom)
        fig_name = f"{m_name}_{session_folder.name}_probes_{color_by}_{camera_view}.{params['file_format']}"
        print(f"  {m_name} / {session_folder.name}: saving {n_added} probe(s) → {output_folder / fig_name}")
        scene.screenshot(name=fig_name, scale=params['scale'])
        scene.close()
        n_figures += 1

    return m_name, n_figures


def generate_per_session_visualizations(params: dict, parallel: bool = False, max_workers: int = None) -> None:
    """
    Render one figure per (mouse, session) rather than one combined figure across
    all mice/sessions. Each figure is saved into that session's own folder:
    {input_data_path}/{mouse_id}/{session_folder}/Anatomy/figures/
    Reuses the same probe metadata filtering, coloring, and camera settings as
    generate_brain_visualization, via the shared add_single_probe helper.

    Set parallel=True to process different mice concurrently in separate worker
    processes (via _process_single_mouse_sessions). Parallelism is across mice,
    not within a single mouse's sessions, and each worker builds its own
    brainrender/VTK Scene — keep max_workers modest if you hit GPU/memory issues.
    max_workers defaults to cpu_count() - 1.
    """
    color_by = params['color_by']
    days     = params['day_of_recording']
    assert isinstance(days, list)

    print(f"[per-session] color_by={color_by}  camera={params['camera_view']}  format={params['file_format']}  days={days}")

    probe_info_df = pd.read_excel(params['probe_info_path'])
    probe_info_df = probe_info_df[probe_info_df['valid'] == 1]
    probe_info_df['day_of_recording'] = probe_info_df['day_of_recording'].astype(int)
    probe_info_df = probe_info_df[probe_info_df['day_of_recording'].isin(days)]

    if color_by == 'target_area':
        target_areas = [a for a in probe_info_df['target_area'].unique() if isinstance(a, str)]
        if params['target_cmap'] == 'auto':
            random.seed(42)
            colors = ['#' + hashlib.md5(f'42-{i}'.encode()).hexdigest()[:6] for i in range(len(target_areas))]
            target_areas_color_dict = dict(zip(target_areas, colors))
        else:
            target_areas_color_dict = TARGET_AREA_CUSTOM_CMAP
    else:
        target_areas_color_dict = {}

    mouse_info_df = pd.read_excel(params['mouse_info_path'])
    mouse_info_df.rename(columns={'mouse_name': 'mouse_id'}, inplace=True)
    mouse_info_df = mouse_info_df[mouse_info_df['exclude'] == 0]
    mouse_info_df = mouse_info_df[mouse_info_df.learning_category.isin(params['learning_category'])]

    mouse_list     = probe_info_df['mouse_name'].unique()
    mouse_list_sub = [m for m in mouse_list if m in mouse_info_df['mouse_id'].unique()]

    # Optionally restrict to a specific set of mice, e.g. params['mouse_ids'] = ['AB123', 'MH045'].
    # None or an empty list means "all mice" (no filtering).
    mouse_ids = params.get('mouse_ids')
    if mouse_ids:
        missing = [m for m in mouse_ids if m not in mouse_list_sub]
        if missing:
            print(f"  WARNING: requested mouse_ids not found in filtered probe/mouse tables: {missing}")
        mouse_list_sub = [m for m in mouse_list_sub if m in mouse_ids]
    print(f"  Processing {len(mouse_list_sub)} mice: {mouse_list_sub}")

    mouse_jobs = [(m_name, probe_info_df[probe_info_df['mouse_name'] == m_name].copy()) for m_name in mouse_list_sub]

    if not parallel or len(mouse_jobs) <= 1:
        for m_name, mouse_probes in mouse_jobs:
            _process_single_mouse_sessions(params, m_name, mouse_probes, target_areas_color_dict)
        return

    max_workers = max_workers or max(1, (os.cpu_count() or 4) - 1)
    print(f"Processing {len(mouse_jobs)} mice across {max_workers} worker processes...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_single_mouse_sessions, params, m_name, mouse_probes, target_areas_color_dict): m_name
            for m_name, mouse_probes in mouse_jobs
        }
        for future in as_completed(futures):
            m_name = futures[future]
            try:
                _, n_figures = future.result()
                print(f"  {m_name}: {n_figures} figure(s) saved.")
            except Exception as e:
                print(f"  {m_name}: FAILED — {e}")


# Define the dictionary with parameters
params = {
    'input_data_path_axel': r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Axel_Bisi\data',
    'input_data_path_myriam': r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Myriam_Hamon\data',
    'probe_info_path': r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\share_internal\Axel_Bisi_Share\dataset_info\joint_probe_insertion_info.xlsx',
    'mouse_info_path': r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\share_internal\Axel_Bisi_Share\dataset_info\joint_mouse_reference_weight.xlsx',
    'output_folder_path': r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Axel_Bisi\combined_results_ks4\anatomy',
    'color_by': 'reward_group',  # 'reward_group', 'target_area', 'experimenter', 'area_acronym_custom', 'none'
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
    'areas_to_show': [],  # e.g. ['wS1', 'wM1'] or [''] for none
    'filter_shanks': False,
    'overlay_areas': True,
    'label_areas': False,
    'scale': 3,
    'file_format': 'png',  # 'png', 'svg', 'pdf', 'eps'
    'learning_category': ['moderate','good'],
    'day_of_recording': [0],  # list of day_of_recording values, or [0,1,...] for multiple
    'mouse_ids': [],  # e.g. ['AB123', 'MH045'] to restrict per-session figures to specific mice; None/[] = all mice
    'analyzer': 'Axel_Bisi',  # always determines where per-session figures are saved: analysis/<analyzer>/data/<mouse>/<session>/Anatomy/figures
    'animate': True,          # set True to also render a rotating video for each sweep iteration
    'animation': {
        'fps':      30,        # frames per second
        'duration': 8,         # seconds for a full 360° rotation (longer = slower)
        'zoom': 0.2,  # adjust per view if needed

    },
}

if __name__ == "__main__":
    mode     = 'combined'  # 'combined' (one figure across all mice/sessions) or 'per_session'
    parallel = True        # run jobs across multiple worker processes
    max_workers = 30     # None = cpu_count() - 1; lower this if you hit GPU/memory issues

    if mode == 'combined':
        color_by_sweep = ['area_acronym_custom', 'reward_group', 'target_area', 'none']
        camera_views   = ['sagittal', 'top', 'angled', 'frontal']
        file_formats   = ['png', 'svg', 'pdf']

        if parallel:
            run_combined_sweep_parallel(params, color_by_sweep, camera_views, file_formats, max_workers=max_workers)
        else:
            for color in color_by_sweep:
                params['color_by'] = color
                for camera_view in camera_views:
                    params['camera_view'] = camera_view
                    for file_format in file_formats:
                        params['file_format'] = file_format
                        generate_brain_visualization(params)

    else:
        # Single direct call: one figure per mouse/session, uncolored ('none'),
        # restricted to whichever mice are listed in params['mouse_ids'].
        params['color_by']    = 'none'
        params['camera_view'] = 'top'
        params['file_format'] = 'png'
        params['mouse_ids']   = ['MH001']  # <-- set the mice to render here
        params['analyzer']    = 'Axel_Bisi'         # <-- ALL figures save under analysis/Axel_Bisi/data/..., regardless of mouse
        generate_per_session_visualizations(params, parallel=parallel, max_workers=max_workers)