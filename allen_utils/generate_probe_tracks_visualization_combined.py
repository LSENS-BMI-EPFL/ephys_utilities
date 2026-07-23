#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: brainrender
@file: generate_probe_tracks_visualization.py
@time: 3/3/2025 12:53 PM
"""

import os
import pandas as pd
from pathlib import Path
import numpy as np
import hashlib
import random

import brainrender
from brainrender import Scene
from brainrender.actors import Points


brainrender.settings.SHOW_AXES = False
brainrender.settings.ROOT_ALPHA = 0.1
brainrender.settings.ROOT_COLOR = [0.99, 0.99, 0.99]
brainrender.settings.SHADER_STYLE = 'cartoon'

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
    'OFC': '##54110c',
    'SC': '#8a6c27'
}

EXPERIMENTER_CMAP = {
    'AB': '#127ac4',
    'MH': '#6915bd'
}

def generate_brain_visualization(params):
    """
    Using specified parameters, plot probe tracks with brainrender.
    """

    print('Generating probe track visualization...')
    # -------------------------------
    # Paths
    # -------------------------------
    input_data_path_axel = Path(params['input_data_path_axel'])
    input_data_path_myriam = Path(params['input_data_path_myriam'])
    probe_info_path = Path(params['probe_info_path'])
    mouse_info_path = Path(params['mouse_info_path'])
    output_folder_path = Path(params['output_folder_path'])

    # -------------------------------
    # Customize for transparency and settings
    # -------------------------------
    color_by = params['color_by']
    transparent = params['transparent']
    dark_background = params['dark_background']

    if transparent:
        brainrender.settings.SCREENSHOT_TRANSPARENT_BACKGROUND = True
        probe_color = "#262626"  # light grey blue-ish
    else:
        brainrender.settings.SCREENSHOT_TRANSPARENT_BACKGROUND = False
        probe_color = "#262626"  # very dark gray

    if dark_background:
        brainrender.settings.BACKGROUND_COLOR = [0, 0, 0]
        probe_color = "#ebf7ff"  # light grey blue-ish

    # -------------------------------
    # Camera options
    # -------------------------------
    camera_view = params['camera_view']
    camera_options = params['camera_options']
    camera = camera_options[params['camera_view']]  # e.g., "angled", "frontal", "top", "sagittal"
    if camera_view == 'frontal':
        zoom=1.0
    elif camera_view == 'sagittal':
        zoom=0.5
    elif camera_view == 'top':
        zoom=1.7
    elif camera_view =='angled':
        zoom=1.3
    cam_suffix = params['camera_view']

    # -------------------------------
    # Areas to show
    # -------------------------------
    areas_to_show = params['areas_to_show']
    str_suffix = '_'.join(areas_to_show)
    overlay_areas = params['overlay_areas']
    label_areas = params['label_areas']
    if not areas_to_show:
        overlay_areas = False
        str_suffix = 'empty'

    # -------------------------------
    # Scene setup
    # -------------------------------
    if color_by == 'reward_group':
        output_folder_path = output_folder_path / 'reward_group'
    elif color_by == 'target_area':
        output_folder_path = output_folder_path / 'target_area'
    elif color_by == 'experimenter':
        output_folder_path = output_folder_path / 'experimenter'
    elif color_by == 'none':
        output_folder_path = output_folder_path / 'tracks'
    output_folder_path.mkdir(parents=True, exist_ok=True)

    scene = Scene(inset=False, title="", screenshots_folder=output_folder_path, title_color='darkgrey',
                  atlas_name='allen_mouse_bluebrain_barrels_10um')

    if overlay_areas:
        for area in areas_to_show:
            actor = scene.add_brain_region(area, alpha=0.1, hemisphere='left', silhouette=True)
            if label_areas:
                scene.add_label(actor, area, size=500, color=None, radius=100, xoffset=0, yoffset=-500, zoffset=100)

    # -------------------------------
    # Probe and mouse information
    # -------------------------------
    probe_info_df = pd.read_excel(probe_info_path)
    probe_info_df = probe_info_df[probe_info_df['anatomy'] == 1]  # anatomical data exists
    mouse_list = probe_info_df.mouse_name.unique()

    if color_by == 'target_area':
        def hash_color(seed, num_colors):
            random.seed(seed)  # Set the seed for reproducibility
            colors = []
            for i in range(num_colors):
                # Create a hash based on the seed and index
                seed_bytes = f'{seed}-{i}'.encode('utf-8')
                hash_object = hashlib.md5(seed_bytes)
                hex_color = '#' + hash_object.hexdigest()[:6]  # Use the first 6 characters for a CSS hex color
                colors.append(hex_color)
            return colors

        # Create a color dict for each target region
        target_areas = probe_info_df['target_area'].unique()
        target_areas = [r for r in target_areas if isinstance(r, str)]
        if params['target_cmap'] == 'auto':
            target_areas_color_dict = {area: c for (c, area) in
                                   zip(hash_color(seed=42, num_colors=len(target_areas)), target_areas)}
        elif params['target_cmap'] == 'custom_cmap':
            target_areas_color_dict = TARGET_AREA_CUSTOM_CMAP  # or use custom color map

    # -------------------------------
    # Mouse info
    # -------------------------------
    mouse_info_df = pd.read_excel(mouse_info_path)
    mouse_info_df.rename(columns={'mouse_name': 'mouse_id'}, inplace=True)
    mouse_info_df = mouse_info_df[mouse_info_df['exclude'] == 0]  # included mice

    # Keep only included mice from the recordings
    mouse_list_sub = [m for m in mouse_list if m in mouse_info_df.mouse_id.unique()]
    #mouse_list = [m for m in mouse_list if int(m[2:])>=116]

    probe_arrays_total = []
    mouse_list_sub_in_plot = []

    for m_name in mouse_list_sub:
        print(f'Plotting probe tracks for mouse: {m_name}')

        experimenter = m_name[0:2]

        # Axel's data
        if m_name.startswith('AB'):
            input_data_path = input_data_path_axel
            #if int(m_name[2:]) < 116: # TODO: resolve which NWB version used
            #if int(m_name[2:]) < 94: # older brainglobe output
            if int(m_name[2:]) < 80:  # older brainglobe output
                 data_folder = Path(
                os.path.join(input_data_path, m_name, 'brainreg', 'manual_segmentation', 'standard_space', 'tracks'))
            else:
                data_folder = Path(
                    os.path.join(input_data_path, 'Axel_Bisi', m_name, 'fused', 'registered', 'segmentation', 'atlas_space',
                                 'tracks'))
                data_folder = Path(
                    os.path.join(input_data_path, m_name, 'Anatomy', 'fused', 'registered', 'segmentation', 'atlas_space',
                                 'tracks'))
        # Myriam's data
        elif m_name.startswith('MH'):
            input_data_path = input_data_path_myriam
            data_folder = Path(os.path.join(input_data_path, 'Myriam_Hamon', m_name, 'fused', 'registered', 'segmentation', 'atlas_space',
                 'tracks'))
            data_folder = Path(os.path.join(input_data_path, m_name, 'Anatomy', 'fused', 'registered', 'segmentation', 'atlas_space',
                 'tracks'))

        # Get reward group
        if color_by == 'reward_group':
            reward_group = probe_info_df[probe_info_df.mouse_name == m_name]['reward_group'].values[0]
            if reward_group == 'R+':
                probe_color = 'forestgreen'
            elif reward_group == 'R-':
                probe_color = 'crimson'

        if not os.path.exists(data_folder):
            continue

        # Get probe imec files
        probe_arrays = sorted([f for f in os.listdir(data_folder) if f.endswith('.npy') and f.startswith('imec')])
        probe_tables = sorted([f for f in os.listdir(data_folder) if f.endswith('.csv') and f.startswith('imec')])
        probe_files = zip(probe_arrays, probe_tables)

        # Iterate over probe files
        if m_name.startswith('AB') or m_name.startswith('MH'):
            process_id_func = lambda x: x.split('_')[0][-1] # get probe ID number, ignoring any suffixes separated by underscores

        for probe_array, probe_table in probe_files:
            probe_name = probe_array.split('.')[0] # follows naming of the probes files e.g. "imec0.npy"
            probe_id = process_id_func(probe_name)

            # Get probe insertion info
            probe_info = probe_info_df[(probe_info_df.mouse_name == m_name)
                                       & (probe_info_df.probe_id == int(probe_id))]
            is_valid = probe_info['valid'].values[0]
            if not is_valid:
                continue

            probe_path = os.path.join(data_folder, probe_array)
            probe_table_path = os.path.join(data_folder, probe_table)
            probe_arrays_total.append(probe_path)
            if m_name.startswith('AB'):
                if int(m_name[2:]) < 100 and 'mapped' not in probe_name:  # correction files for wrong hemisphere, for some mice only #TODO: update as this should be redone now
                    continue

            # ----------------------------------------------------
            # Optional: Load probe areas for filtering probe spans
            # ----------------------------------------------------
            if params['filter_shanks']:
                probe_track_table = pd.read_csv(probe_table_path)
                col_mapper = {'Region acronym': 'ccf_acronym', 'Distance from first position [um]':'distance_from_surface', 'Position':'distance_from_surface'}
                probe_track_table.rename(columns=col_mapper, inplace=True)

                probe_track_areas = probe_track_table['ccf_acronym'].unique() # find present areas
                #probe_areas_filtered = [s for s in probe_track_areas if 'MOs' in s] # get areas of interest
                probe_areas_filtered = [s for s in probe_track_areas if s in areas_to_show] # get areas of interest
                if not probe_areas_filtered:
                    continue
                rows_selected = probe_track_table[probe_track_table['ccf_acronym'].isin(probe_areas_filtered)]['distance_from_surface'].index
                rows_selected_range = (min(rows_selected)-1, max(rows_selected)) #Note:for NP1.0 only

            # Load probe tracks
            probe_track = np.load(probe_path, allow_pickle=True)

            # Interpolate straight line between two extrema
            probe_length = probe_track.shape[0]
            probe_track = np.linspace(probe_track[0], probe_track[-1], num=probe_length)

            # For deeper SC probes, only plot the last 200 rows (bank 0)
            probe_target = probe_info['target_area'].values[0]
            if probe_target == 'SC' and probe_info['depth'].values[0] > 4000:
                probe_track = probe_track[-200:]

            if params['filter_shanks']:
                if rows_selected.shape[0]!=0:
                    #print(rows_selected, rows_selected_range)
                    probe_track = probe_track[np.array(rows_selected),:]
                    #probe_track = probe_track[rows_selected_range[0]:rows_selected_range[1],:]
                    #print(probe_track.shape)

            if color_by == 'target_area':
                probe_color = target_areas_color_dict[probe_target]
            elif color_by == 'experimenter':
                probe_color = EXPERIMENTER_CMAP[experimenter]


            # Add probe to scene
            p = scene.add(
                Points(
                    probe_track,
                    name=probe_name,
                    colors=probe_color,  # very dark gray
                    radius=20,  # shank width is 70 microns
                    alpha=1.0,
                    res=15,
                )
            )
            mouse_list_sub_in_plot.append(m_name)

            # Optionally, add a label to the probe
            # scene.add_label(p, f'{m_name}_{probe_id}')

    # -------------------------------
    # Render Scene
    # -------------------------------
    scene.render(interactive=False, camera=camera, zoom=zoom)

    # -------------------------------
    # Save Screenshot
    # -------------------------------
    scale = params['scale']
    file_format = params['file_format']
    if transparent:
        if color_by== 'reward_group':
            str_suffix += '_reward_group'
        if color_by== 'target_area':
            str_suffix += '_target_region'
        if color_by == 'experimenter':
            str_suffix += '_experimenter'
        if params['filter_shanks']:
            str_suffix += '_filtered'
        if dark_background:
            str_suffix += '_dark'
        fig_name = "all_probes_in_atlas_{}_{}_transparent.{}".format(str_suffix, cam_suffix,
                                                                     file_format)  # else, name is current time

    else:
        if color_by=='reward_group':
            str_suffix += '_reward_group'
        if color_by=='target_area':
            str_suffix += '_target_region'
        if color_by == 'experimenter':
            str_suffix += '_experimenter'
        if params['filter_shanks']:
            str_suffix += '_filtered'
        if dark_background:
            str_suffix += '_dark'
        fig_name = "all_probes_in_atlas_{}_{}.{}".format(str_suffix, cam_suffix, file_format)

    mouse_list_sub_in_plot = list(set(mouse_list_sub_in_plot))  # unique mice in plot
    print(f'Saving figures showing {len(probe_arrays_total)} probe insertions in {len(mouse_list_sub_in_plot)} mice.')
    print(f'Saving figure at location {output_folder_path / fig_name}')
    scene.screenshot(name=fig_name, scale=scale)
    scene.close()


# Define the dictionary with parameters
params = {
    #'input_data_path_axel': r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Axel_Bisi\ImagedBrains',
    #'input_data_path_myriam': r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Myriam_Hamon\ImagedBrains',
    'input_data_path_axel': r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Axel_Bisi\data',
    'input_data_path_myriam': r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Myriam_Hamon\data',
    'probe_info_path': r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\share_internal\Axel_Bisi_Share\dataset_info\joint_probe_insertion_info.xlsx',
    'mouse_info_path': r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\share_internal\Axel_Bisi_Share\dataset_info\joint_mouse_reference_weight.xlsx',
    'output_folder_path': r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Axel_Bisi\combined_results\anatomy',
    'color_by': 'reward_group',  # 'reward_group', 'target_area', 'none'
    'target_cmap': 'custom_cmap', #'auto' or 'custom_cmap'
    'transparent': True,
    'dark_background': False,

    'camera_options': {
        "sagittal": {
            "pos": (5053.37, 2365.62, -19213.1),
            "focal_point": (6587.84, 3849.09, -5688.16),
            "viewup": (-0.0169435, -0.993686, 0.110913),
            "roll": 178.319,
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
    'camera_view': 'angled',  # Change to "sagittal", "frontal", "top", "angled"
    #'areas_to_show': ['STR', 'Isocortex', 'TH', 'MB','HPF'],  # Add specific areas if needed
    'areas_to_show': [''],  # Add specific areas if needed
    'filter_shanks':True,
    'overlay_areas': False,
    'label_areas': False,
    'scale': 3,
    'file_format': 'png',  # 'png', 'svg', 'pdf', 'eps'
}

color_by = ['target_area', 'reward_group', 'none'] #'target_area', 'reward_group', 'none', 'experimenter'
#color_by = ['none']
camera_views = ['sagittal', 'frontal', 'top', 'angled']
file_formats = ['png', 'svg']
for color in color_by:
    params['color_by'] = color
    for camera_view in camera_views:
        params['camera_view'] = camera_view
        for file_format in file_formats:
            params['file_format'] = file_format
            generate_brain_visualization(params)
