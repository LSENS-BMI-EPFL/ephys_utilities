#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: brain_wide_analysis
@file: plotting_utils.py
@time: 2/7/2024 1:35 PM
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.colors as mc
import colorsys
import cmasher as cmr
import random
import copy

from ephys_utilities import allen_utils as allen

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

CP_SUB_AREA_CMAP = {
    'DLS':'#9d5bf5',
    'DMS':'#431663',
    'VS':'#69a6f0',
    'TS':'#d769f0',
    'CP':'k'
}
MOS_SUB_AREA_CMAP = { # by location subdivision
    'MOs-p':'#66e85f',
    'MOs-m':'#3e9c3a',
    'MOs-a':'#295e26',
}
MO_SUB_AREA_CMAP = { # by target
    'MO-wM1':'#a8e63e',
    'MO-wM2':'#66e85f',
    'MO-tjM1':'#3e9c3a',
    'MO-ALM':'#1c451a',
}


def get_hot_to_cold_colors():
    hotcold = ['#AEFDFF', '#60FDFA', '#2ADEF6', '#2593FF', '#2D47F9', '#3810DC', '#3D019D',
               '#313131',
               '#97023D', '#D90D39', '#F8432D', '#FF8E25', '#F7DA29', '#FAFD5B', '#FFFDA9']
    return hotcold

def get_passive_hot_to_cold_colors():
    hotcold = ['#99C2FF', '#0041A3',
               '#313131',
               '#AA0531', '#FC9CB5']
    return hotcold


def format_probe_trajectory_dict(nwb_location_dict):
    """
    Format NWB location dictionary for insertion trajectory.
    :param nwb_location_dict:
    :return:
    """
    formatted_dict = {}
    keys_to_keep = ['azimuth', 'elevation', 'depth']
    for key, value in nwb_location_dict.items():
        if key=='azimuth':
            value = str(value)+'°'
        elif key=='elevation':
            value = str(value)+'°'
        elif key=='depth':
            value = str(value)+r' $\mu$m'
        formatted_dict[key] = value
    dict_2_string = ', '.join([f'{key}: {value}' for key, value in formatted_dict.items() if key in keys_to_keep])
    return dict_2_string


def format_probe_target_dict(nwb_location_dict):
    """
    Format NWB location dictionary for stereotaxic insertion target.
    :param nwb_location_dict:
    :return:
    """
    keys_to_keep = ['area', 'AP', 'ML']
    nwb_location_dict['AP'] = nwb_location_dict.pop('ap')
    nwb_location_dict['ML'] = nwb_location_dict.pop('ml')
    dict_2_string = ', '.join([f'{key}: {value}' for key, value in nwb_location_dict.items() if key in keys_to_keep])
    if 'SC' in dict_2_string:
        nwb_location_dict['ap'] = -3.8
        nwb_location_dict['ml'] = 0.5
    return dict_2_string


def remove_top_right_frame(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return

def remove_bottom_right_frame(ax):
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return

def color_to_rgba(color_name):
    """
    Converts color name to RGB.
    :param color_name:
    :return:
    """

    return colors.to_rgba(color_name)

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    From: https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    :param color: Matplotlib color string.
    :param amount: Number between 0 and 1.
    :return:
    """

    try:
        c = mc.cnames[color]
    except KeyError:
        c = color

    h, l, s = colorsys.rgb_to_hls(*mc.to_rgb(c))
    # new lightness
    l_new = 1 - amount * (1 - l)
    # clamp to [0,1]
    l_new = min(max(l_new, 0), 1)

    return colorsys.hls_to_rgb(h, l_new, s)


def adjust_lightness(color, amount=0.5):
    """
    Same as lighten_color but adjusts brightness to lighter color if amount>1 or darker if amount<1.
    Input can be matplotlib color string, hex string, or RGB tuple.
    From: https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    :param color: Matplotlib color string.
    :param amount: Number between 0 and 1.
    :return:
    """

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def adjust_saturation(color, amount=1.0):
    """
    Adjust the saturation of a color.
    Input can be matplotlib color string, hex string, or RGB tuple.
    :param color: Matplotlib color string, hex string, or RGB tuple.
    :param amount: Saturation multiplier (>1 = more saturated, <1 = less saturated).
    :return: RGB tuple with adjusted saturation.
    """
    # Convert color string or hex to RGB tuple
    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    r, g, b = mc.to_rgb(c)
    # Convert to HLS
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    # Adjust saturation
    s = max(0, min(1, s * amount))
    # Convert back to RGB
    return colorsys.hls_to_rgb(h, l, s)



def make_cmap_n_from_color_lite2dark(color, N):
    """
    Make ListedColormap from matplotlib color of size N using the lighten_color function.
    :param color: Matplotlib color string.
    :param N: Number of colors to have in cmap.
    :return:
    """
    light_factors = np.linspace(0.2, 1, N)
    cmap = colors.ListedColormap(colors=[lighten_color(color, amount=i) for i in light_factors])
    return cmap


def make_diverging_cmap(color1, color2, N, midpoint_color='grey'):
    """
    Make a diverging ListedColormap from two colors with N steps, having a midpoint color.

    :param color1: Matplotlib color string for the start color.
    :param color2: Matplotlib color string for the end color.
    :param N: Number of colors to have in cmap.
    :param midpoint_color: Color at the midpoint of the colormap.
    :return: ListedColormap object.
    """
    light_factors = np.linspace(0.2, 1, (N - 1) // 2)
    colors1 = [lighten_color(color1, amount=factor) for factor in light_factors]
    colors1 = colors1[::-1]
    colors2 = [lighten_color(color2, amount=factor) for factor in light_factors]
    if midpoint_color is None:
        colors_combined = colors1 + colors2[:-1]
    else:
        if N % 2 == 0:
            colors_combined = colors1 + [midpoint_color] + colors2
        else:
            colors_combined = colors1 + [midpoint_color] + colors2[:-1]
    cmap = colors.ListedColormap(colors_combined)
    return cmap

def make_cmap_from_color_list(color_list, cmap_name='custom_cmap', n_bins=100):
    cm = mc.LinearSegmentedColormap.from_list(cmap_name, color_list, N=n_bins)
    return cm

def make_cmap_discrete_random(N, saturation=0.65, lightness=0.5):
    """
    Create a discrete color palette with N distinct colors using HLS color space.
    :param N:
    :param saturation:
    :param lightness:
    :return:
    """
    palette = []
    for i in range(N):
        hue = i / N  # Evenly spaced hues
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        palette.append(rgb)
    random.shuffle(palette)
    return palette

def combine_cmaps(cmap1_name, cmap2_name, name='combined', ratio=0.5, N=256):
    """
    Concatenate two cmasher colormaps into a single colormap.
    :param cmap1_name: Name of first colormap (string)
    :param cmap2_name: Name of second colormap (string)
    :param name: Name of the new combined colormap
    :param ratio: Fraction of cmap1 vs cmap2 in total length (0–1)
    :param N: Total number of colors
    :return: Combined LinearSegmentedColormap
    """
    cmap1 = cmr.get_sub_cmap(cmap1_name, 0, 1)
    cmap2 = cmr.get_sub_cmap(cmap2_name, 0, 1)

    n1 = int(N * ratio)
    n2 = N - n1

    colors1 = cmap1(np.linspace(0, 1, n1))
    colors2 = cmap2(np.linspace(0, 1, n2))
    colors = np.vstack((colors1, colors2))

    return mc.LinearSegmentedColormap.from_list(name, colors)

def diverging_cmap(cmap_left, cmap_right, name='diverging', N=256):
    left = cmr.get_sub_cmap(cmap_left, 0, 1)(np.linspace(1, 0, N//2))
    right = cmr.get_sub_cmap(cmap_right, 0, 1)(np.linspace(0, 1, N//2))
    colors = np.vstack([left, right])
    return mc.LinearSegmentedColormap.from_list(name, colors)

def create_normalizer(values, N):
    """
    Create a normalizer to map data values to the colormap.

    :param values: List or array of data values.
    :param N: Number of colors in the colormap.
    :return: BoundaryNorm object.
    """
    boundaries = np.linspace(min(values), max(values), N + 1)
    norm = colors.BoundaryNorm(boundaries, N)
    return norm


def get_color_by_index(index, N, cmap):
    """
    Get the color from the colormap corresponding to an integer index.

    :param index: Integer index for which the color is needed.
    :param N: Number of colors in the colormap.
    :param cmap: The colormap.
    :return: The color corresponding to the index.
    """
    if index < 0 or index >= N:
        raise ValueError(f"Index {index} is out of bounds for colormap of size {N}")
    normalized_value = index / (N - 1)  # Normalize the index to the range [0, 1]
    return cmap(normalized_value)

def save_figure_to_files(fig, save_path, file_name, suffix=None, file_types=list, dpi=500):
    """
    Save figure to file.
    :param fig: Figure to save.
    :param save_path: Path to save figure.
    :param file_name: Name of file.
    :param suffix: Suffix to add to file name.
    :param file_types: List of file types to save.
    :param dpi: Resolution of figure.
    :return:
    """

    if file_types is None:
        file_types = ['png', 'eps', 'pdf']

    if suffix is not None:
        file_name = file_name + '_' + suffix

    for file_type in file_types:
        file_format = '.{}'.format(file_type)
        file_path = os.path.join(save_path, file_name + file_format)

        print('Saving in: {}'.format(file_path))
        if file_type == 'eps':
            fig.savefig(file_path, dpi=dpi, bbox_inches='tight', transparent=True)
        else:
            fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
    return


def save_figure_with_options(figure, file_formats, filename, output_dir='', dark_background=False):
    # Make transparent for dark background
    if dark_background:
        figure.patch.set_alpha(0)
        figure.set_facecolor('#f4f4ec')
        for ax in figure.get_axes():
            ax.set_facecolor('#f4f4ec')
        #plt.rcParams.update({'axes.facecolor': '#f4f4ec',  # very pale beige
        #                        'figure.facecolor': '#f4f4ec'})
        transparent = True
        filename = filename + '_transparent'
    else:
        transparent = False

    # Save the figure in each specified file format
    for file_format in file_formats:
        file_path = os.path.join(output_dir, f"{filename}.{file_format}")
        figure.savefig(file_path, transparent=transparent, bbox_inches='tight', dpi='figure')

    return

def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    """
    Render a matplotlib table.
    :param data:
    :param col_width:
    :param row_height:
    :param font_size:
    :param header_color:
    :param row_colors:
    :param edge_color:
    :param bbox:
    :param header_columns:
    :param ax:
    :param kwargs:
    :return:
    """
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])

    return ax.get_figure(), ax

def save_figure_with_options(figure, file_formats, filename, output_dir='', dark_background=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Make transparent for dark background
    if dark_background:
        figure.patch.set_alpha(0)
        figure.set_facecolor('#f4f4ec')
        for ax in figure.get_axes():
            ax.set_facecolor('#f4f4ec')
        #plt.rcParams.update({'axes.facecolor': '#f4f4ec',  # very pale beige
        #                        'figure.facecolor': '#f4f4ec'})
        transparent = True
        filename = filename + '_transparent'
    else:
        transparent = False

    # Save the figure in each specified file format
    for file_format in file_formats:
        file_path = os.path.join(output_dir, f"{filename}.{file_format}")
        figure.savefig(file_path, transparent=transparent, bbox_inches='tight', dpi='figure')

    return


def plot_matrix_with_label(matrix, matrix_params, label_colors=None, patch_width=0.02):
    """
    Plot a matrix with color patches for labels.
    :param matrix:
    :param matrix_params:
    :param label_colors:
    :param patch_width:
    :return:
    """
    matrix = matrix[:matrix_params['n_max_trials'],:]
    rows, cols = matrix.shape
    label_colors = label_colors[:rows]

    # Create a grid of subplots
    fig, (ax_matrix, ax_patches) = plt.subplots(1, 2,
                                                gridspec_kw={'width_ratios': [1, patch_width]},
                                                figsize=(10, 6),
                                                dpi=500)

    # Plot matrix using specified colormap
    im = ax_matrix.imshow(matrix,
                     cmap=matrix_params['cmap'],
                     aspect='auto',
                     interpolation='none',
                     vmin=matrix_params['vmin'],
                     vmax=matrix_params['vmax']
                     )

    ## Add colorbar
    #divider = make_axes_locatable(ax_matrix)
    #cax = divider.new_vertical(size="5%", pad=0.5, pack_start=True)
    #fig.add_axes(cax)
    #cbar = fig.colorbar(im, cax=cax)
    #cbar.ax.tick_params(labelsize=12)
    #cbar_label = matrix_params['cbar_label']
    #cbar.set_label(cbar_label, fontsize=14)

    # Add color patches for labels outside the matrix subfigure
    if label_colors is not None:
        for i, color in enumerate(label_colors):
            ax_patches.fill_between([0, 1], rows-i-1, rows-i, color=color) # this fills patches top-down

    # Set limits and aspect ratio for the patch subplot to make it touch the matrix
    ax_patches.set_xlim(0, 1)
    ax_patches.set_ylim(0, rows)
    ax_patches.set_aspect('auto')

    # Remove axis ticks and labels for the patches subplot
    ax_patches.axis('off')

    # Adjust the space between subplots
    plt.subplots_adjust(wspace=0.01)

    return fig, (ax_matrix, ax_patches)


def align_y_axis(axs):
    """
    Align the y-axis of all subplots in a given row or column.

    Parameters:
    axs : 1D or 2D array-like of matplotlib.axes.Axes
        A 1D or 2D array or list of matplotlib Axes objects.
    """
    # Check if axs is 2D array-like
    if hasattr(axs[0], '__iter__'):
        # It's a 2D array-like, align y-axis for each row
        for row in axs:
            min_y = float('inf')
            max_y = float('-inf')
            for ax in row:
                y_min, y_max = ax.get_ylim()
                min_y = -2
                max_y = max(max_y, y_max) * 3

            for ax in row:
                ax.set_ylim(min_y, max_y)
    else:
        # It's a 1D array-like, align y-axis for the whole array
        min_y = float('inf')
        max_y = float('-inf')
        for ax in axs:
            y_min, y_max = ax.get_ylim()
            min_y = -2
            max_y = max(max_y, y_max) * 3

        for ax in axs:
            ax.set_ylim(min_y, max_y)


def align_x_axis(axs):
    """
    Align the x-axis of all subplots in a given row or column.

    Parameters:
    axs : 1D or 2D array-like of matplotlib.axes.Axes
        A 1D or 2D array or list of matplotlib Axes objects.
    """
    # Check if axs is 2D array-like
    if hasattr(axs[0], '__iter__'):
        # It's a 2D array-like, align x-axis for each column
        num_cols = len(axs[0])
        for col in range(num_cols):
            min_x = float('inf')
            max_x = float('-inf')
            for row in axs:
                x_min, x_max = row[col].get_xlim()
                min_x = min(min_x, x_min)
                max_x = max(max_x, x_max)

            for row in axs:
                row[col].set_xlim(min_x, max_x)
    else:
        # It's a 1D array-like, align x-axis for the whole array
        min_x = float('inf')
        max_x = float('-inf')
        for ax in axs:
            x_min, x_max = ax.get_xlim()
            min_x = min(min_x, x_min)
            max_x = max(max_x, x_max)

        for ax in axs:
            ax.set_xlim(min_x, max_x)

    return

def apply_y_ticks_to_all_subplots(axs):
    """

    :param axs:
    :return:
    """
    # Determine if axs is 2D array-like
    if hasattr(axs[0], '__iter__'):
        # It's a 2D array-like, find the global maximum y-value
        max_y = float('-inf')
        min_y = float('inf')
        for row in axs:
            for ax in row:
                y_max = ax.get_ylim()[1]
                y_min = ax.get_ylim()[0]
                max_y = max(max_y, y_max)
                min_y = min(min_y, y_min)
        # Set the same y-axis range for all subplots
        for row in axs:
            for ax in row:
                ax.set_ylim(min_y, max_y+1)
                ax.yaxis.set_visible(True)
                ax.tick_params(axis='y', which='both', labelleft=True)
    else:
        # It's a 1D array-like, find the global maximum y-value
        max_y = float('-inf')
        min_y = float('inf')
        for ax in axs:
            y_max = ax.get_ylim()[1]
            y_min = ax.get_ylim()[0]
            max_y = max(max_y, y_max+1)
            min_y = min(min_y, y_min)
        # Set the same y-axis range for all subplots
        for ax in axs:
            ax.set_ylim(min_y, max_y)
            ax.yaxis.set_visible(True)
            ax.tick_params(axis='y', which='both', labelleft=True)

    return

def get_area_df(df, area, nomenclature):
    if nomenclature == 'area_acronym_custom':
        return df[df['area_acronym_custom'] == area]
    elif nomenclature == 'target_region':
        return allen.apply_target_region_filters(df, area)
    else:
        print('Unknown area nomenclature:', nomenclature)
        return None


def get_n_numbers(df):
    """
    Get and generate titls with N numbers per reward groups.
    :param df: pd.DataFrame containing neural units
    :return:
    """

    #Check if 1 or 2 reward_groups
    reward_groups = df['reward_group'].unique()

    if len(reward_groups)==1:
        #Check which group is present
        if reward_groups[0] == 1:
            n_mice_rplus = df[df.reward_group==1].mouse_id.nunique() if 'mouse_id' in df.columns else None
            n_units_rplus = df[df.reward_group==1].unit_id.nunique() if 'unit_id' in df.columns else None
            n_mice_rminus = None
            n_units_rminus = None
        elif reward_groups[0] == 0:
            n_mice_rplus = None
            n_units_rplus = None
            n_mice_rminus = df[df.reward_group==0].mouse_id.nunique() if 'mouse_id' in df.columns else None
            n_units_rminus = df[df.reward_group==0].unit_id.nunique() if 'unit_id' in df.columns else None

    elif len(reward_groups) == 2:
        n_mice_rplus = df[df.reward_group==1].mouse_id.nunique() if 'mouse_id' in df.columns else None
        n_units_rplus = df[df.reward_group==1].unit_id.nunique() if 'unit_id' in df.columns else None
        n_mice_rminus = df[df.reward_group==0].mouse_id.nunique() if 'mouse_id' in df.columns else None
        n_units_rminus = df[df.reward_group==0].unit_id.nunique() if 'unit_id' in df.columns else None
    else:
        print('Unsupported number of reward groups:', len(reward_groups))
        n_mice_rplus = None
        n_units_rplus = None
        n_mice_rminus = None
        n_units_rminus = None

    return n_mice_rplus, n_units_rplus, n_mice_rminus, n_units_rminus

def get_numbers_title(df, area):
    """
    Get and generate titls with N numbers per reward groups.
    :param df: pd.DataFrame containing neural units
    :return:
    """

    #Check if 1 or 2 reward_groups
    reward_groups = df['reward_group'].unique()

    if len(reward_groups)==1:
        #Check which group is present
        if reward_groups[0] == 1:
            n_mice_rplus = df.mouse_id.nunique()
            n_units_rplus = df.unit_id.nunique()
            title_rplus = f'PETH {area} \n ({n_units_rplus} units, {n_mice_rplus} mice)'
            title_rminus = f'No R-'
        elif reward_groups[0] == 0:
            n_mice_rminus = df.mouse_id.nunique()
            n_units_rminus = df.unit_id.nunique()
            title_rplus = f'No R+'
            title_rminus = f'PETH {area} \n ({n_units_rminus} units, {n_mice_rminus} mice)'

    elif len(reward_groups) == 2:
        n_mice_rplus = df[df.reward_group==1].mouse_id.nunique()
        n_units_rplus = df[df.reward_group==1].unit_id.nunique()
        n_mice_rminus = df[df.reward_group==0].mouse_id.nunique()
        n_units_rminus = df[df.reward_group==0].unit_id.nunique()
        title_rplus = f'PETH {area} \n ({n_units_rplus} units, {n_mice_rplus} mice)'
        title_rminus = f'PETH {area} \n ({n_units_rminus} units, {n_mice_rminus} mice)'
    else:
        print('Unsupported number of reward groups:', len(reward_groups))
        title_rplus = 'PETH'
        title_rminus = 'PETH'
    return title_rplus, title_rminus

def get_default_trial_type_color_dict_palette():
    """Returns a dictionary with default trial type / reward_group colors."""
    default_palette = {
            'whisker_trial': ['crimson', 'forestgreen'],
            'auditory_trial': [lighten_color('mediumblue', 0.7), 'mediumblue'],
            'no_stim_trial': [lighten_color('k', 0.7), 'k']
        }
    return default_palette

def get_trial_type_color_palettes_from_hue(trial_type, hue):
    """
    Returns a dictionary with reward-group based colors based on the specified hue.
    :param trial_type: whisker_trial, auditory_trial, or no_stim_trial
    :param hue: a string specifying the hue, e.g. 'block_type_perf'
    :return: A dictionary with reward_group as keys and a dictionary of colors for hue values
    """
    default_palette = get_default_trial_type_color_dict_palette()
    trial_colors_reward_group = default_palette[trial_type]
    if hue == 'lick_flag':
        palette_dict = {0: [lighten_color(trial_colors_reward_group[0], 0.5), # no lick, unrewarded gorup
                            lighten_color(trial_colors_reward_group[1], 0.5)], # no lick, rewarded group
                        1: [lighten_color(trial_colors_reward_group[0], 1), # lick, unrewarded group
                             lighten_color(trial_colors_reward_group[1], 1)]} # lick, rewarded group
    elif hue == 'block_perf_type':
        if trial_type=='whisker_trial':
            palette_dict = {0: {'naive': adjust_lightness(trial_colors_reward_group[0], 1.5),
                                'expert': adjust_lightness(trial_colors_reward_group[0], 0.7)},
                            1: {'naive': adjust_lightness(trial_colors_reward_group[1], 1.5),
                                'expert': adjust_lightness(trial_colors_reward_group[1], 0.7)}}
        else:
            palette_dict = {0: {'naive': adjust_lightness(trial_colors_reward_group[0], 1.5),
                                'expert': adjust_lightness(trial_colors_reward_group[0], 0.7)},
                            1: {'naive': adjust_lightness(trial_colors_reward_group[0], 1.5),
                                'expert': adjust_lightness(trial_colors_reward_group[0], 0.7)}}
    else:
        print('Palette not implemented for hue:', hue)
        palette_dict = None

    return palette_dict

def switch_color_in_palette(palette, color_a='forestgreen', color_b='crimson'):
    """
    Replace all occurrences of color_a with color_b in the given palette.
    :param palette:
    :param color_a:
    :param color_b:
    :return:
    """
    palette_copy = copy.deepcopy(palette)

    # Use the recursive replacement function
    def _replace(p):
        if isinstance(p, dict):
            return {k: _replace(v) for k, v in p.items()}
        elif isinstance(p, list):
            return [_replace(v) for v in p]
        elif isinstance(p, tuple):
            return tuple(_replace(v) for v in p)
        elif isinstance(p, str) and p.lower() == color_a.lower():
            return color_b
        else:
            return p

    return _replace(palette_copy)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def pval_to_color_size(pval, cmap_name="white_orange",
                       min_size=20, max_size=200,
                       vmin=1e-6, vmax=1.0):
    """
    Convert a p-value into:
      - a color on a white→orange colormap
      - a marker size scaled on a log scale

    Parameters
    ----------
    pval : float
        p-value (0 < p <= 1)
    cmap_name : str
        Name for internally created cmap
    min_size : int
        Marker size for large p-values
    max_size : int
        Marker size for small p-values
    vmin : float
        Min p-value for scaling (log)
    vmax : float
        Max p-value for scaling (log)

    Returns
    -------
    color : RGBA tuple
    size  : float
    """

    # make sure pval is within bounds
    pval = np.clip(pval, vmin, vmax)

    # --- log-transform pval ---
    # log p will run from log(vmin) → log(vmax)
    log_p = (np.log(pval) - np.log(vmin)) / (np.log(vmax) - np.log(vmin))
    log_p = np.clip(log_p, 0, 1)

    # --- create white→orange colormap ---
    cmap = LinearSegmentedColormap.from_list(
        cmap_name,
        ["white", "orange"]
    )

    # get color
    color = cmap(log_p)

    # --- scale dot size ---
    size = min_size + (1 - log_p) * (max_size - min_size)

    return color, size



def unify_row_ylims(fig):
    """
    Unify y-limits for each row in a grid of subplots.
    Assumes subplots are arranged in a regular matrix (n_rows x n_cols).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure containing subplots.
    """
    # Grab all axes in draw order, then reshape by rows
    axes = [ax for ax in fig.axes if ax.get_visible()]

    # Determine grid size by inspecting axes positions
    # (Matplotlib stores them in row-major order)
    # Here we infer rows by unique y-coordinates of axes
    yvals = sorted(set([round(ax.get_position().y0, 5) for ax in axes]), reverse=True)
    rows = []
    for y in yvals:
        row_axes = [ax for ax in axes if round(ax.get_position().y0, 5) == y]
        # Sort left→right
        row_axes.sort(key=lambda a: a.get_position().x0)
        rows.append(row_axes)

    # For each row, unify y-limits
    for row_axes in rows:
        ymins = [ax.get_ylim()[0] for ax in row_axes]
        ymaxs = [ax.get_ylim()[1] for ax in row_axes]
        ymin = min(ymins)
        ymax = max(ymaxs)
        for ax in row_axes:
            ax.set_ylim(ymin, ymax)
