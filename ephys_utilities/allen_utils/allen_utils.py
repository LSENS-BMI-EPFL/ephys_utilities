#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: unit_analysis
@file: allen_utils.py
@time: 1/21/2025 3:36 PM
"""

# Imports
import os
import socket
import json
import pathlib
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.spatial import cKDTree

# Get Allen atlas colors
hostname = socket.gethostname()
if 'haas' in hostname:
    PATH_TO_ATLAS = r'/mnt/lsens-analysis/Axel_Bisi/Anatomy/allen_mouse_bluebrain_barrels_10um_v1.0'
else:
    PATH_TO_ATLAS = r'M:\analysis\Axel_Bisi\Anatomy\allen_mouse_bluebrain_barrels_10um_v1.0'

def get_cortical_areas():
    """
    Retrieve a list of cortical(++) area acronyms.
    :return: List of cortical area acronyms
    """
    return [
        'FRP', 'MOp',
        'MOs', 'MOs-a', 'MOs-m', 'MOs-p',
        'SSp-bfd', 'SSp-m', 'SSp-ul', 'SSp-ll', 'SSp-un', 'SSp-n', 'SSp-tr',
        'SSs', 'AUD', 'AUDp', 'AUDd', 'AUDv',
        'ACA', 'ACAv', 'ACAd',
        'VIS','VISa', 'VISp', 'VISam', 'VISl', 'VISpm', 'VISrl', 'VISal',
        'PL', 'ILA', 'IL',
        'RSP', 'RSPv', 'RSPd','RSPagl',
        'SCm', 'SCsg', 'SCzo', 'SCiw', 'SCop', 'SCs',
        'ORB', 'ORBm', 'ORBl', 'ORBvl',
        'AId', 'AIv', 'AIp',
        'FRP', 'VISC'
    ]

def get_allen_color_dict():
    """
    Get Allen atlas colors formatted as dictionary of RGB arrays.
    :return:
    """

    with open(os.path.join(PATH_TO_ATLAS, 'structures.json')) as f:
        structures_dict = json.load(f)

    area_colors = {area['acronym']: np.array(area['rgb_triplet']) / 255 for area in structures_dict}
    return area_colors

def get_excluded_areas():
    """
    Retrieve a list of excluded area acronyms.
    :return: List of excluded area acronyms
    """
    excluded_areas = ["alv", "amc", "aco", "act", "arb", "ar", "bic", "bsc", "c", "cpd", "cbc", "cbp", "cbf", "AQ",
                      "epsc", "mfbc", "cett", "chpl", "cing", "cVIIIn", "fx", "stc", "cc", "fa", "ccb", "ee", "fp",
                      "ccs", "cst", "cm", "tspc", "cuf", "tspd", "dtd", "das", "dc", "df", "dhc", "lotd", "drt", "sctd",
                      "mfbse", "ec", "em", "eps", "VIIn", "fr", "fiber tracts", "fi", "fxs", "V4", "ccg", "gVIIn",
                      "hbc", "hc", "mfsbshy", "icp", "cic", "int", "lfbs", "ll", "lot", "lotg", "V4r", "mp",
                      "mfbsma", "mtg", "mtt", "mct", "mfb", "mfbs", "ml", "mlf", "mcp", "moV", "nst", "IIIn", "In",
                      "onl", "och", "IIn", "or", "opt", "fxpo", "pc", "pm", "py", "pyd", "root", "rust", "sV", "ts",
                      "sptV", "sm", "st", "SEZ", "scp", "dscp", "csc", "scwm", "sup", "tsp", "lfbst", "V3", "tb", "Vn",
                      "IVn", "uf", "Xn", "vhc", "sctv", "vtd", "VS", "vVIIIn", "VIIIn", "von",
                      'VL', 'I', 'VS'
                      'nan',
                      ]
    #ignored_areas = ['ND', 'VISC', 'SSp-ll', 'GU', 'HA', 'HY', 'MA', 'NOT', 'P'] #areas to ignore because too few neurons
    return excluded_areas



def contains_layer(region):
    """Check if a region name contains a layer number excluding CA1, CA2, and CA3."""
    if region in ['CA1', 'CA2', 'CA3']:
        return False
    else:
        return bool(re.search(r'\d+[a-zA-Z]*', str(region)))  # e.g., "6a", "6b"

def generalize_region(region):
    """Generalize region names based on predefined rules."""
    region_map = {
        "ACA": "ACA",
        "AD":"ATN",
        "AI": "AI",
        "Ai": "AI",
        "AMd":"ATN",
        "AMv":"ATN",
        "AM":"ATN",
        "AON":"OLF",
        "APN":"APN",
        "AUD": "AUD",
        "AV":"ATN",
        "BLAp":"BLA",
        "BLAv":"BLA",
        "BLAa":"BLA",
        "BST":"PAL",
        "CEA": "CEA",
        "CL": "ILM",
        "CM":"ILM",
        "CTXsp": "CTXsp",
        "DG": "DG",
        "Eth": "VP",
        "EPd": "EP",
        "EPv": "EP",
        'HPF':'HPF',
        "HY":"HY",
        "IG":"HPF",
        "IGL":"LGN",
        "IntG":"LGN",
        "INC":"PAG",
        "LD":"ATN",
        "LGd": "LGN",
        "LGv": "LGN",
        "LP": "LGN",
        "LH":"HA",
        "LS": "LS",
        "LT": "MB",
        "MD":"MED",
        "MH":"HA",
        "MS":"MSC",
        "MGm":"MGN",
        "MGv":"MGN",
        "MGd":"MGN",
        "MMd":"HY",
        "MMme":"HY",
        "MT":"MB",
        "NLL":"Pons",
        "NPC":"MB",
        "ORB": "ORB",
        "PAL": "PAL",
        "PCN": "ILM",
        "PF": "ILM",
        "PIL":"ILM", # confirm, also, use DORpm vs DORsm ? PIL-PP in lit.
        "PIR":"OLF",
        "POL":"LAT",
        "POST":"SUB",
        "PoT":"VP",
        "PO":"VP",
        "PP":"ILM",# PIL-PP in lit., ILMN: intralaminar nuclei
        "PPN":"MB",
        "PR":"MED",
        "PRC":"PAG",
        "PRNr":"Pons",
        "PVT": "MTN",
        "PV": "MTN",
        "RE":"MTN",
        "RPF":"MB",
        "RR":"MB",
        "RSP": "RSP",
        "SAG":"MB",
        "SGN":"LAT",
        "SPFm":"VP",
        "SPFp":"VP",
        "SPF":"VP",
        "SI":"PAL",
        "SMT":"MED",
        #"STR": "STR",
        "SUM":"HY",
        "TEa": "TEa",
        "TRS":"PAL",
        "VPL": "VP",
        "VPM": "VP",
        "VAL":"VM",
        "VM":"VM",
        "Xi":"MTN"
    }
    for key in region_map:
        try:
            if region.startswith(key):
                return region_map[key]
        except AttributeError as err:
            print(err, region)
    if region.startswith("LA"):
        return "LAT" if region.startswith("LAT") else "LA"

    if region.startswith("MY"): #medulla
        return "MY"

    if region.startswith("SC"):
        return "SCm" if region in ["SCdg", "SCdw", "SCig", "SCiw"] else "SCs"

    if region.startswith("SSp-bfd"):
        return "SSp-bfd"

    #if region.startswith("VIS"):
    #    if region=='VISC':
    #        return 'VISC'
    #    else:
    #        return 'VIS'

    return region  # Default: no change


def handle_ssp_bfd(region):
    """Special case: handle SSp-bfd barrels (e.g., "SSp-bfd-C4"/"SSp-bfd-Gamma" -> "SSp-bfd")."""
    return re.sub(r'SSp-bfd-[A-Za-z0-9]+', 'SSp-bfd', region) if "SSp-bfd" in region else region

def handle_ssp_whisker(region, row=None):
    """Special case: unify somatosensory orofacial-related subregions to 'SSp-oro'."""
    ssp_orofacial = ['SSp-bfd', 'SSs'] #more evidence that SSp-unassigned is closer to orofacial (bfd) than body based on connectivity studies
    if region in ssp_orofacial and row['target_region'] in ['wS1', 'wS2']:
        return 'SSp-w'
    else:
        return region

def handle_ssp_oro(region, row=None):
    """Special case: unify somatosensory orofacial-related subregions to 'SSp-oro'."""
    ssp_orofacial = ['SSp-m', 'SSp-n', 'SSp-un'] #more evidence that SSp-unassigned is closer to orofacial (bfd) than body based on connectivity studies
    if region in ssp_orofacial:
        return 'SSp-oro'
    else:
        return region

def handle_ssp_body(region, row=None):
    """Special case: unify somatosensory body-related subregions to 'SSp-body'."""
    ssp_body = ['SSp-ul', 'SSp-ll', 'SSp-tr']
    if region in ssp_body:
        return 'SSp-body'
    else:
        return region

def handle_mpfc(region, row=None):
    """Special case: unify medial PFC subregions to 'mPFC'."""
    mpfc_areas = {"PL", "ILA", "IL", "ACA", "ACAd", "ACAv"}
    if region in mpfc_areas:
        return 'mPFC'
    else:
        return region

def handle_ppc(region, row=None):
    """Special case: unify PPC subregions to 'PPC'."""
    pcc_areas = ['PTLp', 'VISa', 'VISrl', 'SSp-tr', 'SSp-un', 'SSp-bfd']
    if region in pcc_areas and row['target_region']=='PPC':
        return 'PPC'
    else:
        return region

def handle_v1(region, row=None):
    """Special case: unify PPC subregions to 'PPC'."""
    v1_areas = ['VIS', 'VISp', 'VISal', 'VISl', 'VISpm', 'VISam', 'VISpl', 'VISli', 'VISpor', 'VISrl']
    if region in v1_areas:
        return 'V1'
    else:
        return region

def handle_lsf(region, row=None):
    """Special case: unify LS and SF subregions to 'LSX'."""
    lat_septal_complex = ['SF', 'LS', 'SH']
    if region in lat_septal_complex:
        return 'LSX'
    else:
        return region

def handle_cortical_subplate(region, row=None):
    """Special case: unify LS and SF subregions to 'LSX'."""
    ctx_suplate = ['CLA', 'EP', 'LA', 'BLA', 'BMA', 'PA']
    if region in ctx_suplate:
        return 'CTXsp'
    else:
        return region

def handle_hypothalamus(region, row=None):
    """Special case: handle hypothalamic areas."""
    hypo_areas = ['ZI']
    if region in hypo_areas:
        return 'HY'
    else:
        return region


def apply_context_handlers(region, row):
    """
    Apply row-context-dependent overrides on top of the lexical `region` label
    produced by simplify_area(). Order matters: broader regroupings first,
    the most specific (target_region-gated) override last so it wins.
    """
    region = handle_ssp_oro(region, row)
    region = handle_ssp_body(region, row)
    region = handle_lsf(region, row)
    region = handle_hypothalamus(region, row)
    region = handle_mpfc(region, row)
    region = handle_ppc(region, row)
    region = handle_v1(region, row)
    region = handle_cortical_subplate(region, row)
    return region


def simplify_area(ccf_acronym, ccf_parent_acronym):
    """Decide and return the simplified area name."""
    base_region = ccf_acronym if not contains_layer(ccf_acronym) or ccf_acronym in ['CA1', 'CA2',
                                                                                    'CA3'] else ccf_parent_acronym
    return handle_ssp_bfd(generalize_region(base_region))


def create_area_custom_column(df):
    """
    Using helps, create a new column 'area_acronym_custom' based on 'ccf_acronym' and 'ccf_parent_acronym'.
    - If ccf_acronym contains a layer number, use ccf_parent_acronym unless the region is CA1, CA2, or CA3.
    - Simplifies visual areas (e.g., VISpm, VISa, VISal) to "VIS".
    - Simplifies auditory areas (e.g., AUDd, AUDpo, AUDp, AUDv) to "AUD".
    - Simplifies ORBv to "ORB".
    - Handles specific cases like SSp-bfd barrel indications (e.g., SSp-bfd-C4 -> SSp-bfd).

    :param df: A pandas DataFrame containing 'ccf_acronym' and 'ccf_parent_acronym' columns.
    :return: DataFrame with a new column 'area_acronym_custom'.
    """
    def simplify_per_nomenclature(row):
        # Prefer ephys-align atlas fields if they exist and are not NaN
        if ('ccf_atlas_acronym' in row) and ('ccf_atlas_parent_acronym' in row):
            region = simplify_area(row['ccf_atlas_acronym'], row.get('ccf_atlas_parent_acronym', None))
        else:
            region = simplify_area(row['ccf_acronym'], row.get('ccf_parent_acronym', None))

        return apply_context_handlers(region, row)

    df['area_acronym_custom'] = df.apply(simplify_per_nomenclature, axis=1)
    return df

def extract_layer_info_original(ccf_acronym):
    """Extract and return layer information from a regionF name."""
    match = re.search(r'(\d+[a-zA-Z]*)', ccf_acronym)
    if match:
        layer = match.group(0)
        return "2/3" if layer in ["2", "3"] else layer
    return None

def extract_layer_info(ccf_acronym):
    """
    Robustly extract layer info from a region name like:
    'MOp2/3', 'VISp2', 'S1-2', ...
    Returns strings like '2/3', '4', '5a', or None if not found / input is NaN.
    """
    # handle None, NaN, non-string inputs
    if pd.isna(ccf_acronym):
        return None

    s = str(ccf_acronym)
    _layer_re = re.compile(r'(?<!\w)([1-6](?:/[23])?(?:[a-zA-Z])?)(?!\w)')
    m = _layer_re.search(s)
    if not m:
        return None

    layer = m.group(1)
    # map plain '2' to '2/3' per your rule
    if layer == '2':
        return '2/3'
    return layer


def create_layer_number_column(df):
    """Create a column 'layer_number' that only extracts layer information."""
    if 'ccf_atlas_acronym' in df.columns:
        col='ccf_atlas_acronym'
    else:
        col='ccf_acronym'
    df['layer_number'] = df[col].apply(extract_layer_info_original) #Check which is nans
    return df


def create_ccf_acronym_no_layer_column(df):
    """Create a column 'ccf_acronym_no_layer' that keeps the original ccf_acronym unless it contains a layer number, in which case it uses the parent acronym."""
    if 'ccf_atlas_acronym' in df.columns and 'ccf_atlas_parent_acronym' in df.columns:
        col='ccf_atlas_acronym'
        col_parent='ccf_atlas_parent_acronym'
        df['ccf_atlas_acronym_no_layer'] = df.apply(
            lambda row: handle_ssp_bfd(row[col_parent]) if contains_layer(row[col]) else row[col], axis=1)

    elif 'ccf_acronym' in df.columns and 'ccf_parent_acronym' in df.columns:
        col='ccf_acronym'
        col_parent='ccf_parent_acronym'
        df['ccf_acronym_no_layer'] = df.apply(
            lambda row: handle_ssp_bfd(row[col_parent]) if contains_layer(row[col]) else row[col], axis=1)
    else:
        print('Warning: cannot estimate ccf column without layer.')
    return df

def get_target_region_order():
    """
    Get a set order of target regions for plotting.
    :return:
    """
    return ['wS1', 'wS2', 'A1', 'PPC', 'DLS', 'wM1', 'wM2', 'tjM1', 'ALM', 'OFC', 'SC']

def get_area_acronym_custom_order():
    """
    Get the order of brain areas for plotting.
    """
    area_order = ['MOp', 'MOs', 'MO-tjM1', 'MO-ALM', 'MO-wM1', 'MO-wM2', 'mPFC', 'FRP', 'ACA', 'PL', 'ORB', 'AI',
                  'SSp-bfd', 'SSs', 'SSp-m', 'SSp-n', 'SSp-ul', 'SSp-ll', 'SSp-tr', 'SSp-un', 'VISC', 'GU', 'SSp-oro', 'SSp-body',
                  'AUD', 'TEa', 'PPC', 'V1', 'VIS', 'VISa', 'VISp', 'VISam', 'VISl', 'VISpm', 'VISrl', 'VISal',
                  'CLA', 'EP', 'CTXsp',
                  'CA1', 'CA2', 'CA3', 'DG', 'HPF', 'SUB', 'PROs',
                  'CP', 'DMS', 'DLS', 'TS', 'STR', 'ACB', 'VS', 'FS', 'LS', 'SF', 'GPe', 'GPi', 'PAL', 'MS', 'LSX',
                  'VT', 'TH', 'VPL', 'VPM', 'VP', 'LD', 'RT', 'PO', 'LGN', 'LP', 'ATN', 'LAT', 'MGN', 'MED', 'MTN', 'ILM', 'HA', 'CL', 'ILN',
                  'RSP',
                  'SCs', 'SCm', 'MB', 'VTA', 'MRN', 'PAG', 'RN', 'SNr', 'APN', 'OP',
                  'Pons', 'MY',
                  'AON', 'OLF', 'PIR',
                  'BLA', 'LA', 'CEA','HY', 'ZI']
    return area_order

def get_custom_area_groups():
    """
    Get the custom area groups for plotting.
    """

    area_groups = {
        #'Motor and frontal areas': ['MOp', 'MOs', 'MO-tjM1', 'MO-ALM', 'MO-wM1', 'MO-wM2', 'mPFC', 'FRP', 'ACA', 'PL', 'ORB'],
        'Motor areas': ['MOp', 'MOs', 'MO-tjM1', 'MO-ALM', 'MO-wM1', 'MO-wM2'],
        'Frontal areas': ['mPFC', 'FRP', 'ACA', 'PL', 'ORB'],
        #'Somatosensory areas': ['SSp-bfd', 'SSs', 'SSp-m', 'SSp-n', 'SSp-ul', 'SSp-ll', 'SSp-tr', 'SSp-un', 'VISC', 'GU', 'AI'],
        'Somatosensory-orofacial': ['SSp-m', 'SSp-n', 'SSp-un', 'SSp-oro'],
        'Somatosensory-body': ['SSp-ul', 'SSp-ll', 'SSp-tr', 'SSp-body'],
        'Somatosensory-whisker': ['SSp-bfd', 'SSs'],
        'Auditory areas': ['AUD', 'TEa'],
        'Retrosplenial areas': ['RSP'],
        'Posterior parietal areas': ['PPC', 'PTLp', 'VISa', 'VISrl', 'SSp-tr'],
        'Visual areas': ['V1', 'VIS', 'VISp', 'VISal', 'VISl', 'VISpm', 'VISam', 'VISpl', 'VISli', 'VISpor'],
        'Insular areas': ['AI','GU','VISC'],
        'Hippocampus': ['CA1', 'CA2', 'CA3', 'DG', 'HPF', 'SUB', 'ProS', 'POST', 'IG'],
        'Cortical subplate': ['CTXsp','CLA', 'EP', 'LA', 'BLA', 'BMA', 'PA'],
        'Striatum': ['CP', 'DMS', 'DLS', 'TS', 'STR', 'VS', 'ACB', 'FS'],
        'Pallidum': ['GPe', 'GPi', 'PAL', 'MS', 'MSC', 'MA'],
        'Lateral septal complex': ['LS', 'SF', 'LSX', 'SH'], # striatum by development but not at all functionally
        'Thalamus': ['TH', 'VPL', 'VPM', 'VP', 'VT', 'LD', 'RT', 'PO', 'PT', 'LGN', 'LP', 'ATN', 'LAT', 'MGN', 'MED', 'MTN', 'MT', 'ILM', 'HA', 'CL', 'ILN', 'IAD'],
        'Midbrain': ['SCs', 'SCm', 'MB', 'VTA', 'MRN', 'PAG', 'RN', 'SNr', 'APN', 'OP', 'NOT', 'ND'],
        'Pons and medulla': ['Pons', 'MY', 'P'],
        'Olfactory areas': ['AON', 'OLF', 'PIR'],
        'Amygdala and hypothalamus': ['CEA','MEA', 'HY', 'ZI', 'PMd', 'Mmd']
    }
    return area_groups

def get_area_group_custom_order():
    return get_custom_area_groups().keys()

def get_custom_area_groups_from_name():
    """
    Get custom area groups from a predefined dictionary based on area names.

    :return: Dictionary of area groups with area acronyms as keys
    """
    area_groups = get_custom_area_groups()
    area_groups_from_name = {area: group for group, areas in area_groups.items() for area in areas}
    return area_groups_from_name

def get_custom_area_groups_colors():
    """Get custom area group colors for plotting, here Allen colors."""
    area_group_colors = {
        'Motor and frontal areas': '#1f9d5a',
        'Motor areas': '#1f9d5a',
        'Frontal areas': '#8333f2',
        'Somatosensory areas': '#188064',
        'Somatosensory-orofacial': '#a66829',
        'Somatosensory-body': '#e0c92f',
        'Somatosensory-whisker': '#f7b519',
        'Auditory areas': '#2c2cdb',
        'Retrosplenial areas': '#1aa698',
        'Posterior parietal areas': '#009fac',
        'Visual areas': '#08858c',
        'Insular areas':'#219866',
        'Hippocampus': '#7ed04b',
        'Cortical subplate': '#69b366',
        'Striatum': '#54c2ff',
        'Pallidum': '#7a94d6',
        'Lateral septal complex': '#90e7ed',
        'Thalamus': '#ff8084',
        'Midbrain': '#ff90ff',
        'Pons and medulla': '#fcae72',
        'Olfactory areas': '#7bc9ac',
        'Amygdala and hypothalamus': '#e64438'
    }
    return area_group_colors

def get_custom_area_color_per_group():
    """
    Using custom area group above, return a dictionary and list of single-area colors.

    """

    # Make cmap with as many colors as number of area groups
    group_color_palette = get_custom_area_groups_colors() # colors from Allen atlas
    area_groups = get_custom_area_groups() # potentially not all groups are present
    default_color = '#888888'
    colors = [group_color_palette.get(group, default_color) for group in area_groups]
    #colors = [group_color_palette[i % len(group_color_palette)] for i in range(len(area_groups))] # Note: keep in case

    # Create a dictionary mapping each single area to its group color
    area_color_dict = {}
    for (group_name, areas), color in zip(area_groups.items(), colors):
        for area in areas:
            area_color_dict[area] = color

    # Make it also a list
    area_color_list = list(area_color_dict.values())
    return area_color_dict, area_color_list

def get_color_per_area(area_list):
    """
    Using the above functions and for a given list of areas, make a palette for seaborn to associate each area to its color.
    :param area_list:
    :return:
    """
    area_color_dict, _ = get_custom_area_color_per_group()
    color_palette = {area: area_color_dict.get(area, '#000000') for area in area_list}  # default to black if not found
    print('Areas not found:', [area for area in area_list if area not in area_color_dict])
    return color_palette

def create_legend_figure(color_dict, rectangles=True, title='Legend'):
    """
    Create a legend figure for the areas with their corresponding colors.
    :param color_dict: A dictionary mapping strings to colors (e.g., {"Label": "#ff0000"}).
    :param rectangles:  If True, use colored rectangles; otherwise, color the text directly.
    :param title: Title for the legend figure.
    :return: a figure containing the legend.
    """

    #color_dict = get_custom_area_groups_colors()

    fig, ax = plt.subplots(figsize=(2, 3), dpi=300)
    ax.axis('off')  # Turn off the axes for a cleaner look

    legend_elements = [Patch(facecolor=color, label=label) for label, color in color_dict.items()]
    ax.legend(handles=legend_elements, loc='upper left', frameon=False, title=title)
    fig.tight_layout()


    return fig

def get_isocortex_mapping():
    """ For RS/FS distinctions."""
    isocortex_groups = {
        'Isocortex': [
            'MOp', 'MOs', 'MO-tjM1', 'MO-ALM', 'MO-wM1', 'MO-wM2',
            'mPFC', 'FRP', 'ACA', 'PL', 'ORB',
            'SSp-bfd', 'SSs', 'SSp-m', 'SSp-n', 'SSp-ul', 'SSp-ll',
            'SSp-tr', 'SSp-un', 'VISC', 'GU', 'AI',
            'AUD', 'TEa',
            'RSP',
            'PPC', 'VIS', 'VISa', 'VISp', 'VISam', 'VISl',
            'VISpm', 'VISrl', 'VISal'
        ],
        'Non-isocortex': [
            'CA1', 'CA2', 'CA3', 'DG', 'HPF',
            'CP', 'DMS', 'DLS', 'TS', 'STR', 'VS', 'ACB', 'FS', 'LS',
            'SF', 'GPe', 'GPi', 'PAL', 'MS',
            'TH', 'VPL', 'VPM', 'VP', 'LD', 'RT', 'PO', 'LGN', 'LP',
            'ATN', 'LAT', 'MGN', 'MED', 'MTN', 'ILM', 'HA', 'CL',
            'SCs', 'SCm', 'MB', 'VTA', 'MRN', 'PAG', 'RN', 'SNr', 'APN',
            'Pons', 'MY',
            'AON', 'OLF', 'PIR',
            'BLA', 'BLAa', 'LA', 'CEA', 'MEA', 'HY', 'ZI'
        ]
    }
    return isocortex_groups

def apply_target_region_filters(peth_table, area):
    """
    Apply specific area filters based on the area name.

    :param peth_area: Subset of PETH table for a specific area
    :param area: Specifically-named brain area
    :return: Filtered PETH table
    """
    specific_filters = {
        'wS1': ['SSp-bfd'],
        'wM1': ['MOp', 'MOs', 'MOs-a', 'MOs-m','MOs-p','SSp-ll'],
        'wS2': ['SSs', 'SSp-bfd'],
        'wM2': ['MOp', 'MOs', 'MOs-a', 'MOs-m', 'MOs-p', 'SSp-ll'],
        'mPFC': ['PL', 'ILA', 'ACA', 'ACAd', 'ACAv'],
        'tjM1': ['MOp', 'MOs', 'SSp-m', 'MOs-a', 'MOs-m', 'MOs-p'],
        'A1': ['AUD', 'AUDd', 'AUDp', 'AUDv', 'AUDpo'],
        'DLS': ['STRd', 'CP', 'DLS'],
        'DMS': ['STRd', 'CP', 'DMS'],
        'VS': ['STRv', 'STR', 'ACB', 'CP', 'VS'],
        'TS': ['STRd', 'STR', 'CP', 'TS'],
        'SC': ['SC', 'SCs', 'SCiw', 'SCop', 'SCm', 'SCzo', 'SCsg'],
        'OFC': ['ORB', 'ORBm', 'ORBl', 'ORBvl'],
        'ALM': ['MOp', 'MOs', 'MOS-a', 'MOs-m', 'MOs-p'],
        'PPC': ['VIS', 'VISa', 'VISrl', 'SSp-tr', 'SSp-un', 'SSp-bfd'],
        'V1': ['VIS', 'VISp', 'VISl', 'VISpm', 'VISpl', 'VISli', 'VISpor'],
    }

    if area in specific_filters.keys():
        # Keep only areas specified in filter and actually targeted e.g. SSp-bfd
        peth_area = peth_table[(peth_table['area_acronym_custom'].isin(specific_filters[area]))
                                & (peth_table['target_region'] == area)]
    else:
        print(f'{area} is not part of the target area dict. Skipping.')
        peth_area = None

    return peth_area

def subset_data_for_area(data_df, area, params): #TODO: unused, delete or refactor
    """
    Subset the PETH table for a specific area based on the nomenclature.

    :param data_df: PETH table with all mice data
    :param area: Specific brain area
    :param params: Plotting parameters
    :return: Subset of PETH table for the specified area or None if no data
    """
    area_nomenclature = params.get('area_nomenclature', 'ccf_parent_acronym')
    if area_nomenclature == 'area_acronym':
        peth_area = data_df[data_df['area_acronym'] == area]
        if peth_area.empty:
            return None
    elif area_nomenclature == 'target_region':
        peth_area = data_df[data_df['target_region'] == area]
        # Further filter cortical areas
        ctx_areas = get_cortical_areas()
        peth_area = peth_area[peth_area['ccf_parent_acronym'].isin(ctx_areas)]
        # Apply specific selections
        peth_area = apply_target_region_filters(peth_area, area)
    elif area_nomenclature in ['ccf_parent_acronym', 'ccf_acronym']:
        peth_area = data_df[data_df['area_acronym'] == area]
        if peth_area.empty:
            return None
    else:
        print(f'Unknown area nomenclature: {area_nomenclature}')
        return None

    return peth_area

def compute_physical_coordinates_from_df(df, target_coords):
    """
    Compute approximate AP, ML, DV coordinates from entry points and trajectory angles in physical space for each recording site.
    :param df: pd.DataFrame with columns:
    :param target_coords: dict with target names as keys and (AP_entry, ML_entry) tuples as values.
    :return:
    """
    def _compute_row(row):
        # Get entry point
        AP_entry, ML_entry = target_coords[row['target']]
        depth = row['depth']
        az = np.deg2rad(row['azimuth'])
        el = np.deg2rad(row['elevation'])

        # Direction cosines
        dAP = np.cos(el) * np.sin(az)
        dML = np.cos(el) * np.cos(az)
        dDV = np.sin(el)

        # Compute coordinates
        AP = AP_entry + depth * dAP
        ML = ML_entry + depth * dML
        DV = depth * dDV

        return pd.Series({'ap_sample': AP, 'ml_sample': ML, 'dv_sample': DV})

    coords = df.apply(_compute_row, axis=1)
    return pd.concat([df, coords], axis=1)


def create_bregma_centric_coords_from_ccf(df):
    """
    Convert CCF coordinates in BrainGlobe space into bregma-centric coordinates.
    i.e. from (0,0,0)=(A,S,R) anterior top right corner to (0,0,0)=bregma.
    Using IBL bregma estimate:
    https://docs.internationalbrainlab.org/_autosummary/iblatlas.atlas.ALLEN_CCF_LANDMARKS_MLAPDV_UM.html
    :param df: unit_table pd.DataFrame with columns 'ccf_ap', 'ccf_ml', 'ccf_dv', and 'mouse_id'.
    :return:
    """
    # Convert columns to numeric
    df[['ccf_ap', 'ccf_ml', 'ccf_dv']] = df[['ccf_ap', 'ccf_ml', 'ccf_dv']].astype(float)

    ibl_bregma_coords = dict(ap=5400, ml=5739, dv=332)

    # Define conversion functions
    def func_to_ml(row):
        to_ml = lambda x: x - ibl_bregma_coords['ml']
        return to_ml(row['ccf_ml'])


    #ap = (-self.channels['y'] * 1e6) + 5400
    def func_to_ap(row):
        to_ap = lambda x: -x + ibl_bregma_coords['ap'] # AP positive is anterior relative to bregma
        return to_ap(row['ccf_ap'])

    #dv = (abs(self.channels['z'] * 1e6)) + 332
    def func_to_dv(row):
        to_dv = lambda x: x - ibl_bregma_coords['dv']
        return to_dv(row['ccf_dv'])


    # Apply conversions
    df['ap'] = df.apply(func_to_ap, axis=1)
    df['ml'] = df.apply(func_to_ml, axis=1)
    df['dv'] = df.apply(func_to_dv, axis=1)

    return df


def create_areas_subdivisions(df, verbose=False):
    """
    Divide large areas into smaller subdivisions for better visualization.
    For MOs/MOp, use only target-based mapping (no coordinate-based subdivision).

    :param df: pd.DataFrame with columns:
               - 'area_acronym_custom'
               - 'ap', 'ml', 'dv'
               - optionally 'target_region'
    :return: Updated DataFrame with new area assignments.
    """
    if verbose:
        print('Creating area subdivisions...')

    # Parent-child definitions (coordinate-based for CP/STR only)
    parent_child_dict = {
        'CP': ['VS', 'DLS', 'DMS', 'TS'], # VS includes STRv, ACB
        'STRd': ['VS', 'DLS', 'DMS', 'TS'],
        'STRv': ['VS', 'DLS', 'DMS', 'TS'],
        'STR': ['VS', 'DLS', 'DMS', 'TS'],
        'ACB': ['VS', 'DLS', 'DMS', 'TS'],
        'FS': ['VS', 'DLS', 'DMS', 'TS'],
    }

    # Coordinate boundaries for striatum subdivisions
    coord_boundaries = {
        'DMS': {'ap': (-1500, 6000), 'ml': (0, 3000), 'dv': (0, 7000)},
        'DLS': {'ap': (-1500, 500), 'ml': (2400, 6000), 'dv': (0, 7000)},
        'TS': {'ap': (-6000, -1500), 'ml': (0, 6000), 'dv': (0, 7000)},
        'VS': {'ap': (0, 6000), 'ml': (0, 6000), 'dv': (4000, 7000)},
    }

    # Target-region → label mapping for MOs/MOp
    target_region_map = {
        'tjM1': 'MO-tjM1',
        'ALM': 'MO-ALM',
        'OFC': 'MO-ALM',
        'wS2': 'MO-wM1', # TODO: DEAL WITH AB127
        'DLS': 'MO-wM1', #TODO: DEAL WITH AB156
        'wM1': 'MO-wM1',
        'PPC': 'MO-wM1',
        'wM2': 'MO-wM2'
    }

    df = df.copy()
    df['__assigned'] = False

    # --- Handle MOs / MOp separately (target-based only) ---
    if 'target_region' in df.columns:
        mos_mask = df['area_acronym_custom'].isin(['MOs', 'MOp'])
        for target_val, new_label in target_region_map.items():
            mask = mos_mask & (df['target_region'].astype(str) == str(target_val))
            df.loc[mask, 'area_acronym_custom'] = new_label
            df.loc[mask, '__assigned'] = True

        remaining = df[mos_mask & ~df['__assigned']]
        if len(remaining):
            if verbose:
                print(f"Unassigned MOs/MOp: {len(remaining)} (targets: {remaining['target_region'].unique()})")

    # --- Handle other parent areas (like CP) using coordinates ---
    all_striatum_aliases = ['CP', 'STR', 'STRd', 'STRv', 'ACB']

    for parent_area, subdivisions in parent_child_dict.items():
        if verbose:
           print(f'- Subdividing {parent_area} → {subdivisions}')

        for sub_area in subdivisions:
            bounds = coord_boundaries[sub_area]

            # VS comes from all striatal aliases
            if sub_area == 'VS':
                parent_mask = df['area_acronym_custom'].isin(all_striatum_aliases) & (~df['__assigned'])
            else:
                parent_mask = (df['area_acronym_custom'] == parent_area) & (~df['__assigned'])

            ap_mask = df['ap'].between(*bounds['ap'])
            ml_mask = df['ml'].between(*bounds['ml'])
            dv_mask = df['dv'].between(*bounds['dv'])
            mask = parent_mask & ap_mask & ml_mask & dv_mask

            df.loc[mask, 'area_acronym_custom'] = sub_area
            df.loc[mask, '__assigned'] = True
            #print(f"{sub_area}: {mask.sum()} units assigned")

        remaining = df[(df['area_acronym_custom'] == parent_area) & (~df['__assigned'])]
        remaining_coords = remaining[['ap', 'ml', 'dv']].values
        if verbose:
            print(f"Unassigned {parent_area}: {len(remaining)} units at coords:", remaining_coords)

    df.drop(columns=['__assigned'], inplace=True)
    return df

def create_area_groupings(df, verbose=False):
    """
    Create area groupings based on custom area acronyms.
    :param df:
    :param verbose:
    :return:
    """
    if verbose:
        print('Creating area groupings...')
    medial_pfc = {"PL", "ILA", "IL", "ACA", "ACAd", "ACAv"} #ORBm for mPFC?
    ppc = {"VIS", "VISa", "VISam", "VISl", "VISpm", "VISrl", "VISal", "SSp-tr", "SSp-un"}

    def classify(acronym):
        if acronym in medial_pfc:
            return "mPFC"
        elif acronym in ppc:
            return "PPC"

    df = df.copy()
    # Override area_acronym_custom
    df['area_acronym_custom'] = df.apply(lambda row: classify(row['area_acronym_custom']) if classify(row['area_acronym_custom']) else row['area_acronym_custom'], axis=1)
    return df

def create_thalamic_groupings(df, verbose=False):
    """
    Create thalamic area groupings based on functional organization.
    Groups thalamic nuclei into 8 functional clusters; non-thalamic areas are unchanged.
    :param df: DataFrame with 'area_acronym_custom' column
    :param verbose:
    :return: DataFrame with thalamic nuclei grouped in 'area_acronym_custom'
    """
    if verbose:
        print('Creating thalamic groupings...')

    anterior          = {"AD", "AV", "AM", "IAD", "IAM", "LD", "ATN"}
    medial            = {"MD", "MDc", "MDl", "MDm", "IMD", "SMT", "PR", "MT"}
    midline_intralaminar = {"PVT", "PT", "Re", "Rh", "Xi", "CM", "CL", "PC", "Pf", "PIL", "ILM", "IntG", "ILN"}
    ventral           = {"VPM", "VPL", "VPMpc", "VPLpc", "VAL", "VM", "PoT", "VP", "VT"}
    posterior_lp      = {"PO", "LP", "SPA", "PP", "SGN", "Eth", "SubG"}
    visual_lgn        = {"LGd", "LGd-co", "LGd-ip", "LGd-sh", "LGv", "IGL", "OPT"}
    auditory_mgn      = {"MGv", "MGd", "MGm", "SG", "LT", "MG", "MGN"}
    reticular         = {"RT"}

    def classify(acronym):
        if acronym in anterior:
            return "ATN" #limbic, memory, context
        elif acronym in medial:
            return "MT" #limbic, prefrontal relay, cognitive control
        elif acronym in midline_intralaminar:
            return "ILN" # arousal, salience
        elif acronym in ventral:
            return "VT" # more somatosensory and motor and cerebellar
        elif acronym in posterior_lp:
            return "PT" # higher-order multimodal, cortical feedback
        elif acronym in visual_lgn:
            return "LGN" # visual
        elif acronym in auditory_mgn:
            return "MGN" # auditory
        elif acronym in reticular:
            return "RT" #inhibitory gating

    df = df.copy()
    df['area_acronym_custom'] = df.apply(
        lambda row: classify(row['area_acronym_custom']) or row['area_acronym_custom'],
        axis=1
    )
    return df

def process_allen_labels(df, split_merge_areas=False):
    """
    Process the DataFrame to create custom area acronyms, layer numbers, and bregma-centric coordinates.
    :param df: unit_table pd.DataFrame from NWB files
    :param params: dictionary of parameters
    :return:
    """

    # Check if processing can be performed
    required_cols = ['target_region', 'ccf_atlas_acronym', 'ccf_atlas_parent_acronym', 'ccf_ap', 'ccf_ml', 'ccf_dv', 'mouse_id']
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        print(f'allen_utils.process_allen_labels: Warning - missing required columns for processing. Available columns: {df.columns}, Missing: {missing_cols}')
        text = f"Missing required columns for processing Allen labels: {', '.join(missing_cols)}. Available columns: {', '.join(df.columns)}"
        raise ValueError(text)

    # Remove unwanted areas
    df = df[~df['ccf_atlas_acronym'].isin(get_excluded_areas())]

    # Create custom area acronyms simplifying ccf areas acronyms
    #temp: remove nan from required columns
    df['ccf_atlas_acronym'] = df['ccf_atlas_acronym'].apply(lambda x: str(x))
    rows_with_nan = df.loc[
        df[required_cols].isna().any(axis=1),
        ["mouse_id", "cluster_id", "ccf_atlas_acronym"]
    ]
    print(f"Warning: mouse with nan ccf labels ({len(rows_with_nan)}):", rows_with_nan.mouse_id.unique(), rows_with_nan)
    df = create_area_custom_column(df)

    # Create layer number column
    df = create_layer_number_column(df)

    # Create a ccf_acronym_no_layer column, copy of ccf_acronym just without layer info
    df = create_ccf_acronym_no_layer_column(df)

    # Create bregma-centric coordinates, going from CCF (BrainGlobe) to bregma-centric coordinates using IBL bregma estimate
    df = create_bregma_centric_coords_from_ccf(df)

    # Create areas subdivisions for specific areas using custom boundaries
    if split_merge_areas:
        df = create_areas_subdivisions(df, verbose=False)
        #df = create_area_groupings(df, verbose=False)
        df = create_thalamic_groupings(df, verbose=False)

    # Add higher-level area groups
    area2group = ({a: g for g, areas in get_custom_area_groups().items() for a in areas})
    df['area_group'] = df['area_acronym_custom'].map(area2group).fillna('Other')
    other_counts = df.loc[df['area_group'] == 'Other', 'area_acronym_custom'].value_counts()
    if len(other_counts):
        print(f"Warning: {len(other_counts)} area(s) mapped to 'Other' (not in get_custom_area_groups()):")
        for area, n in other_counts.items():
            print(f"  {area}: {n} units")

    # Add isocortex / rest
    existing_areas = df['area_acronym_custom'].unique()
    # get areas not present in
    missing_areas = [a for a in existing_areas if a not in get_area_acronym_custom_order()]
    isocortex2group = {area: group for group, areas in get_isocortex_mapping().items() for area in areas}
    df['isocortex_group'] = df['area_acronym_custom'].map(isocortex2group).fillna('Other')


    return df

def keep_shared_areas(data_df, nomenclature, n_min_units=10, n_min_mice=3):
    """
    Keep set of shared areas across reward groups with (at least n_min_units units) and n_min_mice mice in each group.
    :param data_df: dataframe containing at least columns for reward_group, mouse_id, unit_id, and the specified nomenclature (e.g., area_acronym_custom).
    :param nomenclature: the column name in data_df to use for area labels (e.g., 'area_acronym_custom').
    :param n_min_units: minimum number of unique units required in each reward group for an area to be kept, across mice
    :param n_min_mice: minimum number of unique mice required in each reward group for an area to be kept.
    :return:
    """
    print(f'Filtering for R+/R- shared areas in {nomenclature} with at least '
          f'{n_min_units} units and {n_min_mice} mice in each reward group...')

    if nomenclature not in data_df.columns:
        raise ValueError(f"Specified nomenclature column '{nomenclature}' not found.")

    # Intersect areas across reward groups
    areas_rplus = data_df.loc[data_df['reward_group'] == 1, nomenclature].unique()
    areas_rminus = data_df.loc[data_df['reward_group'] == 0, nomenclature].unique()
    areas_intersect = set(areas_rplus).intersection(set(areas_rminus))

    print(f'Found {len(areas_intersect)} R+/R- shared areas: {areas_intersect}')

    # Count mice (always required)
    n_mice = (
        data_df
        .groupby(['reward_group', nomenclature])['mouse_id']
        .nunique()
        .unstack(fill_value=0)
    )

    # Count units ONLY if quality_label exists i.e. unit-level data present
    use_unit_threshold = 'quality_label' in data_df.columns

    if use_unit_threshold:
        print("Using unit counts (filtered by quality_label).")
        df_units = data_df[data_df['quality_label'].isin(['good', 'mua'])]

        n_units = (
            df_units
            .groupby(['reward_group', nomenclature])['unit_id']
            .nunique()
            .unstack(fill_value=0)
        )
    else:
        print("No 'quality_label' column found — skipping unit threshold.")
        n_units = None  # not used

    # Apply thresholds
    shared_areas = []

    for area in areas_intersect:

        # ---- Mouse condition (always enforced)
        mice_rplus = n_mice.get(area, {}).get(1, 0) if area in n_mice.columns else 0
        mice_rminus = n_mice.get(area, {}).get(0, 0) if area in n_mice.columns else 0
        mice_ok = (mice_rplus >= n_min_mice) and (mice_rminus >= n_min_mice)

        # ---- Unit condition (only if quality_label exists)
        if use_unit_threshold and n_min_units > 0:
            units_rplus = n_units.get(area, {}).get(1, 0) if area in n_units.columns else 0
            units_rminus = n_units.get(area, {}).get(0, 0) if area in n_units.columns else 0
            units_ok = (units_rplus >= n_min_units) and (units_rminus >= n_min_units)
        else:
            units_ok = True  # ignore unit threshold

        if mice_ok and units_ok:
            shared_areas.append(area)

    print(f'Keeping {len(shared_areas)} shared areas meeting thresholds.')
    if len(shared_areas) > 0:
        print("Shared areas:", shared_areas)

    # Final filtering
    data_df = data_df[data_df[nomenclature].isin(shared_areas)]

    return data_df, shared_areas



def simplify_area_no_layer(ccf_acronym, ccf_parent_acronym):
    """Layer-collapse only — matches create_ccf_acronym_no_layer_column's logic.
    Use this to build MERGE_KEY for external area-lookup tables (Harris, Gao, Liu),
    since it stays in the same raw nomenclature as unit_table's MERGE_KEY."""
    base = ccf_parent_acronym if contains_layer(ccf_acronym) else ccf_acronym
    return handle_ssp_bfd(base)

def merge_liu_avg_ipsi_opt(df, cols_priority=None):
    """
    Merge Liu et al. avg_ipsi values onto the DataFrame using multiple possible columns.

    :param df: DataFrame containing atlas acronym columns
    :param cols_priority: list of column names to try in order
                          (default: ['ccf_atlas_acronym', 'ccf_atlas_parent_acronym'])
    :return: DataFrame with new column 'avg_ipsi'
    """

    liu_avg_ipsi = load_liu_et_al_avg_ipsi()

    if cols_priority is None:
        cols_priority = ['ccf_atlas_parent_acronym']
    else:
        cols_priority = [cols_priority]

    # Initialize column
    df['avg_ipsi'] = np.nan

    for col in cols_priority:

        missing_mask = df['avg_ipsi'].isna()
        matched_values = df.loc[missing_mask, col].map(liu_avg_ipsi)
        df.loc[missing_mask, 'avg_ipsi'] = matched_values
        #print(f"{col}: matched {(~matched_values.isna()).sum()} rows")

    # Final report
    total_missing = df['avg_ipsi'].isna().sum()
    print(f"Total missing rows after merge: {total_missing}")

    return df


DATA_DIR = pathlib.Path(__file__).parent.parent / 'allen_utils' / 'data'
MERGE_KEY = 'ccf_atlas_acronym_no_layer'

def _load_excel(filename, missing_file_hint='', **kwargs):
    """Generic excel loader with a clear error message if the file is missing."""
    path_to_file = DATA_DIR / filename
    if not path_to_file.is_file():
        raise FileNotFoundError(f"Data file not found at {path_to_file}. {missing_file_hint}")
    return pd.read_excel(path_to_file, **kwargs)


def _dedupe_on_key(df, value_cols, source_name):
    """Collapse duplicate MERGE_KEY rows by averaging value_cols, warning if it happens."""
    n_dupes = df[MERGE_KEY].duplicated().sum()
    if n_dupes:
        print(f"{source_name}: {n_dupes} duplicate '{MERGE_KEY}' rows found, averaging {value_cols}.")
        df = df.groupby(MERGE_KEY, as_index=False)[value_cols].mean()
    return df


def _merge_and_report(df, source_df, value_col, source_name):
    """Left-merge source_df onto df on MERGE_KEY, overwrite value_col, report unmatched keys."""
    if MERGE_KEY not in df.columns:
        raise KeyError(f"'{MERGE_KEY}' column not found in target DataFrame.")

    df = df.merge(source_df, on=MERGE_KEY, how='left', suffixes=('', '_new'))
    if f'{value_col}_new' in df.columns:
        df[value_col] = df[f'{value_col}_new']
        df = df.drop(columns=[f'{value_col}_new'])

    matched = sorted(df.loc[df[value_col].notna(), MERGE_KEY].dropna().unique().tolist())
    unmatched = sorted(df.loc[df[value_col].isna(), MERGE_KEY].dropna().unique().tolist())
    print(f"{source_name}: added column '{value_col}' -- "
          f"matched {df[value_col].notna().sum()}/{len(df)} rows, "
          f"{len(matched)} matched {MERGE_KEY} values, "
          f"{len(unmatched)} unmatched {MERGE_KEY} values.")
    #if matched:
        #print(f"  Matched: {matched}")
    #if unmatched:
        #print(f"  Unmatched: {unmatched}")

    return df


# ---------------------------------------------------------------- Harris et al.

def load_process_hierarchy_from_harris():
    """
    Load the Allen atlas hierarchy from Harris et al. 2019 (cortex and thalamus only),
    collapsed to one row per no-layer area.
    :return: DataFrame with columns [MERGE_KEY, 'cc_tc_ct_iterated'].
    """
    filename = 'hierarchy_summary_CreConf.xlsx'
    hint = 'Get data from: https://github.com/AllenInstitute/MouseBrainHierarchy/tree/master'
    hierarchy_df = _load_excel(filename, hint, sheet_name='hierarchy_all_regions')
    hierarchy_df = hierarchy_df.rename(columns={'CC+TC+CT iterated': 'cc_tc_ct_iterated',
                                                 'areas': 'ccf_acronym'})

    # create_area_custom_column is assumed to strip layer suffixes (e.g. "VISp2/3" -> "VISp")
    # and return this in 'area_acronym_custom'
    #hierarchy_df = create_area_custom_column(hierarchy_df)
    hierarchy_df[MERGE_KEY] = hierarchy_df.apply(
        lambda row: simplify_area_no_layer(row['ccf_acronym'], row.get('ccf_parent_acronym', row['ccf_acronym'])),
        axis=1
    )
    #hierarchy_df[MERGE_KEY] = hierarchy_df['area_acronym_custom']

    hierarchy_df = _dedupe_on_key(hierarchy_df, ['cc_tc_ct_iterated'], 'Harris hierarchy')
    return hierarchy_df


def merge_hierarchy_from_harris(df):
    """Merge Harris et al. hierarchy scores onto df, matched on MERGE_KEY."""
    return _merge_and_report(df, load_process_hierarchy_from_harris(), 'cc_tc_ct_iterated', 'Harris hierarchy')


# ---------------------------------------------------------------- Liu et al.

def load_liu_et_al_avg_ipsi():
    """
    Load Liu et al. group-average ipsi fluorescence values (log-transformed).
    :return: DataFrame with columns [MERGE_KEY, 'avg_ipsi'] (avg_ipsi is ln-transformed).
    """
    filename = 'Liu_et_al_Group_averages_ranked.xlsx'
    liu_df = _load_excel(filename)
    liu_df = liu_df.iloc[2:].reset_index(drop=True)  # first two rows are headers
    liu_df = liu_df.rename(columns={'Unnamed: 0': MERGE_KEY})
    liu_df = liu_df[[MERGE_KEY, 'avg_ipsi']].dropna(subset=[MERGE_KEY])
    liu_df['avg_ipsi'] = pd.to_numeric(liu_df['avg_ipsi'], errors='coerce')
    #with np.errstate(divide="ignore", invalid="ignore"):
    #    liu_df['avg_ipsi'] = np.where(liu_df['avg_ipsi'] > 0, np.log10(liu_df['avg_ipsi']), np.nan)
    return _dedupe_on_key(liu_df, ['avg_ipsi'], 'Liu et al.')


def merge_liu_avg_ipsi(df):
    """Merge Liu et al. avg_ipsi values onto df, matched on MERGE_KEY. Adds 'avg_ipsi'."""
    liu_df = load_liu_et_al_avg_ipsi().rename(columns={'avg_ipsi': 'avg_ipsi'})
    return _merge_and_report(df, liu_df, 'avg_ipsi', 'Liu et al.')


# ---------------------------------------------------------------- Gao et al.

def load_process_hierarchy_from_gao():
    """
    Load the Gao et al. corticocortical hierarchy scores.
    :return: DataFrame with columns [MERGE_KEY, 'cc_hierarchy_score'].
    """
    filename = 'Gao_cc_hierarchy_mmc6.xlsx'
    hierarchy_df = _load_excel(filename, 'Get from the Gao et al. supplementary materials on Github.')
    hierarchy_df = hierarchy_df.rename(columns={'Mean': 'cc_hierarchy_score', 'Region': MERGE_KEY})
    hierarchy_df = hierarchy_df[[MERGE_KEY, 'cc_hierarchy_score']]
    return _dedupe_on_key(hierarchy_df, ['cc_hierarchy_score'], 'Gao et al.')


def merge_hierarchy_from_gao(df):
    """Merge Gao et al. hierarchy scores onto df, matched on MERGE_KEY."""
    return _merge_and_report(df, load_process_hierarchy_from_gao(), 'cc_hierarchy_score', 'Gao et al.')


def load_process_hierarchy_columns_from_gao():
    """
    Load the Gao et al. corticocortical hierarchy scores at the single
    cortical column level, together with each column's 3D CCF coordinates
    (so that individual neurons can later be assigned to their nearest
    column).

    :return: DataFrame with columns [MERGE_KEY, 'ColumnID',
        'cc_hierarchy_score_columns', 'col_ap', 'col_dv', 'col_ml'].
        NOT deduped on MERGE_KEY (area), since several columns can share the
        same area label and we need every individual column's coordinates.
    """
    GAO_COLUMN_COORD_MAP = {'X': 'ap', 'Y': 'dv', 'Z': 'ml'}

    filename = 'Gao_cortical_columns_ROIs.xlsx'
    hierarchy_df = _load_excel(filename, 'Get from the Gao et al. supplementary materials on Github.')
    hierarchy_df = hierarchy_df.rename(columns={
        'HierarchyScore': 'cc_hierarchy_score_columns',
        'Region': MERGE_KEY,
        'X': f'col_{GAO_COLUMN_COORD_MAP["X"]}',
        'Y': f'col_{GAO_COLUMN_COORD_MAP["Y"]}',
        'Z': f'col_{GAO_COLUMN_COORD_MAP["Z"]}',
        'U': 'col_u',
        'V': 'col_v',
    })
    hierarchy_df = hierarchy_df[[MERGE_KEY, 'ColumnID', 'cc_hierarchy_score_columns',
                                  'col_ap', 'col_dv', 'col_ml', 'col_u', 'col_v']]
    # NB: Gao et al. X/Y/Z are 10 um-resolution CCFv3 voxel indices; convert to um
    # to match ccf_atlas_ml/dv/ap.
    hierarchy_df[['col_ap', 'col_dv', 'col_ml']] *= 10

    # Gao's columns are defined on the opposite hemisphere from our neurons;
    # mirror col_ml about the midline so both sit in the same hemisphere.
    CCF_ML_MIDLINE = 5700  # half of full CCF ML extent in um -- confirm against your atlas
    hierarchy_df['col_ml'] = 2 * CCF_ML_MIDLINE - hierarchy_df['col_ml']

    return hierarchy_df.reset_index(drop=True)

def merge_hierarchy_columns_from_gao(df):
    """
    Assign each cortical neuron in df to its nearest Gao et al. column
    (nearest neighbor in ml/dv/ap CCF space) and attach that column's
    hierarchy score.

    A neuron is only eligible for assignment if its area label
    (df[MERGE_KEY]) is one of the areas present in the Gao et al. column
    table (i.e. it's a cortical area covered by Gao et al.). Neurons in any
    other area are left with NaN.

    :param df: DataFrame with columns [MERGE_KEY, 'ccf_Atlas_ml',
        'ccf_Atlas_dv', 'ccf_Atlas_ap'].
    :return: df with two new columns added: 'cc_hierarchy_score_columns'
        (nearest column's hierarchy score) and 'nearest_gao_column_id'
        (Gao et al. ColumnID of the nearest column), both NaN for
        non-cortical / non-covered neurons.
    """
    hierarchy_df = load_process_hierarchy_columns_from_gao()
    cortical_areas = set(hierarchy_df[MERGE_KEY].unique())

    df = df.copy()
    df['cc_hierarchy_score_columns'] = np.nan
    df['nearest_gao_column_id'] = np.nan

    is_eligible = df[MERGE_KEY].isin(cortical_areas)
    #print('Matched:', list(df[MERGE_KEY][is_eligible].unique()))
    n_eligible = int(is_eligible.sum())
    n_total = len(df)

    if n_eligible == 0:
        print(f"[Gao et al.] No neurons matched a cortical area covered by "
              f"the column table ({n_total} neurons total); skipping nearest-column assignment.")
        return df

    column_coords = hierarchy_df[['col_ml', 'col_dv', 'col_ap']].to_numpy()
    tree = cKDTree(column_coords) #create tree

    neuron_coords = df.loc[is_eligible, ['ccf_atlas_ml', 'ccf_atlas_dv', 'ccf_atlas_ap']].to_numpy()
    distances, nearest_idx = tree.query(neuron_coords, k=1) # look for nearest neighbor

    df.loc[is_eligible, 'cc_hierarchy_score_columns'] = hierarchy_df['cc_hierarchy_score_columns'].to_numpy()[nearest_idx]
    df.loc[is_eligible, 'nearest_gao_column_id'] = hierarchy_df['ColumnID'].to_numpy()[nearest_idx]

    print(f"[Gao et al.] Assigned nearest cortical column to {n_eligible}/{n_total} neurons "
          f"({n_total - n_eligible} left as NaN: area not covered by Gao et al. columns). "
          f"Mean nearest-neighbor distance = {distances.mean():.1f}, "
          f"max = {distances.max():.1f} (same units as ccf_atlas_ml/dv/ap).")

    return df