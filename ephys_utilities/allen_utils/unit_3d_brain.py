"""
Interactive 3D visualisation of probe track coordinates with brain surface.

Data sources:
  - Valid sessions: M:\\share_internal\\Axel_Bisi_Share\\dataset_info\\joint_probe_insertion_info.xlsx
  - Track files:    M:\\analysis\\Axel_Bisi\\data\\{mouse}\\{session}\\Anatomy\\fused\\registered
                    \\segmentation\\atlas_space\\tracks

Outputs (per color scheme):
  unit_3d_brain_{region|ccf_acronym}.html  — interactive Plotly (open in browser)
  unit_3d_brain_{region|ccf_acronym}.mp4   — rotating animation

Color schemes:
  'region'      — broad anatomical grouping via REGION_DICT
  'ccf_acronym' — raw CCF parent acronym

Brain surface: IBL AllenAtlas via marching cubes (left hemisphere only).
Right-hemisphere probes are mirrored to left. Bregma: ML=5739, AP=5400, DV=332 µm.

Filters applied:
  - Session must appear in joint_probe_insertion_info.xlsx (valid probe insertions)
  - Session folder must contain at least one 'Ephys' subfolder (one level below session dir)
  - tracks file must exist and be non-empty
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from iblatlas import atlas as ibl_atlas
from skimage import measure
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── paths ─────────────────────────────────────────────────────────────────────

INSERTION_INFO_XLSX = Path(
    r'M:\share_internal\Axel_Bisi_Share\dataset_info\joint_probe_insertion_info.xlsx'
)
DATA_ROOT = Path(r'M:\analysis\Axel_Bisi\data')
SAVE_PATH = Path(
    r'M:\analysis\Axel_Bisi\combined_results\unit_coordinates'
)
LOCAL_OUT = SAVE_PATH#Path('outputs/figures')

# ── atlas constants ────────────────────────────────────────────────────────────

BREGMA_ML_UM = 5739.0
BREGMA_AP_UM = 5400.0
BREGMA_DV_UM = 332.0

# Probe cylinder display config (radius auto-computed per pair)
DV_CORTEX_SURFACE_UM = 200
DV_CORTEX_BOTTOM_UM  = -1400

# ── region / color config ─────────────────────────────────────────────────────

REGION_DICT = {
    'HIP': ['CA1', 'CA2', 'CA3', 'DG', 'SUB', 'ProS'],
    'RSP': ['RSPagl', 'RSPd', 'RSPv'],
    'wM1': ['MOs'],
    'wM2': ['MOp'],
    'tjS1': ['SSp-tr', 'SSp-ll', 'SSp-ul', 'SSp-un'],
    'TH':  ['VPM', 'VPL', 'POL', 'LP', 'LD', 'MD', 'VM', 'VAL', 'CM',
            'IMD', 'PF', 'CL', 'RT', 'SGN', 'PP', 'SPFp'],
    'wS1': ['SSp-bfd'],
    'ACA': ['ACAd', 'ACAv'],
    'mid': ['MRN', 'SCm', 'SCsg', 'SCdg', 'IC', 'PAG'],
    'STR': ['CP', 'ACB', 'LS', 'LSc', 'LSr', 'LSv', 'SF', 'SI'],
}

REGION_COLORS = {
    'HIP':        '#4C72B0',
    'RSP':        '#DD8452',
    'wM1':        '#59A14F',
    'wM2':        '#76B7B2',
    'tjS1':       '#E15759',
    'TH':         '#B07AA1',
    'wS1':        '#FF9DA7',
    'ACA':        '#9C755F',
    'mid':        '#BAB0AC',
    'STR':        '#EDC948',
    'Unassigned': '#CCCCCC',
}

_AMBIGUOUS = {'root', 'fiber tracts', 'cc', 'fxs', 'lfbst', 'mfbc',
              'Unassigned', 'unknown', ''}

# Build reverse lookup: ccf_acronym → region label
_ACRONYM_TO_REGION: dict[str, str] = {}
for _region, _acrs in REGION_DICT.items():
    for _a in _acrs:
        _ACRONYM_TO_REGION[_a] = _region


# ── color scheme ──────────────────────────────────────────────────────────────

def build_color_scheme(df: pd.DataFrame, color_by: str) -> tuple[str, list, dict]:
    """Return (col, ordered_labels, color_dict) for the requested grouping."""
    if color_by == 'region':
        col     = 'region'
        named   = [r for r in REGION_DICT if r in df[col].values]
        unassigned = ['Unassigned'] if 'Unassigned' in df[col].values else []
        ordered = named + unassigned
        colors  = {**REGION_COLORS,
                   **{r: '#888888' for r in ordered if r not in REGION_COLORS}}
        return col, ordered, colors

    if color_by == 'ccf_acronym':
        col    = 'ccf_acronym'
        counts = df[col].value_counts()
        named  = [l for l in counts.index if l not in _AMBIGUOUS]
        last   = [l for l in counts.index if l in _AMBIGUOUS]
        ordered = named + last
        palette = sns.color_palette('tab20', n_colors=max(len(named), 1))
        colors  = {lbl: f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
                   for lbl, (r, g, b) in zip(named, palette)}
        for lbl in last:
            colors[lbl] = '#CCCCCC'
        return col, ordered, colors

    raise ValueError(f'Unknown color_by={color_by!r}')


# ── session discovery ──────────────────────────────────────────────────────────

def load_valid_sessions(xlsx_path: Path) -> pd.DataFrame:
    """
    Read joint_probe_insertion_info.xlsx and return a DataFrame with columns:
      mouse_name, date (str YYYYMMDD), probe_id
    Only rows where all three fields are non-null are kept.
    """
    df = pd.read_excel(xlsx_path, dtype=str)
    df['date'] = df['date'].astype(str)
    #df.columns = df.columns.str.strip()


    ## Flexible column detection — adapt these if the actual column names differ
    #col_map = {}
    #for col in df.columns:
    #    cl = col.lower().strip()
    #    if 'mouse_name' in cl or 'subject' in cl or 'animal' in cl:
    #        col_map.setdefault('mouse_name', col)
    #    elif 'date' in cl:
    #        col_map.setdefault('date', col)
    #    elif 'probe_id' in cl:
    #        col_map.setdefault('probe_id', col)
#
    #missing = [k for k in ('mouse_name', 'date', 'probe_id') if k not in col_map]
    #if missing:
    #    raise ValueError(
    #        f"Could not auto-detect columns {missing} in {xlsx_path}.\n"
    #        f"Available columns: {list(df.columns)}"
    #    )

    #out = df[[col_map['mouse_name'], col_map['date'], col_map['probe_id']]].copy()
    out = df[['mouse_name', 'date', 'probe_id']].copy()
    out.columns = ['mouse_name', 'date', 'probe_id']
    out = out.dropna(subset=['mouse_name', 'date', 'probe_id'])
    out = out[out['mouse_name'].str.strip() != '']

    # Normalise date to 8-digit string (drop time part if present)
    #out['date'] = out['date'].str.strip().str[:8]

    print(f"  {len(out)} valid probe insertions across "
          f"{out['mouse_name'].nunique()} mice from spreadsheet.")
    return out


def _has_ephys_subfolder(session_dir: Path) -> bool:
    """Return True if any direct child of session_dir is named 'Ephys' (case-insensitive)."""
    try:
        return any(
            p.is_dir() and 'ephys' in p.name.lower()
            for p in session_dir.iterdir()
        )
    except PermissionError:
        return False


def discover_tracks(valid_df: pd.DataFrame) -> list[dict]:
    """
    For every (mouse_name, date) pair in valid_df, find session dirs matching
      DATA_ROOT / mouse_name / {mouse_name}_{date}_* 
    that also have an Ephys subfolder, then collect all tracks files.

    Returns a list of dicts:
      {mouse, session_dir, session_name, probe_id, tracks_path}
    """
    records = []

    # Group by mouse so we only list each mouse directory once
    for mouse, group in valid_df.groupby('mouse_name'):
        mouse_dir = DATA_ROOT / mouse
        if not mouse_dir.is_dir():
            continue

        valid_dates = set(group['date'].tolist())

        # List session dirs once per mouse — O(1) filesystem call per mouse
        try:
            session_dirs = [p for p in mouse_dir.iterdir() if p.is_dir()]
        except PermissionError:
            continue

        for sess_dir in session_dirs:
            name = sess_dir.name  # e.g. AB142_20241128_113227
            parts = name.split('_')
            if len(parts) < 2:
                continue
            date_part = parts[1] if len(parts) > 1 else ''
            DD, MM, YYYY = date_part[0:2], date_part[2:4], date_part[4:8]
            YYYY, MM, DD = date_part[0:4], date_part[4:6], date_part[6:8]
            date_part = f"{DD}.{MM}.{YYYY}"

            if not _has_ephys_subfolder(sess_dir):
                continue

            tracks_path = (sess_dir / 'Anatomy' / 'fused' / 'registered'
                           / 'segmentation' / 'atlas_space' / 'tracks')
            if not tracks_path.is_dir():
                continue

            # One entry per probe registered in the spreadsheet for this session
            probe_ids = group[group['date'] == date_part]['probe_id'].tolist()
            for probe_id in probe_ids:
                records.append({
                    'mouse':        mouse,
                    'session_dir':  sess_dir,
                    'session_name': name,
                    'probe_id':     probe_id,
                    'tracks_path':  tracks_path,
                })

    print(f"  Discovered {len(records)} probe/session entries across "
          f"{len({r['mouse'] for r in records})} mice.")
    return records


# ── coordinate loading ────────────────────────────────────────────────────────

def _load_one_track(record: dict) -> list[dict]:
    """
    Load a single track file for a probe insertion.
    Tries tracks_path/{probe_id}.csv, then {probe_id}.npy, then any file
    whose stem matches probe_id (case-insensitive).

    Expected CSV/NPY format: rows of [AP_um, ML_um, DV_um] in CCF voxel µm
    (will be converted to bregma-relative µm).

    Returns a list of row-dicts ready for DataFrame construction.
    """
    tracks_path: Path = record['tracks_path']
    probe_id: str     = record['probe_id']

    candidate = None
    for suffix in ('.csv', '.npy', '.txt'):
        p = tracks_path / f'{probe_id}{suffix}'
        if p.exists():
            candidate = p
            break

    if candidate is None:
        # Fuzzy match — any file whose stem contains the probe id
        for f in tracks_path.iterdir():
            if probe_id.lower() in f.stem.lower():
                candidate = f
                break

    if candidate is None:
        return []

    try:
        if candidate.suffix == '.npy':
            arr = np.load(candidate)
        else:
            arr = np.loadtxt(candidate, delimiter=',')
    except Exception:
        return []

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 3:
        return []

    # CCF voxel µm → bregma-relative µm
    # Convention matches original script: ap positive = anterior, dv positive = dorsal
    ap_um = BREGMA_AP_UM - arr[:, 0]
    ml_um = arr[:, 1] - BREGMA_ML_UM
    dv_um = BREGMA_DV_UM - arr[:, 2]

    # Mirror right-hemisphere probes to left (ml < 0 means left)
    if float(np.median(ml_um)) > 0:
        ml_um = -ml_um

    rows = []
    for i in range(len(ap_um)):
        rows.append({
            'mouse':       record['mouse'],
            'session':     record['session_name'],
            'probe_id':    probe_id,
            'ap_um':       float(ap_um[i]),
            'ml_um':       float(ml_um[i]),
            'dv_um':       float(dv_um[i]),
            'ccf_acronym': 'unknown',   # populated below if atlas lookup available
            'region':      'Unassigned',
        })
    return rows


def collect_coordinates(records: list[dict], max_workers: int = 8) -> pd.DataFrame:
    """
    Load all track files in parallel and return a unified DataFrame.
    """
    all_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_load_one_track, rec): rec for rec in records}
        for fut in tqdm(as_completed(futures), total=len(futures), desc='Loading tracks'):
            all_rows.extend(fut.result())

    df = pd.DataFrame(all_rows)
    print(f"  Loaded {len(df):,} track points from {df['session'].nunique()} sessions.")
    return df


def assign_regions_from_atlas(df: pd.DataFrame, ba: ibl_atlas.AllenAtlas) -> pd.DataFrame:
    """
    For each point, look up the CCF acronym from the atlas label volume and
    assign the broad region via REGION_DICT.
    Done in vectorised form for speed.
    """
    if df.empty:
        return df

    # Convert bregma-relative µm → atlas xyz (metres) for IBL atlas mlapdv lookup
    # IBL AllenAtlas.get_labels expects xyz in metres: [ml, ap, dv]
    ml_m = df['ml_um'].values * 1e-6
    ap_m = df['ap_um'].values * 1e-6
    dv_m = df['dv_um'].values * 1e-6

    # Stack as (N, 3) in ml, ap, dv order expected by IBL atlas
    xyz = np.column_stack([ml_m, ap_m, dv_m])

    try:
        # get_labels returns integer region IDs
        region_ids = ba.get_labels(xyz, mapping='Allen')  # shape (N,)
        acronyms   = [ba.regions.acronym[rid] if rid != 0 else 'unknown'
                      for rid in region_ids]
    except Exception as e:
        print(f"  Warning: atlas lookup failed ({e}), acronyms will be 'unknown'.")
        acronyms = ['unknown'] * len(df)

    df = df.copy()
    df['ccf_acronym'] = acronyms
    df['region']      = [_ACRONYM_TO_REGION.get(a, 'Unassigned') for a in acronyms]
    return df


# ── brain surface ─────────────────────────────────────────────────────────────

def build_brain_surface(ba: ibl_atlas.AllenAtlas, decimate: int = 3):
    """Left-hemisphere brain surface mesh via marching cubes, in bregma-relative µm."""
    print('Computing brain surface mesh …')
    label      = ba.label[::decimate, ::decimate, ::decimate]
    brain_mask = label != 0

    verts, faces, _, _ = measure.marching_cubes(
        brain_mask.astype(np.float32), level=0.5, step_size=1,
    )
    res   = ba.res_um * decimate
    ap_um = (ba.bc.y0 * 1e6) + verts[:, 0] * res * np.sign(ba.bc.dy)
    ml_um = (ba.bc.x0 * 1e6) + verts[:, 1] * res * np.sign(ba.bc.dx)
    dv_um = (ba.bc.z0 * 1e6) + verts[:, 2] * res * np.sign(ba.bc.dz)

    keep     = ml_um <= 200
    vert_map = np.full(len(verts), -1, dtype=int)
    vert_map[keep] = np.arange(keep.sum())
    ap_um = ap_um[keep]; ml_um = ml_um[keep]; dv_um = dv_um[keep]
    valid_faces = keep[faces].all(axis=1)
    faces = vert_map[faces[valid_faces]]

    print(f'  mesh: {len(ap_um):,} vertices, {len(faces):,} faces')
    return ap_um, ml_um, dv_um, faces


# ── probe cylinders ───────────────────────────────────────────────────────────

def _cylinder_mesh(ml_c, ap_c, radius, dv_bottom, dv_top, n_theta=60):
    theta = np.linspace(0, 2 * np.pi, n_theta + 1)
    dv    = np.array([dv_bottom, dv_top])
    theta_g, dv_g = np.meshgrid(theta, dv)
    return ml_c + radius * np.cos(theta_g), ap_c + radius * np.sin(theta_g), dv_g


def build_probe_cylinders(df: pd.DataFrame, ba: ibl_atlas.AllenAtlas,
                           radius_um: float = 200.0) -> tuple[list, list]:
    """
    Build one cylinder per unique (session, probe_id) at its median ML/AP position.
    Returns (plotly_traces, mpl_params).
    """
    palette = sns.color_palette('husl', n_colors=max(df['probe_id'].nunique(), 1))
    probe_colors = {p: c for p, c in zip(sorted(df['probe_id'].unique()), palette)}

    plotly_traces = []
    mpl_params    = []
    dz_sign = int(np.sign(ba.bc.dz))

    for (session, probe_id), sub in df.groupby(['session', 'probe_id']):
        ml_c = float(sub['ml_um'].median())
        ap_c = float(sub['ap_um'].median())

        # DV extent from atlas
        ap_i = int(round((ap_c - ba.bc.y0 * 1e6) / (ba.res_um * int(np.sign(ba.bc.dy)))))
        ml_i = int(round((ml_c - ba.bc.x0 * 1e6) / (ba.res_um * int(np.sign(ba.bc.dx)))))
        ap_i = int(np.clip(ap_i, 0, ba.label.shape[0] - 1))
        ml_i = int(np.clip(ml_i, 0, ba.label.shape[1] - 1))
        dv_col = ba.label[ap_i, ml_i, :]
        brain_voxels = np.where(dv_col > 0)[0]
        if len(brain_voxels) > 0:
            surf_i  = brain_voxels[0]
            dv_surf = ba.bc.z0 * 1e6 + surf_i * ba.res_um * dz_sign
        else:
            dv_surf = DV_CORTEX_SURFACE_UM
        dv_bottom = dv_surf - 1400.0

        r, g, b   = probe_colors[probe_id]
        rgba_str  = f'rgba({int(r*255)},{int(g*255)},{int(b*255)},0.5)'
        ml_g, ap_g, dv_g = _cylinder_mesh(ml_c, ap_c, radius_um, dv_bottom, dv_surf)

        plotly_traces.append(go.Surface(
            x=ml_g, y=ap_g, z=dv_g,
            colorscale=[[0, rgba_str], [1, rgba_str]],
            showscale=False, opacity=0.5,
            name=f'{session} — {probe_id}',
            hoverinfo='name', showlegend=True,
            legendgroup=probe_id,
        ))
        mpl_params.append({
            'ml_g': ml_g, 'ap_g': ap_g, 'dv_g': dv_g,
            'color': (r, g, b), 'name': f'{probe_id}',
        })

    return plotly_traces, mpl_params


# ── plotly interactive ────────────────────────────────────────────────────────

def make_plotly_figure(df: pd.DataFrame, mesh,
                       color_by: str = 'region',
                       probe_cylinders: list | None = None) -> go.Figure:
    """
    Interactive 3D scatter with:
      - Dropdown to filter by mouse
      - Dropdown to filter by session
      - Dropdown to filter by region/ccf_acronym
    """
    ap_m, ml_m, dv_m, faces = mesh
    col, ordered, colors = build_color_scheme(df, color_by)

    # ── base traces ──────────────────────────────────────────────────────────
    traces = [go.Mesh3d(
        x=ml_m, y=ap_m, z=dv_m,
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color='lightgrey', opacity=0.12,
        flatshading=False, lighting=dict(ambient=0.8, diffuse=0.5),
        showscale=False, name='Brain surface', hoverinfo='skip',
    )]

    # One trace per (label × mouse × session) for fine-grained filtering
    for label in ordered:
        sub_label = df[df[col] == label]
        if sub_label.empty:
            continue

        for (mouse, session), sub in sub_label.groupby(['mouse', 'session']):
            is_bg   = label in _AMBIGUOUS
            color   = colors.get(label, '#888888')
            opacity = 0.15 if is_bg else 0.7
            size    = 2    if is_bg else 3

            traces.append(go.Scatter3d(
                x=sub['ml_um'], y=sub['ap_um'], z=sub['dv_um'],
                mode='markers',
                marker=dict(size=size, color=color, opacity=opacity,
                            line=dict(width=0)),
                name=label,
                customdata=np.column_stack([
                    sub['mouse'], sub['session'], sub['probe_id'], sub[col]
                ]),
                hovertemplate=(
                    '<b>%{customdata[0]} / %{customdata[1]}</b><br>'
                    'Probe: %{customdata[2]}<br>'
                    '%{customdata[3]}<br>'
                    'ML=%{x:.0f} AP=%{y:.0f} DV=%{z:.0f}<extra></extra>'
                ),
                visible=True,
                # Store metadata for dropdown filtering
                meta={'mouse': mouse, 'session': session, 'label': label},
            ))

    if probe_cylinders:
        traces.extend(probe_cylinders)

    fig = go.Figure(data=traces)

    # ── build dropdown helpers ───────────────────────────────────────────────
    mice    = sorted(df['mouse'].unique().tolist())
    sessions = sorted(df['session'].unique().tolist())
    labels   = ordered

    def _visibility(filter_key: str, filter_val: str | None) -> list[bool]:
        """Compute visibility array for all traces given a filter."""
        vis = []
        for tr in fig.data:
            meta = getattr(tr, 'meta', None)
            if meta is None:                 # brain surface / cylinders
                vis.append(True)
                continue
            if filter_key == 'all':
                vis.append(True)
            elif filter_key == 'mouse':
                vis.append(meta.get('mouse') == filter_val)
            elif filter_key == 'session':
                vis.append(meta.get('session') == filter_val)
            elif filter_key == 'label':
                vis.append(meta.get('label') == filter_val)
            else:
                vis.append(True)
        return vis

    # ── dropdown menus ───────────────────────────────────────────────────────
    def _buttons(filter_key: str, values: list[str]) -> list[dict]:
        btns = [dict(
            label='All',
            method='restyle',
            args=[{'visible': _visibility('all', None)}],
        )]
        for v in values:
            btns.append(dict(
                label=v,
                method='restyle',
                args=[{'visible': _visibility(filter_key, v)}],
            ))
        return btns

    updatemenus = [
        dict(
            buttons=_buttons('mouse', mice),
            direction='down', showactive=True, x=0.0, xanchor='left',
            y=1.15, yanchor='top', pad={'r': 10},
            bgcolor='white', bordercolor='#ccc',
            font=dict(size=11),
            active=0,
        ),
        dict(
            buttons=_buttons('session', sessions),
            direction='down', showactive=True, x=0.25, xanchor='left',
            y=1.15, yanchor='top', pad={'r': 10},
            bgcolor='white', bordercolor='#ccc',
            font=dict(size=11),
            active=0,
        ),
        dict(
            buttons=_buttons('label', labels),
            direction='down', showactive=True, x=0.55, xanchor='left',
            y=1.15, yanchor='top', pad={'r': 10},
            bgcolor='white', bordercolor='#ccc',
            font=dict(size=11),
            active=0,
        ),
    ]

    annotations = [
        dict(text='Mouse',   x=0.0,  xref='paper', y=1.18, yref='paper',
             showarrow=False, font=dict(size=11)),
        dict(text='Session', x=0.25, xref='paper', y=1.18, yref='paper',
             showarrow=False, font=dict(size=11)),
        dict(text=col.replace('_', ' ').title(),
             x=0.55, xref='paper', y=1.18, yref='paper',
             showarrow=False, font=dict(size=11)),
    ]

    fig.update_layout(
        title=dict(text=f'Probe track coordinates [{color_by}]', font_size=13),
        scene=dict(
            xaxis=dict(title='ML (µm)', backgroundcolor='white', gridcolor='#ddd'),
            yaxis=dict(title='AP (µm)', backgroundcolor='white', gridcolor='#ddd'),
            zaxis=dict(title='DV (µm)', backgroundcolor='white', gridcolor='#ddd'),
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8)),
        ),
        updatemenus=updatemenus,
        annotations=annotations,
        legend=dict(itemsizing='constant', font_size=10),
        margin=dict(l=0, r=0, t=100, b=0),
        paper_bgcolor='white',
    )
    return fig


# ── matplotlib rotating animation ────────────────────────────────────────────

def make_rotation_animation(df: pd.DataFrame, mesh,
                             color_by: str = 'region',
                             n_frames: int = 120, fps: int = 30,
                             probe_mpl_params: list | None = None):
    ap_m, ml_m, dv_m, faces = mesh
    verts = np.column_stack([ml_m, ap_m, dv_m])
    col, ordered, colors = build_color_scheme(df, color_by)

    fig = plt.figure(figsize=(8, 7), dpi=150)
    ax  = fig.add_subplot(111, projection='3d')

    stride     = max(1, len(faces) // 8000)
    face_verts = verts[faces[::stride]]
    poly = Poly3DCollection(face_verts, alpha=0.06, facecolor='lightgrey',
                            edgecolor='none', zorder=0)
    ax.add_collection3d(poly)

    if probe_mpl_params:
        for p in probe_mpl_params:
            ax.plot_surface(p['ml_g'], p['ap_g'], p['dv_g'],
                            color=p['color'], alpha=0.45, linewidth=0,
                            label=p['name'])

    for label in ordered:
        sub   = df[df[col] == label]
        if sub.empty:
            continue
        is_bg = label in _AMBIGUOUS
        ax.scatter(sub['ml_um'], sub['ap_um'], sub['dv_um'],
                   c=colors.get(label, '#888888'),
                   s=0.5 if is_bg else 2.0,
                   alpha=0.15 if is_bg else 0.6,
                   linewidths=0,
                   label=label if not is_bg else '_nolegend_',
                   rasterized=True)

    ax.set_xlabel('ML (µm)', fontsize=7, labelpad=2)
    ax.set_ylabel('AP (µm)', fontsize=7, labelpad=2)
    ax.set_zlabel('DV (µm)', fontsize=7, labelpad=2)
    ax.tick_params(labelsize=6)
    ax.set_title(f'Probe tracks [{color_by}]', fontsize=8, pad=4)
    ax.legend(loc='upper left', fontsize=6, frameon=False,
              markerscale=3, bbox_to_anchor=(-0.05, 1.05))

    def update(frame):
        ax.view_init(elev=20, azim=frame * 360 / n_frames)
        return (poly,)

    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                   interval=1000 / fps, blit=False)
    return fig, anim


# ── save helpers ──────────────────────────────────────────────────────────────

def save_html(fig, stem: str):
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    p = LOCAL_OUT / f'{stem}.html'
    fig.write_html(str(p), include_plotlyjs='cdn')
    print(f'  Saved interactive: {p}')
    if SAVE_PATH.parent.exists():
        SAVE_PATH.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(SAVE_PATH / f'{stem}.html'), include_plotlyjs='cdn')


def save_animation(fig, anim, stem: str, fps: int = 30):
    import shutil
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    p      = LOCAL_OUT / f'{stem}.mp4'
    writer = animation.FFMpegWriter(fps=fps, bitrate=2000,
                                    extra_args=['-vcodec', 'libx264',
                                                '-pix_fmt', 'yuv420p'])
    anim.save(str(p), writer=writer)
    plt.close(fig)
    print(f'  Saved animation: {p}')
    if SAVE_PATH.parent.exists():
        SAVE_PATH.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, SAVE_PATH / f'{stem}.mp4')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print('─' * 60)
    print('1 / 5  Loading valid sessions from spreadsheet …')
    valid_df = load_valid_sessions(INSERTION_INFO_XLSX)

    print('─' * 60)
    print('2 / 5  Discovering track files on disk …')
    records = discover_tracks(valid_df)
    if not records:
        print('No valid records found — check DATA_ROOT and spreadsheet paths.')
        return

    print('─' * 60)
    print('3 / 5  Loading track coordinates (parallel) …')
    df_raw = collect_coordinates(records)
    if df_raw.empty:
        print('No track data loaded — check tracks file format.')
        return

    print('─' * 60)
    print('4 / 5  Loading AllenAtlas & assigning CCF regions …')
    ba   = ibl_atlas.AllenAtlas()
    df   = assign_regions_from_atlas(df_raw, ba)
    mesh = build_brain_surface(ba, decimate=3)

    print('  Building probe cylinders …')
    probe_plotly, probe_mpl = build_probe_cylinders(df, ba)

    print('─' * 60)
    print('5 / 5  Generating figures …')
    for color_by in ('region', 'ccf_acronym'):
        stem = f'unit_3d_brain_{color_by}'
        print(f'\n  [{color_by}]')

        fig_plotly = make_plotly_figure(df, mesh, color_by=color_by,
                                        probe_cylinders=probe_plotly)
        save_html(fig_plotly, stem)

        fig_mpl, anim = make_rotation_animation(df, mesh, color_by=color_by,
                                                 n_frames=120, fps=30,
                                                 probe_mpl_params=probe_mpl)
        save_animation(fig_mpl, anim, stem)

    # ── summary ───────────────────────────────────────────────────────────────
    print('\n' + '─' * 60)
    included_mice = sorted(df['mouse'].unique().tolist())
    print(f'Done.  {len(included_mice)} mice included:')
    for m in included_mice:
        n_sess  = df[df['mouse'] == m]['session'].nunique()
        n_units = (df['mouse'] == m).sum()
        print(f'  {m:20s}  {n_sess} session(s)  {n_units:>6,} track points')


if __name__ == '__main__':
    main()
