"""
Interactive 3D visualisation of probe track coordinates using BrainGlobe/Brainrender
meshes, rendered interactively via Plotly.

Architecture
────────────
  Brainrender / BrainGlobe  →  region mesh geometry (.obj files, µm in ASR space)
  Plotly                    →  fully interactive HTML (no display server needed)

BrainGlobe allen_mouse_25um coordinate convention
  ASR  =  Anterior / Superior / Right
  axes:  [AP_µm,  DV_µm,  ML_µm]  (all in µm, origin at atlas corner)

Track .npy file column convention (confirmed):
  col 0  →  AP  (anterior–posterior, CCF µm)
  col 1  →  DV  (dorsal–ventral,    CCF µm)   ← NOTE: NOT ML
  col 2  →  ML  (medio-lateral,     CCF µm)   ← NOTE: NOT DV

Data sources (same as unit_3d_brain_bisi.py)
  - Valid sessions: INSERTION_INFO_XLSX
  - Track files:    DATA_ROOT/{mouse}/{session}/Anatomy/fused/registered/
                    segmentation/atlas_space/tracks/{probe_id}.{csv|npy|txt}

Outputs
  unit_3d_brain_brainrender_{area_acronym_custom|ccf_atlas_acronym|target_region}.html

Filters
  - Session in spreadsheet (valid probe insertion)
  - At least one 'Ephys' subfolder one level below session dir
  - tracks file exists and is non-empty

Interactive controls (Plotly dropdowns + legend)
  - Filter by Mouse / Session / Area
  - Three colour modes:
      area_acronym_custom  — allen_utils.process_allen_labels simplified areas
      ccf_atlas_acronym    — raw CCF acronym from atlas annotation
      target_region        — probe target from spreadsheet; only anatomically
                             matching points coloured (via apply_target_region_filters)
  - Brain region meshes togglable via legend

Requirements
  pip install brainrender brainglobe-atlasapi plotly pandas openpyxl seaborn tqdm
  (atlas is auto-downloaded on first run to ~/.brainglobe)
"""

from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import plotly.graph_objects as go

# ── paths ─────────────────────────────────────────────────────────────────────

INSERTION_INFO_XLSX = Path(
    r'M:\share_internal\Axel_Bisi_Share\dataset_info\joint_probe_insertion_info.xlsx'
)

# Both data roots are searched; missing roots are silently skipped.
DATA_ROOTS: list[Path] = [
    Path(r'M:\analysis\Axel_Bisi\data'),
    Path(r'M:\analysis\Myriam_Hamon\data'),
]

SAVE_PATH = Path(
    r'M:\analysis\Axel_Bisi\combined_results\unit_coordinates'
)
LOCAL_OUT = SAVE_PATH

# Path to allen_utils.py — used to fetch excluded areas at runtime
ALLEN_UTILS_DIR = Path(r'M:\analysis\Axel_Bisi\Github\allen_utils')

# ── recording-day filter ───────────────────────────────────────────────────────
# Set to a non-empty list to keep only those days, e.g. ['Day1', 'Day2'].
# Leave as [] to include all recording days.
DAY_OF_RECORDING_FILTER: list[str] = []

# BrainGlobe atlas to use — must match the CCF space your tracks are in
ATLAS_NAME = 'allen_mouse_25um'

# ── atlas / coordinate constants ───────────────────────────────────────────────

# CCF voxel-µm origin of bregma (same as unit_3d_brain_bisi.py)
BREGMA_AP_UM = 5400.0
BREGMA_ML_UM = 5739.0
BREGMA_DV_UM = 332.0

# Brain regions to render as transparent surface overlays.
# These are BrainGlobe acronyms from allen_mouse_25um.
# Add / remove freely — missing regions are silently skipped.
BRAIN_REGIONS_TO_RENDER: list[str] = [
    'root',                          # whole brain outline
    'CA1', 'CA3', 'DG',              # HIP
    'RSPagl', 'RSPd', 'RSPv',        # RSP
    'MOs', 'MOp',                    # M1/M2
    'SSp-bfd',                       # wS1
    'SSp-tr', 'SSp-ll', 'SSp-ul',   # tjS1
    'VPM', 'VPL', 'LP',              # TH subset
    'ACAd', 'ACAv',                  # ACA
    'CP',                            # STR
    'SCm', 'PAG',                    # mid
]

# Opacity per region type
REGION_ALPHA: dict[str, float] = {
    'root': 0.06,   # near-invisible whole-brain outline
}
DEFAULT_REGION_ALPHA = 0.18

# Mesh decimation: keep every Nth face to reduce file size (~8000 faces target)
MESH_FACE_LIMIT = 8_000

# ── region / color config ─────────────────────────────────────────────────────

REGION_DICT: dict[str, list[str]] = {
    'HIP':  ['CA1', 'CA2', 'CA3', 'DG', 'SUB', 'ProS'],
    'RSP':  ['RSPagl', 'RSPd', 'RSPv'],
    'wM1':  ['MOs'],
    'wM2':  ['MOp'],
    'tjS1': ['SSp-tr', 'SSp-ll', 'SSp-ul', 'SSp-un'],
    'TH':   ['VPM', 'VPL', 'POL', 'LP', 'LD', 'MD', 'VM', 'VAL', 'CM',
             'IMD', 'PF', 'CL', 'RT', 'SGN', 'PP', 'SPFp'],
    'wS1':  ['SSp-bfd'],
    'ACA':  ['ACAd', 'ACAv'],
    'mid':  ['MRN', 'SCm', 'SCsg', 'SCdg', 'IC', 'PAG'],
    'STR':  ['CP', 'ACB', 'LS', 'LSc', 'LSr', 'LSv', 'SF', 'SI'],
}

REGION_COLORS: dict[str, str] = {
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

# Region-level surface fill color (used for transparent meshes)
_REGION_SURFACE_COLORS: dict[str, str] = {
    'root': 'lightgrey',
    'HIP': '#4C72B0', 'RSP': '#DD8452', 'wM1': '#59A14F', 'wM2': '#76B7B2',
    'tjS1': '#E15759', 'TH': '#B07AA1', 'wS1': '#FF9DA7', 'ACA': '#9C755F',
    'mid': '#BAB0AC', 'STR': '#EDC948',
}

_AMBIGUOUS = {'root', 'fiber tracts', 'cc', 'fxs', 'lfbst', 'mfbc',
              'Unassigned', 'unknown', ''}

_ACRONYM_TO_REGION: dict[str, str] = {
    acr: region
    for region, acrs in REGION_DICT.items()
    for acr in acrs
}


# ── excluded-areas loader ─────────────────────────────────────────────────────

def get_excluded_areas() -> set[str]:
    """
    Import allen_utils from ALLEN_UTILS_DIR and return the set of CCF acronyms
    that should be excluded from all plots.  Falls back to an empty set if the
    module cannot be loaded (so the script still runs on machines without it).
    """
    import importlib.util, sys
    module_path = ALLEN_UTILS_DIR / 'allen_utils.py'
    if not module_path.exists():
        print(f"  Warning: allen_utils.py not found at {module_path} — "
              "no areas will be excluded.")
        return set()
    try:
        spec   = importlib.util.spec_from_file_location('allen_utils', module_path)
        mod    = importlib.util.module_from_spec(spec)
        sys.modules['allen_utils'] = mod
        spec.loader.exec_module(mod)
        excluded = set(mod.get_excluded_areas())
        print(f"  Excluding {len(excluded)} areas from allen_utils: "
              f"{sorted(excluded)}")
        return excluded
    except Exception as e:
        print(f"  Warning: could not load allen_utils ({e}) — "
              "no areas will be excluded.")
        return set()



def ccf_to_asr(ap_um: np.ndarray, dv_um: np.ndarray,
               ml_um: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert CCF voxel-µm (AP, DV, ML — actual column order in .npy files)
    to BrainGlobe ASR µm (Anterior, Superior, Right).

    Track col order   →  BrainGlobe ASR axis
      col 0: AP       →  axis 0  A  (anterior; small = rostral)
      col 1: DV       →  axis 1  S  (superior; small = dorsal)
      col 2: ML       →  axis 2  R  (right;    large = right)
    """
    return ap_um.copy(), dv_um.copy(), ml_um.copy()  # (A, S, R)


def bregma_relative(ap_ccf: np.ndarray, ml_ccf: np.ndarray,
                    dv_ccf: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert CCF voxel-µm to bregma-relative µm (kept for hover labels).
    Returns (ap_bregma, ml_bregma, dv_bregma) where
      ap > 0 = anterior, ml < 0 = left, dv > 0 = dorsal.
    """
    ap = BREGMA_AP_UM - ap_ccf
    ml = ml_ccf - BREGMA_ML_UM
    dv = BREGMA_DV_UM - dv_ccf
    return ap, ml, dv


# ── brainrender / brainglobe mesh extraction ───────────────────────────────────

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


def _decimate_mesh(verts: np.ndarray, faces: np.ndarray,
                   limit: int = MESH_FACE_LIMIT) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly subsample faces to at most `limit` triangles."""
    if len(faces) <= limit:
        return verts, faces
    stride = max(1, len(faces) // limit)
    faces  = faces[::stride]
    # Re-index vertices to only those used
    used   = np.unique(faces)
    remap  = np.full(len(verts), -1, dtype=np.int32)
    remap[used] = np.arange(len(used))
    return verts[used], remap[faces]


def load_region_meshes(atlas) -> dict[str, dict]:
    """
    For each region in BRAIN_REGIONS_TO_RENDER, load the .obj mesh via
    brainglobe_atlasapi and extract (vertices, faces) as numpy arrays.

    The atlas stores meshes in ASR µm.  We keep them in that space.

    Returns dict: acronym → {'verts': (N,3), 'faces': (M,3), 'color': hex,
                              'alpha': float, 'broad_region': str}
    """
    from brainglobe_atlasapi import BrainGlobeAtlas
    import vedo

    print(f"  Loading region meshes from BrainGlobe ({ATLAS_NAME}) …")
    meshes: dict[str, dict] = {}

    valid_acrs = set(atlas.lookup_df['acronym'].tolist())

    for acr in tqdm(BRAIN_REGIONS_TO_RENDER, desc='  Regions', leave=False):
        if acr not in valid_acrs:
            print(f"    Skipping {acr!r} — not in atlas")
            continue

        obj_path = atlas.meshfile_from_structure(acr)
        if not Path(obj_path).exists():
            print(f"    Skipping {acr!r} — mesh file not found")
            continue

        try:
            mesh = vedo.load(str(obj_path))
        except Exception as e:
            print(f"    Skipping {acr!r} — load error: {e}")
            continue

        verts = np.array(mesh.vertices, dtype=np.float32)   # (N, 3)  ASR µm
        cells = np.array(mesh.cells,    dtype=np.int32)     # (M, 3 or 4)

        # Ensure triangles
        if cells.shape[1] == 4:
            # Quad → two triangles
            cells = np.vstack([cells[:, :3], cells[:, [0, 2, 3]]])

        # Root mesh needs higher face count for a smooth wireframe outline;
        # sub-regions can be coarser.
        face_limit = 40_000 if acr == 'root' else MESH_FACE_LIMIT
        verts, cells = _decimate_mesh(verts, cells, limit=face_limit)

        # Determine color
        broad = _ACRONYM_TO_REGION.get(acr, acr)
        if acr == 'root':
            color = 'lightgrey'
            broad = 'root'
        else:
            color = _REGION_SURFACE_COLORS.get(broad,
                        '#' + ''.join(f'{c:02x}' for c in
                                      atlas.structures[acr]['rgb_triplet']))
        alpha = REGION_ALPHA.get(acr, DEFAULT_REGION_ALPHA)

        meshes[acr] = {
            'verts':        verts,
            'faces':        cells,
            'color':        color,
            'alpha':        alpha,
            'broad_region': broad,
        }

    print(f"  Loaded {len(meshes)} region meshes.")
    return meshes


# ── session discovery (identical logic to unit_3d_brain_bisi.py) ──────────────

def load_valid_sessions(xlsx_path: Path) -> pd.DataFrame:
    """
    Read joint_probe_insertion_info.xlsx and return a DataFrame with columns:
      mouse_name, date (str YYYYMMDD), probe_id
    Only rows where all three fields are non-null are kept.
    """
    df = pd.read_excel(xlsx_path, dtype=str)
    # Normalise the date column to a plain YYYY-MM-DD string regardless of
    # how Excel stored it (datetime object → "2024-11-28 00:00:00", plain
    # string "28.11.2024", "2024-11-28", integer serial, etc.).
    def _to_ymd(val) -> str:
        import re as _re
        if pd.isna(val):
            return ''
        # Already a datetime / Timestamp
        if hasattr(val, 'strftime'):
            return val.strftime('%Y-%m-%d')
        s = str(val).strip()
        # "2024-11-28 00:00:00" or "2024-11-28"
        m = _re.match(r'^(\d{4})-(\d{2})-(\d{2})', s)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        # "28.11.2024" or "28/11/2024"
        m = _re.match(r'^(\d{1,2})[./](\d{1,2})[./](\d{4})', s)
        if m:
            return f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"
        # "20241128"
        m = _re.match(r'^(\d{4})(\d{2})(\d{2})$', s)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        return s   # leave as-is, will fail to match and show in diagnostics
    df['date'] = df['date'].apply(_to_ymd)
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
    needed = ['mouse_name', 'date', 'probe_id', 'target_area', 'day_of_recording']
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        print(f"  Warning: columns not found in spreadsheet: {missing_cols}. "
              "They will be filled with 'unknown'.")
        for c in missing_cols:
            df[c] = 'unknown'

    out = df[needed].copy()
    out = out.dropna(subset=['mouse_name', 'date', 'probe_id'])
    out = out[out['mouse_name'].str.strip() != '']
    out['target_area']       = out['target_area'].fillna('unknown')
    out['day_of_recording']  = out['day_of_recording'].fillna('unknown').astype(str).str.strip()

    # Apply day_of_recording filter if requested
    if DAY_OF_RECORDING_FILTER:
        before = len(out)
        out = out[out['day_of_recording'].isin(DAY_OF_RECORDING_FILTER)]
        print(f"  day_of_recording filter {DAY_OF_RECORDING_FILTER}: "
              f"{before} → {len(out)} rows kept.")

    print(f"  {len(out)} valid probe insertions across "
          f"{out['mouse_name'].nunique()} mice from spreadsheet.")
    print(f"  Recording days present: {sorted(out['day_of_recording'].unique().tolist())}")
    return out


def _has_ephys_subfolder(session_dir: Path) -> bool:
    try:
        return any(p.is_dir() and 'ephys' in p.name.lower()
                   for p in session_dir.iterdir())
    except PermissionError:
        return False


def discover_tracks(valid_df: pd.DataFrame) -> list[dict]:
    """
    Search every DATA_ROOTS entry for track files matching the valid sessions.

    Diagnostics printed per mouse:
      expected probes (spreadsheet) vs found, sessions with Ephys subfolder,
      sessions where the tracks folder exists, and any missing probe files.
    """
    records: list[dict] = []

    available_roots = [r for r in DATA_ROOTS if r.is_dir()]
    if not available_roots:
        print(f"  ERROR: none of DATA_ROOTS exist: {[str(r) for r in DATA_ROOTS]}")
        return records
    print(f"  Searching {len(available_roots)} data root(s): "
          f"{[r.name for r in available_roots]}")

    for mouse, group in valid_df.groupby('mouse_name'):
        n_expected         = len(group)
        found_probes:   list[str] = []
        missing_probes: list[str] = []
        n_ephys   = 0
        n_tracks  = 0

        for data_root in available_roots:
            mouse_dir = data_root / mouse
            if not mouse_dir.is_dir():
                continue
            try:
                session_dirs = [p for p in mouse_dir.iterdir() if p.is_dir()]
            except PermissionError:
                print(f"    [{mouse}] PermissionError reading {mouse_dir}")
                continue

            for sess_dir in session_dirs:
                name  = sess_dir.name
                parts = name.split('_')
                if len(parts) < 2:
                    continue

                # Folder date is YYYYMMDD; normalise to YYYY-MM-DD to match spreadsheet
                raw_date = parts[1]
                if len(raw_date) != 8 or not raw_date.isdigit():
                    continue
                date_part = f"{raw_date[0:4]}-{raw_date[4:6]}-{raw_date[6:8]}"

                if date_part not in group['date'].values:
                    continue

                if not _has_ephys_subfolder(sess_dir):
                    continue
                n_ephys += 1

                tracks_path = (sess_dir / 'Anatomy' / 'fused' / 'registered'
                               / 'segmentation' / 'atlas_space' / 'tracks')
                if not tracks_path.is_dir():
                    continue
                n_tracks += 1

                sess_rows = group[group['date'] == date_part]
                for _, row in sess_rows.iterrows():
                    probe_id  = str(row['probe_id'])
                    label     = f"{name}/{probe_id}"

                    # Check for probe file existence before committing
                    file_found = False
                    for suffix in ('.npy', '.csv', '.txt'):
                        if (tracks_path / f'{probe_id}{suffix}').exists():
                            file_found = True
                            break
                    if not file_found:
                        try:
                            file_found = any(
                                probe_id.lower() in f.stem.lower()
                                for f in tracks_path.iterdir()
                            )
                        except Exception:
                            pass

                    if file_found:
                        found_probes.append(label)
                        records.append({
                            'mouse':            mouse,
                            'data_root':        str(data_root.name),
                            'session_dir':      sess_dir,
                            'session_name':     name,
                            'probe_id':         probe_id,
                            'target_region':    str(row['target_area']),
                            'day_of_recording': str(row['day_of_recording']),
                            'tracks_path':      tracks_path,
                        })
                    else:
                        missing_probes.append(label)

        # ── per-mouse diagnostic ──────────────────────────────────────────────
        status = "✓" if len(found_probes) == n_expected else \
                 f"⚠  {len(found_probes)}/{n_expected}"
        print(f"  [{mouse}] {status}  "
              f"sessions(Ephys/tracks): {n_ephys}/{n_tracks}  "
              f"probes loaded: {len(found_probes)}")
        if missing_probes:
            print(f"    ✗ No track file found: {missing_probes}")
        if len(found_probes) == 0 and n_expected > 0:
            exp_dates = sorted(group['date'].unique().tolist())
            print(f"    Expected dates (spreadsheet): {exp_dates}")
            for data_root in available_roots:
                mouse_dir = data_root / mouse
                if mouse_dir.is_dir():
                    try:
                        existing = sorted(
                            [p.name for p in mouse_dir.iterdir() if p.is_dir()]
                        )[:6]
                        print(f"    Dirs in {data_root.name}/{mouse}: {existing}")
                    except Exception:
                        pass

    total_mice = len({r['mouse'] for r in records})
    print(f"\n  Total: {len(records)} probe/session entries across "
          f"{total_mice} mice.")
    return records


# ── track loading ─────────────────────────────────────────────────────────────

def _load_one_track(record: dict) -> list[dict]:
    """
    Load one probe track file.  Returns rows with both ASR-µm coords (for
    alignment with BrainGlobe meshes) and bregma-relative µm (for hover labels).

    Track file: rows of [AP_µm, ML_µm, DV_µm] in raw CCF voxel space.
    """
    tracks_path: Path = record['tracks_path']
    probe_id: str     = record['probe_id']

    candidate: Path | None = None
    for suffix in ('.csv', '.npy', '.txt'):
        p = tracks_path / f'{probe_id}{suffix}'
        if p.exists():
            candidate = p
            break
    if candidate is None:
        for f in tracks_path.iterdir():
            if probe_id.lower() in f.stem.lower():
                candidate = f
                break
    if candidate is None:
        return []

    try:
        arr = np.load(candidate) if candidate.suffix == '.npy' \
              else np.loadtxt(candidate, delimiter=',')
    except Exception:
        return []

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 3:
        return []

    ap_ccf = arr[:, 0].astype(np.float32)   # col 0 = AP
    dv_ccf = arr[:, 1].astype(np.float32)   # col 1 = DV  ← was wrongly called ml_ccf
    ml_ccf = arr[:, 2].astype(np.float32)   # col 2 = ML  ← was wrongly called dv_ccf

    # Mirror right-hemisphere probes to left hemisphere
    ml_bregma = ml_ccf - BREGMA_ML_UM
    if float(np.median(ml_bregma)) > 0:
        ml_ccf = 2 * BREGMA_ML_UM - ml_ccf   # mirror across midline

    # ASR µm for atlas-space overlay (AP, DV, ML → A, S, R)
    a_um, s_um, r_um = ccf_to_asr(ap_ccf, dv_ccf, ml_ccf)

    # Bregma-relative for hover labels
    ap_b, ml_b, dv_b = bregma_relative(ap_ccf, ml_ccf, dv_ccf)

    rows: list[dict] = []
    for i in range(len(a_um)):
        rows.append({
            'mouse_id':      record['mouse'],       # allen_utils expects 'mouse_id'
            'mouse':         record['mouse'],        # kept for our own filtering
            'session':       record['session_name'],
            'probe_id':      probe_id,
            'target_region': record['target_region'],
            'day_of_recording': record.get('day_of_recording', 'unknown'),
            # Raw CCF voxel-µm — required by process_allen_labels
            'ccf_ap':        float(ap_ccf[i]),
            'ccf_ml':        float(ml_ccf[i]),
            'ccf_dv':        float(dv_ccf[i]),
            # atlas-space (ASR µm) — used for 3D scatter position
            'a_um':          float(a_um[i]),
            's_um':          float(s_um[i]),
            'r_um':          float(r_um[i]),
            # bregma-relative µm — used for hover labels only
            'ap_bregma':     float(ap_b[i]),
            'ml_bregma':     float(ml_b[i]),
            'dv_bregma':     float(dv_b[i]),
            # placeholders filled by assign_and_process_labels
            'ccf_atlas_acronym':        'unknown',
            'ccf_atlas_parent_acronym': 'unknown',
            'area_acronym_custom':       'Unassigned',
        })
    return rows


def collect_coordinates(records: list[dict], max_workers: int = 8) -> pd.DataFrame:
    all_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_load_one_track, rec): rec for rec in records}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc='Loading tracks'):
            all_rows.extend(fut.result())
    df = pd.DataFrame(all_rows)
    print(f"  Loaded {len(df):,} track points from "
          f"{df['session'].nunique()} sessions.")
    return df


def _load_allen_utils():
    """Load allen_utils module from ALLEN_UTILS_DIR. Returns the module or None."""
    import importlib.util, sys
    module_path = ALLEN_UTILS_DIR / 'allen_utils.py'
    if not module_path.exists():
        print(f"  Warning: allen_utils.py not found at {module_path}")
        return None
    try:
        spec = importlib.util.spec_from_file_location('allen_utils', module_path)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules['allen_utils'] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        print(f"  Warning: could not load allen_utils: {e}")
        return None


def assign_and_process_labels(df: pd.DataFrame, atlas) -> pd.DataFrame:
    """
    Two-step label assignment:

    Step 1 — BrainGlobe atlas lookup (vectorised):
      For each track point, look up ccf_atlas_acronym and ccf_atlas_parent_acronym
      from atlas.annotation using the raw CCF voxel-µm coordinates.

    Step 2 — allen_utils.process_allen_labels:
      Passes the labelled DataFrame through process_allen_labels which:
        - removes excluded areas (get_excluded_areas)
        - creates area_acronym_custom (simplified nomenclature)
        - creates layer_number and ccf_acronym_no_layer columns
        - runs create_bregma_centric_coords_from_ccf for bregma coords

    Points removed by process_allen_labels (excluded areas, fibre tracts, etc.)
    are kept in the output but their area_acronym_custom is set to 'Unassigned'.
    """
    if df.empty:
        return df

    df = df.copy()

    # ── Step 1: atlas acronym lookup ─────────────────────────────────────────
    res    = atlas.resolution[0]
    ap_vox = (df['a_um'].values / res).astype(int)
    dv_vox = (df['s_um'].values / res).astype(int)
    ml_vox = (df['r_um'].values / res).astype(int)

    shape = atlas.annotation.shape
    valid = (
        (ap_vox >= 0) & (ap_vox < shape[0]) &
        (dv_vox >= 0) & (dv_vox < shape[1]) &
        (ml_vox >= 0) & (ml_vox < shape[2])
    )

    region_ids = np.zeros(len(df), dtype=np.int32)
    region_ids[valid] = atlas.annotation[ap_vox[valid], dv_vox[valid], ml_vox[valid]]

    # Build id → (acronym, parent_acronym) map from lookup_df once
    # lookup_df has columns: id, acronym, name, structure_id_path (and parent via path)
    id_to_acr: dict[int, str]    = {}
    id_to_parent: dict[int, str] = {}
    for _, row in atlas.lookup_df.iterrows():
        rid = int(row['id'])
        id_to_acr[rid] = row['acronym']
    # Parent: second-to-last element of structure_id_path for each id
    try:
        for _, row in atlas.structures_df.iterrows() if hasattr(atlas, 'structures_df') else []:
            rid = int(row['id'])
            path = row.get('structure_id_path', '')
            parts = [p for p in str(path).split('/') if p]
            parent_id = int(parts[-2]) if len(parts) >= 2 else rid
            id_to_parent[rid] = id_to_acr.get(parent_id, id_to_acr.get(rid, 'unknown'))
    except Exception:
        pass  # fallback: use acronym itself as parent

    acronyms = [id_to_acr.get(int(rid), 'unknown') for rid in region_ids]
    parents  = [id_to_parent.get(int(rid), acr)
                for rid, acr in zip(region_ids, acronyms)]

    df['ccf_atlas_acronym']        = acronyms
    df['ccf_atlas_parent_acronym'] = parents

    # ── Step 2: process_allen_labels ─────────────────────────────────────────
    allen = _load_allen_utils()
    if allen is None:
        print("  Skipping process_allen_labels — allen_utils unavailable.")
        df['area_acronym_custom'] = df['ccf_atlas_acronym']
        return df

    print("  Running process_allen_labels …")
    try:
        df_processed = allen.process_allen_labels(df, subdivide_areas=False)
        # process_allen_labels drops excluded rows; re-merge on index so we keep
        # all track points and just mark dropped ones as Unassigned.
        df['area_acronym_custom'] = 'Unassigned'
        df.loc[df_processed.index, 'area_acronym_custom'] = df_processed['area_acronym_custom']
        # Also bring back bregma coords computed by process_allen_labels
        for col in ('ap', 'ml', 'dv'):
            if col in df_processed.columns:
                df[col] = np.nan
                df.loc[df_processed.index, col] = df_processed[col]
        if 'layer_number' in df_processed.columns:
            df['layer_number'] = np.nan
            df.loc[df_processed.index, 'layer_number'] = df_processed['layer_number']
    except Exception as e:
        print(f"  Warning: process_allen_labels failed ({e}); "
              "falling back to raw ccf_atlas_acronym.")
        df['area_acronym_custom'] = df['ccf_atlas_acronym']

    return df


# ── color scheme ──────────────────────────────────────────────────────────────

_UNASSIGNED_COLOR = '#CCCCCC'
_ALLEN_UTILS_COLOR_CACHE: dict | None = None   # loaded once, reused


def _get_allen_area_colors() -> dict[str, str]:
    """
    Load area→hex color from allen_utils.get_custom_area_color_per_group().
    Returns empty dict if allen_utils is unavailable.
    Result is cached so the module is only loaded once per process.
    """
    global _ALLEN_UTILS_COLOR_CACHE
    if _ALLEN_UTILS_COLOR_CACHE is not None:
        return _ALLEN_UTILS_COLOR_CACHE
    allen = _load_allen_utils()
    if allen is None:
        _ALLEN_UTILS_COLOR_CACHE = {}
        return {}
    try:
        area_color_dict, _ = allen.get_custom_area_color_per_group()
        # Values from allen_utils are already hex strings like '#1f9d5a'
        _ALLEN_UTILS_COLOR_CACHE = {k: v for k, v in area_color_dict.items()}
    except Exception as e:
        print(f"  Warning: could not load allen area colors ({e})")
        _ALLEN_UTILS_COLOR_CACHE = {}
    return _ALLEN_UTILS_COLOR_CACHE


def _get_allen_target_region_colors() -> dict[str, str]:
    """
    Colors for target_region mode — use get_custom_area_groups_colors() from
    allen_utils, which gives one color per broad group name.  We map each
    target_region label to its group color.  Unknown labels get grey.
    """
    allen = _load_allen_utils()
    if allen is None:
        return {}
    try:
        group_colors  = allen.get_custom_area_groups_colors()    # group → hex
        area_to_group = allen.get_custom_area_groups_from_name() # area  → group
        return {area: group_colors.get(group, _UNASSIGNED_COLOR)
                for area, group in area_to_group.items()}
    except Exception as e:
        print(f"  Warning: could not load target region colors ({e})")
        return {}


def build_color_scheme(df: pd.DataFrame,
                       color_by: str) -> tuple[str, list[str], dict[str, str]]:
    """
    Returns (column_name, ordered_labels, color_dict).

    color_by options:
      'area_acronym_custom'  — allen_utils simplified nomenclature
      'ccf_atlas_acronym'    — raw CCF atlas acronym
      'target_region'        — probe target from spreadsheet
    """

    # ── area_acronym_custom ──────────────────────────────────────────────────
    if color_by == 'area_acronym_custom':
        col    = 'area_acronym_custom'
        allen_colors = _get_allen_area_colors()
        counts = df[col].value_counts()
        # Order follows allen_utils.get_custom_area_order() if available,
        # otherwise by frequency; Unassigned always last.
        allen = _load_allen_utils()
        canonical_order: list[str] = []
        if allen is not None:
            try:
                canonical_order = allen.get_custom_area_order()
            except Exception:
                pass
        seen: set[str] = set()
        ordered: list[str] = []
        for area in canonical_order:
            if area in df[col].values and area not in seen and area != 'Unassigned':
                ordered.append(area)
                seen.add(area)
        for area in counts.index:
            if area not in seen and area != 'Unassigned':
                ordered.append(area)
                seen.add(area)
        if 'Unassigned' in df[col].values:
            ordered.append('Unassigned')
        # Build color dict: use allen_utils colors where available
        colors: dict[str, str] = {}
        palette = sns.color_palette('tab20', n_colors=max(len(ordered), 1))
        for i, area in enumerate(ordered):
            if area == 'Unassigned':
                colors[area] = _UNASSIGNED_COLOR
            elif area in allen_colors:
                colors[area] = allen_colors[area]
            else:
                r, g, b = palette[i % len(palette)]
                colors[area] = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
        return col, ordered, colors

    # ── ccf_atlas_acronym ────────────────────────────────────────────────────
    if color_by == 'ccf_atlas_acronym':
        col    = 'ccf_atlas_acronym'
        counts = df[col].value_counts()
        seen2: set[str] = set()
        named: list[str] = []
        for lbl in counts.index:
            if lbl not in _AMBIGUOUS and lbl not in seen2:
                named.append(lbl)
                seen2.add(lbl)
        last = [lbl for lbl in counts.index if lbl in _AMBIGUOUS and lbl not in seen2]
        ordered2 = named + last
        palette2 = sns.color_palette('tab20', n_colors=max(len(named), 1))
        colors2: dict[str, str] = {
            lbl: f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
            for lbl, (r, g, b) in zip(named, palette2)
        }
        for lbl in last:
            colors2[lbl] = _UNASSIGNED_COLOR
        return col, ordered2, colors2

    # ── target_region ────────────────────────────────────────────────────────
    if color_by == 'target_region':
        col = 'target_region'
        tgt_colors = _get_allen_target_region_colors()
        # Ordered by get_target_region_order() from allen_utils
        allen = _load_allen_utils()
        canonical: list[str] = []
        if allen is not None:
            try:
                canonical = allen.get_target_region_order()
            except Exception:
                pass
        seen3: set[str] = set()
        ordered3: list[str] = []
        for t in canonical:
            if t in df[col].values and t not in seen3:
                ordered3.append(t)
                seen3.add(t)
        for t in df[col].value_counts().index:
            if t not in seen3:
                ordered3.append(t)
                seen3.add(t)
        colors3: dict[str, str] = {}
        palette3 = sns.color_palette('husl', n_colors=max(len(ordered3), 1))
        for i, t in enumerate(ordered3):
            if t in tgt_colors:
                colors3[t] = tgt_colors[t]
            elif t in ('unknown', 'Unassigned', 'nan', 'None'):
                colors3[t] = _UNASSIGNED_COLOR
            else:
                r, g, b = palette3[i % len(palette3)]
                colors3[t] = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
        return col, ordered3, colors3

    raise ValueError(f'Unknown color_by={color_by!r}')


# ── plotly figure ─────────────────────────────────────────────────────────────

def make_plotly_figure(df: pd.DataFrame,
                       region_meshes: dict[str, dict],
                       color_by: str = 'region') -> go.Figure:
    """
    Build a fully interactive Plotly figure:

    3D scene
      ● Brain outline (root mesh) — always visible, grey, very transparent
      ● Sub-region meshes — hidden by default, togglable via legend
      ● Track points — coloured by region or CCF acronym

    Dropdowns (enlarged buttons)
      ▸ Mouse   — show/hide track points by mouse
      ▸ Session — show/hide track points by session
      ▸ Area    — show/hide track points by area label (deduplicated)

    Legend: one entry per unique colour label (no duplicates)
    Rotation: unrestricted (no up-vector constraint)
    Grid: background panes shown, tick labels and tick marks removed
    """
    col, ordered, colors = build_color_scheme(df, color_by)
    traces: list[go.BaseTraceType] = []

    # ── 1. Brain region surface meshes ────────────────────────────────────────
    # 'root' → wireframe outline (Scatter3d lines, always visible, always crisp).
    # Sub-regions → filled Mesh3d, hidden by default (toggle via legend).
    for acr, m in region_meshes.items():
        verts = m['verts']    # (N,3) ASR µm — axes: (AP, DV, ML)
        faces = m['faces']    # (M,3)
        alpha = m['alpha']
        color = m['color']

        is_root = (acr == 'root')

        if is_root:
            # ── Wireframe: collect every unique edge, draw as one Scatter3d
            #    with None-separated segments so it is a single efficient trace.
            edges: set[tuple[int, int]] = set()
            for tri in faces:
                for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                    edges.add((min(int(a), int(b)), max(int(a), int(b))))

            xs, ys, zs = [], [], []
            for a, b in edges:
                # verts: col0=AP, col1=DV, col2=ML  →  plot x=ML, y=AP, z=DV
                xs += [float(verts[a, 2]), float(verts[b, 2]), None]
                ys += [float(verts[a, 0]), float(verts[b, 0]), None]
                zs += [float(verts[a, 1]), float(verts[b, 1]), None]

            traces.append(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode='lines',
                line=dict(color='rgba(120,120,120,0.30)', width=1),
                name='brain outline',
                legendgroup='mesh_root',
                showlegend=False,
                visible=True,
                hoverinfo='skip',
                meta={'kind': 'mesh', 'acr': 'root'},
            ))
        else:
            # Sub-region filled mesh — hidden initially, togglable via legend
            if color == 'lightgrey':
                fill_color = f'rgba(211,211,211,{alpha})'
            elif color.startswith('#'):
                fill_color = _hex_to_rgba(color, alpha)
            else:
                fill_color = color

            traces.append(go.Mesh3d(
                x=verts[:, 2],   # ML
                y=verts[:, 0],   # AP
                z=verts[:, 1],   # DV
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color=fill_color,
                opacity=alpha,
                flatshading=False,
                lighting=dict(ambient=0.9, diffuse=0.3, roughness=0.8),
                lightposition=dict(x=100000, y=100000, z=100000),
                showscale=False,
                name=acr,
                legendgroup=f'mesh_{acr}',
                showlegend=True,
                visible='legendonly',
                hoverinfo='name',
                meta={'kind': 'mesh', 'acr': acr},
            ))

    # ── 2. Track point scatter — one trace per (label × mouse × session) ──────
    # For target_region mode: apply_target_region_filters per label so only
    # anatomically matching points get the label's colour; unmatched points
    # for that target are rendered as a dimmed grey background trace.
    is_target_mode = (color_by == 'target_region')
    allen_mod = _load_allen_utils() if is_target_mode else None

    legend_shown: set[str] = set()

    for label in ordered:
        is_bg    = label in _AMBIGUOUS or label in ('unknown', 'Unassigned', 'nan')
        pt_color = colors.get(label, _UNASSIGNED_COLOR)
        opacity  = 0.10 if is_bg else 0.75
        size     = 1.5  if is_bg else 3.0

        # In target_region mode, determine which rows actually match this target
        # via apply_target_region_filters. All other rows are grey background.
        if is_target_mode and not is_bg and allen_mod is not None:
            sub_all = df[df[col] == label].copy()
            if sub_all.empty:
                continue
            try:
                # apply_target_region_filters needs area_acronym_custom column
                sub_matched = allen_mod.apply_target_region_filters(sub_all, label)
                if sub_matched is None:
                    sub_matched = pd.DataFrame()
                matched_idx = set(sub_matched.index)
            except Exception:
                matched_idx = set(sub_all.index)
            sub_unmatched = sub_all.loc[sub_all.index.difference(matched_idx)]
            sub_colored   = sub_all.loc[list(matched_idx)]
        else:
            sub_colored   = df[df[col] == label]
            sub_unmatched = pd.DataFrame()

        if sub_colored.empty and sub_unmatched.empty:
            continue

        # Helper to add one Scatter3d trace
        def _add_trace(sub_df, trace_color, trace_opacity, trace_size,
                       trace_label, show_in_legend):
            if sub_df.empty:
                return
            traces.append(go.Scatter3d(
                x=sub_df['r_um'],
                y=sub_df['a_um'],
                z=sub_df['s_um'],
                mode='markers',
                marker=dict(size=trace_size, color=trace_color,
                            opacity=trace_opacity, line=dict(width=0)),
                name=trace_label,
                legendgroup=f'pts_{trace_label}',
                showlegend=show_in_legend,
                customdata=np.column_stack([
                    sub_df['mouse'],
                    sub_df['session'],
                    sub_df['probe_id'],
                    sub_df[col],
                    sub_df['ap_bregma'].round(0),
                    sub_df['ml_bregma'].round(0),
                    sub_df['dv_bregma'].round(0),
                ]),
                hovertemplate=(
                    '<b>%{customdata[0]}  /  %{customdata[1]}</b><br>'
                    'Probe: %{customdata[2]}<br>'
                    '%{customdata[3]}<br>'
                    'AP=%{customdata[4]:.0f} µm  '
                    'ML=%{customdata[5]:.0f} µm  '
                    'DV=%{customdata[6]:.0f} µm  (bregma-relative)<br>'
                    '<extra></extra>'
                ),
                visible=True,
                meta={'kind': 'scatter', 'mouse': str(sub_df['mouse'].iloc[0]),
                      'session': str(sub_df['session'].iloc[0]), 'label': trace_label},
            ))

        # Coloured (matched) points — split by (mouse, session) for dropdown filtering
        for (mouse, session), sub in sub_colored.groupby(['mouse', 'session']):
            first = label not in legend_shown
            if first:
                legend_shown.add(label)
            traces.append(go.Scatter3d(
                x=sub['r_um'], y=sub['a_um'], z=sub['s_um'],
                mode='markers',
                marker=dict(size=size, color=pt_color,
                            opacity=opacity, line=dict(width=0)),
                name=label,
                legendgroup=f'pts_{label}',
                showlegend=first,
                customdata=np.column_stack([
                    sub['mouse'], sub['session'], sub['probe_id'],
                    sub[col],
                    sub['ap_bregma'].round(0),
                    sub['ml_bregma'].round(0),
                    sub['dv_bregma'].round(0),
                ]),
                hovertemplate=(
                    '<b>%{customdata[0]}  /  %{customdata[1]}</b><br>'
                    'Probe: %{customdata[2]}<br>'
                    '%{customdata[3]}<br>'
                    'AP=%{customdata[4]:.0f} µm  '
                    'ML=%{customdata[5]:.0f} µm  '
                    'DV=%{customdata[6]:.0f} µm  (bregma-relative)<br>'
                    '<extra></extra>'
                ),
                visible=True,
                meta={'kind': 'scatter', 'mouse': mouse,
                      'session': session, 'label': label},
            ))

        # Unmatched points for this target (greyed out, smaller, same session grouping)
        if not sub_unmatched.empty:
            for (mouse, session), sub in sub_unmatched.groupby(['mouse', 'session']):
                traces.append(go.Scatter3d(
                    x=sub['r_um'], y=sub['a_um'], z=sub['s_um'],
                    mode='markers',
                    marker=dict(size=1.5, color='#bbbbbb',
                                opacity=0.15, line=dict(width=0)),
                    name=f'{label} (other)',
                    legendgroup=f'pts_{label}_other',
                    showlegend=False,
                    hoverinfo='skip',
                    visible=True,
                    meta={'kind': 'scatter', 'mouse': mouse,
                          'session': session, 'label': label},
                ))

    fig = go.Figure(data=traces)

    # ── 3. Nomenclature toggle — three flat radio buttons at top ─────────────
    # Each button switches which set of scatter traces is visible by matching
    # the legendgroup prefix: 'pts_{label}' traces carry a 'nomenclature' meta
    # key set to color_by so the other two figures' buttons filter correctly.
    # Since this function is called once per color_by, we just add a visual
    # reminder strip showing which nomenclature is active.  The three separate
    # HTML files serve as the "toggle" — one per nomenclature.  We add a small
    # annotation bar so the viewer knows which view they are looking at.
    nomenclature_labels = {
        'target_region':       'Target region',
        'area_acronym_custom': 'Area (custom)',
        'ccf_atlas_acronym':   'CCF atlas acronym',
    }
    active_label = nomenclature_labels.get(color_by, color_by)

    # ── 4. Shared axis style — no ticks, no tick labels, subtle grid ──────────
    _ax = dict(
        backgroundcolor='#f5f5f5',
        gridcolor='#d8d8d8',
        showbackground=True,
        showticklabels=False,
        ticks='',
        showspikes=False,
    )

    # ── 5. Layout ─────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=(f'Probe tracks  ·  BrainGlobe {ATLAS_NAME}  ·  '
                  f'<b>{active_label}</b>'),
            font=dict(size=13, family='Arial'),
            y=0.88,   # lowered from 0.97
            x=0.01,
            xanchor='left',
        ),
        scene=dict(
            xaxis=dict(title='ML (µm)',  **_ax),
            yaxis=dict(title='AP (µm)', **_ax),
            zaxis=dict(title='DV (µm)', **_ax),
            aspectmode='data',
            camera=dict(eye=dict(x=1.4, y=-1.6, z=0.7)),
            bgcolor='white',
            dragmode='orbit',
        ),
        legend=dict(
            itemsizing='constant',
            font=dict(size=10),
            tracegroupgap=4,
            x=1.01, xanchor='left',
            y=0.98, yanchor='top',
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='#ccc',
            borderwidth=1,
        ),
        margin=dict(l=0, r=220, t=60, b=0),
        paper_bgcolor='white',
        plot_bgcolor='white',
    )
    return fig


# ── save ──────────────────────────────────────────────────────────────────────

def save_html(fig: go.Figure, stem: str) -> None:
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    p = LOCAL_OUT / f'{stem}.html'
    fig.write_html(str(p), include_plotlyjs='cdn')
    print(f'  Saved: {p}')
    if SAVE_PATH.parent.exists():
        SAVE_PATH.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(SAVE_PATH / f'{stem}.html'), include_plotlyjs='cdn')


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    from brainglobe_atlasapi import BrainGlobeAtlas

    print('─' * 64)
    print('1 / 5  Loading valid sessions from spreadsheet …')
    valid_df = load_valid_sessions(INSERTION_INFO_XLSX)

    print('─' * 64)
    print('2 / 5  Discovering track files …')
    records = discover_tracks(valid_df)
    if not records:
        print('  No valid records found — check DATA_ROOTS and spreadsheet.')
        return

    print('─' * 64)
    print('3 / 5  Loading track coordinates (parallel) …')
    df_raw = collect_coordinates(records)
    if df_raw.empty:
        print('  No track data loaded — check file format.')
        return

    print('─' * 64)
    print(f'4 / 5  Loading BrainGlobe atlas ({ATLAS_NAME}) …')
    print('       (will auto-download ~500 MB on first run)')
    atlas = BrainGlobeAtlas(ATLAS_NAME, check_latest=False)
    print(f'  Atlas resolution: {atlas.resolution} µm,  '
          f'orientation: {atlas.orientation},  '
          f'shape: {atlas.shape}')

    print('  Assigning CCF regions and running process_allen_labels …')
    df = assign_and_process_labels(df_raw, atlas)

    print('  Loading region meshes …')
    region_meshes = load_region_meshes(atlas)

    print('─' * 64)
    print('5 / 5  Generating Plotly figures …')
    for color_by in ('area_acronym_custom', 'ccf_atlas_acronym', 'target_region'):
        stem = f'unit_3d_brain_brainrender_{color_by}'
        print(f'\n  [{color_by}]')
        fig = make_plotly_figure(df, region_meshes, color_by=color_by)
        save_html(fig, stem)

    # ── summary ───────────────────────────────────────────────────────────────
    print('\n' + '─' * 64)
    included = sorted(df['mouse'].unique().tolist())
    print(f'Done.  {len(included)} mice included:\n')
    for m in included:
        sub = df[df['mouse'] == m]
        n_sess   = sub['session'].nunique()
        n_probes = sub['probe_id'].nunique()
        n_pts    = len(sub)
        print(f'  {m:<20s}  {n_sess} session(s)  '
              f'{n_probes} probe(s)  {n_pts:>7,} track points')


if __name__ == '__main__':
    main()
