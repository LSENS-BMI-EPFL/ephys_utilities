"""
Diagnostics for merge_hierarchy_columns_from_gao().

Two complementary checks:
  1. Spatial: do neurons cluster around their assigned column, and does the
     overall point cloud of neurons overlay sensibly on the column grid?
  2. Label consistency: does the assigned column's own Region match the
     neuron's own recorded area label? This is the strongest sanity check --
     coordinates can *look* fine on a scatter plot while still being subtly
     wrong (e.g. axis swap), but a systematic area mismatch will show up here.

Call `plot_column_assignment_diagnostics(df, hierarchy_df)` after running
merge_hierarchy_columns_from_gao(df). Produces one 2x2 figure:
  A) Top view      (ML vs AP) -- columns as area-colored crosses, neurons as
                     small dots, thin lines linking a random subsample of
                     neurons to their assigned column, mismatches in red.
  B) Side view      (AP vs DV), same encoding.
  C) Per-area match rate bar chart.
  D) Nearest-neighbor distance histogram.
"""
"""
Diagnostics for merge_hierarchy_columns_from_gao().

Two complementary checks:
  1. Spatial: do neurons cluster around their assigned column, and does the
     overall point cloud of neurons overlay sensibly on the column grid?
  2. Label consistency: does the assigned column's own Region match the
     neuron's own recorded area label? This is the strongest sanity check --
     coordinates can *look* fine on a scatter plot while still being subtly
     wrong (e.g. axis swap), but a systematic area mismatch will show up here.

Call `plot_column_assignment_diagnostics(df, hierarchy_df)` after running
merge_hierarchy_columns_from_gao(df). Produces one 2x3 figure:
  A) ML vs AP      -- columns as large crosses colored by their
                       cc_hierarchy_score_columns (cmasher viola, central 30%
                       of the colormap, blue=negative/vmin=-1, vmax=0.5),
                       neurons as small plain-gray dots (not colored by
                       area), thin lines linking a random subsample of
                       neurons to their assigned column (green=match,
                       red=mismatched area label).
  B) AP vs DV       -- same encoding.
  C) ML vs DV       -- same encoding.
  D) Flatmap (U, V) -- columns only (neurons have no flatmap coordinate),
                       colored the same way.
  E) Per-area match rate bar chart.
  F) Nearest-neighbor distance histogram.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import cmasher as cmr


def _central_colormap(cmap, frac=0.3, n=256):
    """Return a new colormap using only the central `frac` of `cmap`."""
    lo, hi = 0.5 - frac / 2, 0.5 + frac / 2
    colors = cmap(np.linspace(lo, hi, n))
    return LinearSegmentedColormap.from_list(f'{cmap.name}_central{int(frac*100)}', colors)


def plot_column_assignment_diagnostics(df, hierarchy_df, merge_key='Region',
                                        n_link_lines=150, random_state=0,
                                        figsize=(18, 10), score_col='cc_hierarchy_score_columns',
                                        vmin=-1, vmax=0.5):
    """
    :param df: neuron dataframe AFTER merge_hierarchy_columns_from_gao(),
        must contain [merge_key, 'ccf_atlas_ml', 'ccf_atlas_dv', 'ccf_atlas_ap',
        'nearest_gao_column_id'].
    :param hierarchy_df: output of load_process_hierarchy_columns_from_gao()
        (needs merge_key, 'ColumnID', 'col_ml', 'col_dv', 'col_ap', 'col_u',
        'col_v', score_col).
    :param merge_key: name of the area-label column (MERGE_KEY in your code).
    :param n_link_lines: how many neuron-to-column link lines to draw (subsampled,
        so the plots stay readable).
    :param score_col: column in hierarchy_df holding the hierarchy score used
        to color column centroids.
    :param vmin, vmax: color scale limits for the hierarchy score.
    """
    rng = np.random.default_rng(random_state)

    assigned = df[df['nearest_gao_column_id'].notna()].copy()
    assigned['nearest_gao_column_id'] = assigned['nearest_gao_column_id'].astype(int)

    # column area for each neuron's assigned column, to check label consistency
    col_area_lookup = hierarchy_df.set_index('ColumnID')[merge_key]
    assigned['assigned_column_area'] = assigned['nearest_gao_column_id'].map(col_area_lookup)
    assigned['match'] = assigned[merge_key] == assigned['assigned_column_area']

    match_rate = assigned['match'].mean()
    print(f"Overall area match rate: {match_rate:.1%} "
          f"({assigned['match'].sum()}/{len(assigned)} neurons)")

    # centroid color = hierarchy score, cmasher viola, central 30%, blue=negative
    cmap = _central_colormap(cmr.viola, frac=0.3)
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # --- A, B, C: all three CCF pairwise spatial overlays -------------------
    ccf_views = [
        (axes[0, 0], ('col_ml', 'col_ap'), 'ml', 'ap', 'ML vs AP'),
        (axes[0, 1], ('col_ap', 'col_dv'), 'ap', 'dv', 'AP vs DV'),
        (axes[0, 2], ('col_ml', 'col_dv'), 'ml', 'dv', 'ML vs DV'),
    ]
    for ax, (xcol, ycol), xdim, ydim, title in ccf_views:
        neuron_x, neuron_y = f'ccf_atlas_{xdim}', f'ccf_atlas_{ydim}'

        # neurons as small plain-gray dots (not colored by area/localization)
        ax.scatter(assigned[neuron_x], assigned[neuron_y], s=4, color='0.75',
                   alpha=0.4, zorder=1)

        # columns as large X markers, colored by hierarchy score
        ax.scatter(hierarchy_df[xcol], hierarchy_df[ycol], marker='x', s=70,
                   c=hierarchy_df[score_col], cmap=cmap, norm=norm,
                   linewidths=1.8, zorder=3)

        # link lines for a subsample: green = match, red = mismatch
        sample = assigned.sample(min(n_link_lines, len(assigned)), random_state=rng.integers(1e6))
        for _, row in sample.iterrows():
            col = hierarchy_df.loc[hierarchy_df['ColumnID'] == row['nearest_gao_column_id']].iloc[0]
            ax.plot([row[neuron_x], col[xcol]], [row[neuron_y], col[ycol]],
                    color='green' if row['match'] else 'red', linewidth=0.6,
                    alpha=0.6, zorder=2)

        ax.set_title(title)
        ax.set_xlabel(xdim)
        ax.set_ylabel(ydim)

    # --- D: flatmap (columns only -- neurons have no flatmap coordinate) ----
    ax = axes[1, 0]
    ax.scatter(hierarchy_df['col_u'], hierarchy_df['col_v'], marker='x', s=70,
               c=hierarchy_df[score_col], cmap=cmap, norm=norm, linewidths=1.8)
    ax.set_title('Flatmap')
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.invert_yaxis()  # flatmap V typically increases downward, adjust if needed
    fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.05,
                 label=score_col, aspect=30)

    # --- E: per-area match rate -------------------------------------------
    ax = axes[1, 1]
    rates = assigned.groupby(merge_key)['match'].mean().sort_values()
    ax.barh(rates.index, rates.values, color='steelblue')
    ax.axvline(match_rate, color='k', linestyle='--', linewidth=1,
               label=f'overall = {match_rate:.1%}')
    ax.set_xlabel('fraction of neurons whose nearest column\nhas the SAME area label')
    ax.set_title('Per-area label match rate')
    ax.legend(fontsize=8)
    ax.tick_params(axis='y', labelsize=5)

    # --- F: NN distance histogram ------------------------------------------
    ax = axes[1, 2]
    d = np.sqrt(
        (assigned['ccf_atlas_ml'] - assigned['nearest_gao_column_id'].map(hierarchy_df.set_index('ColumnID')['col_ml']))**2 +
        (assigned['ccf_atlas_dv'] - assigned['nearest_gao_column_id'].map(hierarchy_df.set_index('ColumnID')['col_dv']))**2 +
        (assigned['ccf_atlas_ap'] - assigned['nearest_gao_column_id'].map(hierarchy_df.set_index('ColumnID')['col_ap']))**2
    )
    ax.hist(d, bins=60, color='steelblue')
    ax.set_xlabel('Distance to assigned column centroid')
    ax.set_ylabel('# neurons')
    ax.set_title('Nearest-neighbor distance distribution')

    return fig


# --------------------------------------------------------------------------
# DEMO with synthetic neurons (for illustration only -- swap in your real df)
# --------------------------------------------------------------------------
if __name__ == '__main__':
    MERGE_KEY = 'Region'
    gao_raw = pd.read_excel('/mnt/user-data/uploads/Gao_cortical_columns_ROIs.xlsx')
    hierarchy_df = gao_raw.rename(columns={
        'X': 'col_ap', 'Y': 'col_dv', 'Z': 'col_ml', 'U': 'col_u', 'V': 'col_v',
        'HierarchyScore': 'cc_hierarchy_score_columns',
    })[[MERGE_KEY, 'ColumnID', 'cc_hierarchy_score_columns', 'col_ap', 'col_dv', 'col_ml',
        'col_u', 'col_v']]

    rng = np.random.default_rng(0)
    n_neurons_per_col = 8
    rows = []
    for _, col in hierarchy_df.iterrows():
        for _ in range(n_neurons_per_col):
            # most neurons scattered near their true column (realistic jitter),
            # a few given a big jump to simulate mismatches for the demo
            jump = rng.random() < 0.08
            scale = 120 if jump else 25
            rows.append({
                MERGE_KEY: col[MERGE_KEY],
                'ccf_atlas_ml': col['col_ml'] + rng.normal(0, scale),
                'ccf_atlas_dv': col['col_dv'] + rng.normal(0, scale),
                'ccf_atlas_ap': col['col_ap'] + rng.normal(0, scale),
            })
    synthetic_df = pd.DataFrame(rows)

    # run the actual assignment logic (mirrors merge_hierarchy_columns_from_gao)
    from scipy.spatial import cKDTree
    tree = cKDTree(hierarchy_df[['col_ml', 'col_dv', 'col_ap']].to_numpy())
    _, idx = tree.query(synthetic_df[['ccf_atlas_ml', 'ccf_atlas_dv', 'ccf_atlas_ap']].to_numpy(), k=1)
    synthetic_df['nearest_gao_column_id'] = hierarchy_df['ColumnID'].to_numpy()[idx]
    synthetic_df['cc_hierarchy_score_columns'] = hierarchy_df['cc_hierarchy_score_columns'].to_numpy()[idx]

    fig = plot_column_assignment_diagnostics(synthetic_df, hierarchy_df, merge_key=MERGE_KEY)
    fig.savefig('/mnt/user-data/outputs/column_assignment_diagnostics_demo.png', dpi=150)
    print('Saved demo figure.')