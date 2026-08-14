"""
Plot avg_ipsi (left hemisphere) and avg_contra (right hemisphere) from Liu et al
onto the IBL Swanson flatmap, with log-scale colormap and portrait orientation
(olfactory bulbs on top).

Requires: iblatlas, cmasher, pandas, matplotlib, openpyxl

On first run, iblatlas will automatically download two data files
(swanson2allen.npz, swansonpaths.json) from IBL's S3 bucket (~few MB total).

────────────────────────────────────────────────────────────────────────────────
HOW HIERARCHY RESOLUTION WORKS
────────────────────────────────────────────────────────────────────────────────
The Excel file contains acronyms at mixed levels of the Allen CCF hierarchy
(e.g. 'TH' for thalamus alongside 'MOs' for secondary motor cortex).
IBL's `propagate_down()` resolves this as follows:

1. DOWNWARD FILL (parent → children):
   Each input acronym is looked up in the Allen structure tree and its value is
   propagated down to ALL of its descendants. Every Swanson region that is a
   child of an input region inherits that region's value.

2. CONFLICT RESOLUTION (multiple ancestors map to the same Swanson leaf):
   If a Swanson leaf region has more than one ancestor in the input (e.g. both
   'TH' and one of its sub-nuclei are present), all matching ancestor values are
   collected and resolved by **nanmedian**. For a single ancestor per leaf this
   is just that ancestor's value; for conflicts it is the median of competing
   values.

3. REGIONS NOT RESOLVED:
   - If an input acronym has no descendants in the Swanson flatmap, its value
     is silently dropped (19 such regions — see companion analysis).
   - Swanson regions with no ancestor in the input are shown in empty_color
     (light grey).

4. LOG SCALE:
   Values are log10-transformed before passing to plot_swanson_vector, which
   applies a linear norm internally. Values ≤ 0 are treated as NaN. The
   colorbar ticks are placed at powers of ten and labelled in linear units.

5. BILATERAL MAPPING (single call, no overpainting):
   avg_ipsi  → left  hemisphere (positive Allen IDs)
   avg_contra → right hemisphere (negative Allen IDs — IBL convention for RH)
   Both hemispheres are passed in one plot_swanson_vector call with
   hemisphere='both', avoiding the overpainting that occurs with two calls.

6. PORTRAIT / OB-ON-TOP:
   orientation='portrait' rotates the Swanson map so the olfactory bulbs
   appear at the top of the figure.
────────────────────────────────────────────────────────────────────────────────
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import cmasher as cmr
from iblatlas.regions import BrainRegions
from iblatlas.plots import plot_swanson_vector

# ── Load data ─────────────────────────────────────────────────────────────────
xlsx_path = pathlib.Path(__file__).parent / "Liu_et_al_Group_averages_ranked.xlsx"

df = pd.read_excel(xlsx_path)
data = df[["Unnamed: 0", "avg_ipsi", "avg_contra"]].iloc[2:].copy()
data.columns = ["acronym", "avg_ipsi", "avg_contra"]
data = data.dropna(subset=["acronym"])
data["avg_ipsi"]   = pd.to_numeric(data["avg_ipsi"],   errors="coerce")
data["avg_contra"] = pd.to_numeric(data["avg_contra"], errors="coerce")
data = data.dropna(subset=["avg_ipsi", "avg_contra"])

acronyms    = data["acronym"].values
vals_ipsi   = data["avg_ipsi"].values
vals_contra = data["avg_contra"].values

print(f"Loaded {len(acronyms)} regions")
print(f"  avg_ipsi   range: {np.nanmin(vals_ipsi):.4f} – {np.nanmax(vals_ipsi):.2f}")
print(f"  avg_contra range: {np.nanmin(vals_contra):.4f} – {np.nanmax(vals_contra):.2f}")

# ── Build lateralized ID arrays (IBL convention: negative ID = right hemisphere) ──
br = BrainRegions()
pos_ids = br.acronym2id(acronyms, mapping="Allen")   # left hemisphere
neg_ids = -pos_ids                                    # right hemisphere

all_ids  = np.concatenate([pos_ids,    neg_ids])
all_vals = np.concatenate([vals_ipsi,  vals_contra])

# ── Log10 transform ───────────────────────────────────────────────────────────
all_vals_log = np.where(all_vals > 0, np.log10(all_vals), np.nan)

vmin_log = np.nanmin(all_vals)
vmax_log = np.nanmax(all_vals)
print(f"  log10 range: {vmin_log:.3f} – {vmax_log:.3f}")

# ── Colormap: cmasher ember, truncated to [0.15, 0.85] ───────────────────────
truncated_cmap = mcolors.LinearSegmentedColormap.from_list(
    "ember_trunc",
    cmr.ember(np.linspace(0.05, 0.85, 256))
)
truncated_cmap = matplotlib.colormaps['jet']

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 14), facecolor="white")

plot_swanson_vector(
    acronyms=all_ids,
    values=all_vals,
    ax=ax,
    hemisphere="both",
    br=br,
    orientation="portrait",
    cmap=truncated_cmap,
    vmin=vmin_log,
    vmax=vmax_log,
    empty_color="#c8c8c8",
    show_cbar=False,
    linewidth=0.3,
    edgecolor="k",
)

ax.set_axis_off()

# ── Colorbar: ticks at powers of ten ─────────────────────────────────────────
norm = mcolors.Normalize(vmin=vmin_log, vmax=vmax_log)
sm   = cm.ScalarMappable(cmap=truncated_cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.03, pad=0.02)
cbar.set_label("Fluorescence intensity (a.u.)", fontsize=10)

# Place ticks at every integer power of 10 within the data range
p_min = int(np.floor(vmin_log))
p_max = int(np.ceil(vmax_log))
pow10_log  = [p for p in range(p_min, p_max + 1) if vmin_log <= p <= vmax_log]
pow10_labels = [f"$10^{{{p}}}$" if p != 0 else "1" for p in pow10_log]
cbar.set_ticks(pow10_log)
cbar.set_ticklabels(pow10_labels)

# ── Title & save ──────────────────────────────────────────────────────────────
ax.set_title(
    "Left: avg_ipsi   |   Right: avg_contra\n(log scale, Allen CCF hierarchy → Swanson flatmap)",
    fontsize=10, pad=8
)

fig.tight_layout()

out = pathlib.Path(__file__).parent / "data" /  "swanson_avg_ipsi_contra.png"
fig.savefig(out, dpi=500, bbox_inches="tight")

out = pathlib.Path(__file__).parent / "data" /"swanson_avg_ipsi_contra.pdf"
fig.savefig(out, dpi=500, bbox_inches="tight")

out = pathlib.Path(__file__).parent / "data" /"swanson_avg_ipsi_contra.eps"
fig.savefig(out, dpi=500, bbox_inches="tight")

out = pathlib.Path(__file__).parent / "data" / "swanson_avg_ipsi_contra.svg"
fig.savefig(out, dpi=500, bbox_inches="tight")

print(f"Saved: {out}")
plt.show()