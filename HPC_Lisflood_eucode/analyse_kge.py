# -*- coding: latin-1 -*-
"""
Compute KGE per catchment from Lisflood outputs, compare with predicted and
regionalized KGE, and produce all plots.

UPDATED:
- Recursive catchment discovery (continent/region/catchment_id)
- Dynamic RUN_NAME
- Uses geography_KGE_longrun
- Preserves ALL figures and outputs
"""

import os
import glob
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# -------------------------- USER CONFIG -----------------------------------
CSV_PATH = "optimized_params.csv"
REGIONAL_CSV = "/home/chaliol/experiment_results_HEADCATCHMENTS_noLakes_noRes_minKGE041_geography_geographyclimate.csv"

OUTPUT_BASE = "/home/chaliol/Lisflood_param_test_glofas5"
SOURCE = "/BGFS/DISASTER/russcar/cal_workflow_2025/catchments"

FIG_DIR_NAME = "figure_and_saving"

AUTO_LIMITS = False
CLASSIC_LIMITS = (-0.4, 1.0)
# -------------------------------------------------------------------------


# ---------------------- Robust IO helpers ---------------------------------

def safe_read_csv(path, **kwargs):
    encodings = ["utf-8", "utf-8-sig", "latin1"]
    last_exc = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except Exception as e:
            last_exc = e
    raise RuntimeError(f"Could not read CSV {path}: {last_exc}")

def safe_open_text(path, mode='r'):
    return open(path, mode, encoding='utf-8', errors='replace')

def safe_parse_xml(xml_path):
    try:
        return ET.parse(xml_path)
    except Exception:
        with open(xml_path, 'rb') as f:
            raw = f.read()
        text = raw.decode('utf-8', errors='replace')
        root = ET.fromstring(text)
        return ET.ElementTree(root)


# -------------------------- Catchment indexing ----------------------------

def index_catchments(source_root):
    """
    Index structure:
    source/continent/region/catchment_id/
    """
    index = {}

    for continent in os.listdir(source_root):
        cont_path = os.path.join(source_root, continent)
        if not os.path.isdir(cont_path):
            continue

        for region in os.listdir(cont_path):
            reg_path = os.path.join(cont_path, region)
            if not os.path.isdir(reg_path):
                continue

            for cid in os.listdir(reg_path):
                cid_path = os.path.join(reg_path, cid)
                if not os.path.isdir(cid_path):
                    continue
                try:
                    cid_int = int(cid)
                except ValueError:
                    continue

                index[cid_int] = {
                    "base_dir": cid_path,
                    "continent": continent,
                    "region": region
                }

    return index


# -------------------------- helper functions -------------------------------

def parse_tss(filepath):
    with safe_open_text(filepath, 'r') as f:
        lines = f.readlines()

    days, vals = [], []
    for ln in lines:
        p = ln.strip().split()
        if len(p) < 2:
            continue
        try:
            days.append(int(p[0]))
            vals.append(float(p[1]))
        except Exception:
            continue

    if not days:
        raise ValueError(f"No numeric data in {filepath}")

    return days, np.array(vals)


def read_xml_startdate(xml_path):
    tree = safe_parse_xml(xml_path)
    root = tree.getroot()

    for tv in root.findall('.//textvar'):
        name = tv.get('name') or tv.get('Name')
        if name == 'CalendarDayStart':
            val = tv.get('value') or tv.findtext('value')
            for fmt in ("%d/%m/%Y %H:%M", "%d/%m/%Y"):
                try:
                    return datetime.strptime(val, fmt)
                except Exception:
                    pass
            raise ValueError(f"Bad CalendarDayStart format in {xml_path}")

    raise ValueError(f"CalendarDayStart not found in {xml_path}")


def compute_kge(sim, obs):
    sim = np.asarray(sim, float)
    obs = np.asarray(obs, float)

    mask = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[mask], obs[mask]

    if sim.size < 2:
        return np.nan

    r = np.corrcoef(sim, obs)[0, 1]
    r = 0.0 if np.isnan(r) else r

    alpha = np.std(sim, ddof=1) / np.std(obs, ddof=1) if np.std(obs) != 0 else np.nan
    beta = np.mean(sim) / np.mean(obs) if np.mean(obs) != 0 else np.nan

    comps = [
        (r - 1) ** 2,
        (alpha - 1) ** 2 if not np.isnan(alpha) else 1e6,
        (beta - 1) ** 2 if not np.isnan(beta) else 1e6
    ]

    return float(1 - np.sqrt(sum(comps)))


# ---------------------- plotting helper -----------------------------------

def plot_scatter_density(x, y, savepath, title, xlabel, ylabel,
                         xlim=CLASSIC_LIMITS, ylim=CLASSIC_LIMITS,
                         auto_limits=AUTO_LIMITS):

    from sklearn.metrics import mean_absolute_error, r2_score
    from scipy.stats import pearsonr

    x, y = np.asarray(x), np.asarray(y)
    msk = np.isfinite(x) & np.isfinite(y)
    x, y = x[msk], y[msk]

    if x.size == 0:
        return

    if auto_limits:
        mn = min(x.min(), y.min())
        mx = max(x.max(), y.max())
        pad = (mx - mn) * 0.05
        xlim = ylim = (mn - pad, mx + pad)

    r = pearsonr(x, y)[0] if x.size > 2 else np.nan
    mae = mean_absolute_error(x, y)
    r2 = r2_score(x, y)

    m, b = np.polyfit(x, y, 1) if x.size > 2 else (np.nan, np.nan)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].scatter(x, y, s=20, alpha=0.6)
    axs[0].plot(xlim, xlim, '--',color = "red")
    if np.isfinite(m):
        axs[0].plot(xlim, [m * v + b for v in xlim],'-', color = 'orange')
    axs[0].annotate(f"n={len(x)}\nMAE={mae:.3f}\nR²={r2:.3f}\nr={r:.3f}",
                    (0.05, 0.95), xycoords='axes fraction',
                    va='top', bbox=dict(facecolor='white', alpha=0.7))
    axs[0].set(xlabel=xlabel, ylabel=ylabel, title=title, xlim=xlim, ylim=ylim)

    hb = axs[1].hexbin(x, y, gridsize=50, cmap='plasma', mincnt=1)
    fig.colorbar(hb, ax=axs[1], label='Density')
    axs[1].plot(xlim, xlim, '--',color="red")
    axs[1].set(xlabel=xlabel, ylabel=ylabel, title="Density",
               xlim=xlim, ylim=ylim)

    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.close(fig)

def build_normalized_cdf(values, xgrid):
    """
    Build a normalized empirical CDF.
    Returns values in [0, 1].
    """
    values_sorted = np.sort(values)
    cdf = np.searchsorted(values_sorted, xgrid, side="right")
    return cdf / len(values_sorted)

# ----------------------- MAIN ---------------------------------------------

def main():

    df_params = safe_read_csv(CSV_PATH)
    if 'catchment_id' not in df_params.columns:
        df_params = df_params.rename(columns={'ID': 'catchment_id'})
    df_params['catchment_id'] = df_params['catchment_id'].astype(int)

    df_reg = safe_read_csv(REGIONAL_CSV)
    cols = {c.lower(): c for c in df_reg.columns}
    if 'geography_kge_longrun' in cols:
        df_reg = df_reg.rename(
            columns={cols['geography_kge_longrun']: 'geography_KGE_longrun'}
        )

    id_col = next(c for c in df_reg.columns if c.lower() in ('id', 'catchment_id'))
    df_reg = df_reg.rename(columns={id_col: 'catchment_id'})
    df_reg['catchment_id'] = df_reg['catchment_id'].astype(int)

    print("Indexing catchments...")
    catchments = index_catchments(SOURCE)
    print(f"Found {len(catchments)} catchments")

    out_fig_dir = os.path.join(OUTPUT_BASE, FIG_DIR_NAME)
    os.makedirs(out_fig_dir, exist_ok=True)

    results = []

    for cid in df_params['catchment_id']:
        if cid not in catchments:
            results.append({'catchment_id': cid, 'kge_true': np.nan})
            continue

        info = catchments[cid]
        base = info['base_dir']
        cont, reg = info['continent'], info['region']

        out_dir = os.path.join(OUTPUT_BASE, str(cid), 'out')
        tss_files = glob.glob(os.path.join(out_dir, '*.tss'))
        if not tss_files:
            results.append({'catchment_id': cid, 'kge_true': np.nan})
            continue

        tss = next((f for f in tss_files if 'dis' in f.lower()), tss_files[0])
        days, sim_vals = parse_tss(tss)

        run_name = f"OSLisfloodGloFASv5calibration_v1_{cont}_{reg}Run0.xml"
        xml_path = os.path.join(OUTPUT_BASE, str(cid), 'settings', run_name)
        if not os.path.exists(xml_path):
            xmls = glob.glob(os.path.join(OUTPUT_BASE, str(cid), 'settings', '*.xml'))
            if not xmls:
                results.append({'catchment_id': cid, 'kge_true': np.nan})
                continue
            xml_path = xmls[0]

        start = read_xml_startdate(xml_path)
        dates = [start + timedelta(days=d - 1) for d in days]
        ser_sim = pd.Series(sim_vals, index=pd.to_datetime(dates))

        obs_path = os.path.join(base, 'station', 'observations.csv')
        df_obs = safe_read_csv(obs_path)
        df_obs.iloc[:, 0] = pd.to_datetime(df_obs.iloc[:, 0], errors='coerce')
        ser_obs = pd.Series(df_obs.iloc[:, 1].values, index=df_obs.iloc[:, 0])

        df = pd.concat([ser_sim, ser_obs], axis=1, join='inner')
        df.columns = ['sim', 'obs']

        kge = compute_kge(df['sim'], df['obs'])
        results.append({'catchment_id': cid, 'kge_true': kge})

    df_results = pd.DataFrame(results)
    df_join = (
        df_results
        .merge(df_params[['catchment_id', 'best_kge']], on='catchment_id', how='left')
        .merge(df_reg[['catchment_id', 'geography_KGE_longrun']], on='catchment_id', how='left')
    )

    df_join.to_csv(os.path.join(out_fig_dir, "kge_true_vs_predicted_table.csv"), index=False)

    plot_scatter_density(
        df_join['kge_true'], df_join['best_kge'],
        os.path.join(out_fig_dir, "kge_true_vs_bestkge_scatter_density.png"),
        "True vs Predicted KGE",
        "True KGE", "Predicted KGE"
    )

    plot_scatter_density(
        df_join['kge_true'], df_join['geography_KGE_longrun'],
        os.path.join(out_fig_dir, "kge_true_vs_regionalized_scatter_density.png"),
        "True vs Regionalized KGE",
        "True KGE", "Regionalized KGE"
    )

    # ----------------- FINAL NORMALIZED CDF COMPARISON PLOT -----------------

    print("\nGenerating normalized cumulative distribution comparison...")

    # ---- Extract values ----
    kge_true_vals = df_join['kge_true'].dropna().values
    kge_pred_vals = df_join['best_kge'].dropna().values

    has_reg = 'geography_KGE_longrun' in df_join.columns
    if has_reg:
        kge_reg_vals = df_join['geography_KGE_longrun'].dropna().values

    # ---- X-LIMIT SETTING ----
    if AUTO_LIMITS:
        mins = [kge_true_vals.min(), kge_pred_vals.min()]
        maxs = [kge_true_vals.max(), kge_pred_vals.max()]
        if has_reg:
            mins.append(kge_reg_vals.min())
            maxs.append(kge_reg_vals.max())

        xmin = min(mins)
        xmax = max(maxs)
        pad = (xmax - xmin) * 0.05 if xmax > xmin else 0.1
        xlim = (xmin - pad, xmax + pad)
    else:
        xlim = CLASSIC_LIMITS

    # ---- X-grid ----
    xgrid = np.linspace(xlim[0], xlim[1], 500)

    # ---- Build normalized CDFs ----
    cdf_true = build_normalized_cdf(kge_true_vals, xgrid)
    cdf_pred = build_normalized_cdf(kge_pred_vals, xgrid)
    if has_reg:
        cdf_reg = build_normalized_cdf(kge_reg_vals, xgrid)

    # ---- Plot ----
    plt.figure(figsize=(10, 6))

    plt.plot(xgrid, cdf_true, label='True with Ai Param KGE', linewidth=2)
    plt.plot(xgrid, cdf_pred, label='Predicted by AI KGE', linewidth=2)
    if has_reg:
        plt.plot(xgrid, cdf_reg, label='Regionalized GEOG KGE', linewidth=2)

    plt.xlabel("KGE", fontsize=12)
    plt.ylabel("Cumulative probability", fontsize=12)
    plt.title("Normalized Cumulative Distribution of KGE", fontsize=14)

    plt.xlim(xlim)
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)

    plt.legend(loc='upper left')

    # ---- Save ----
    out_cdf = os.path.join(out_fig_dir, "cdf_kge_comparison_normalized.png")
    plt.savefig(out_cdf, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved normalized CDF comparison figure at:\n  {out_cdf}")

    print("All figures generated successfully.")


if __name__ == "__main__":
    main()