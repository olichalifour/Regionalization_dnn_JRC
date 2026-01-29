import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

def plot_catchment_error_map(df, output_dir=None,
                             true_col=None, pred_col=None,
                             cmap='coolwarm', figsize=(10,6)):
    """
    Plots spatial distribution of catchment prediction errors as a scatter map.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file produced by save_val_predictions().
    output_dir : str, optional
        Directory to save the figure. Defaults to same directory as csv_path.
    true_col : str, optional
        Name of the true variable column (e.g., 'true_burned_area').
        If None, the first column starting with 'true_' is used.
    pred_col : str, optional
        Name of the predicted variable column (e.g., 'pred_burned_area').
        If None, the first column starting with 'pred_' is used.
    cmap : str
        Diverging colormap for mean error.
    figsize : tuple
        Figure size in inches.
    """


    # Infer columns if not provided
    if true_col is None:
        true_col = [c for c in df.columns if c.startswith("true_")][0]
    if pred_col is None:
        pred_col = [c for c in df.columns if c.startswith("pred_")][0]

    if not {'lon', 'lat'}.issubset([c.lower() for c in df.columns]):
        raise ValueError("CSV must contain 'lon' and 'lat' columns.")


    # Compute error
    df["error"] = df[pred_col] - df[true_col]

    # Aggregate per catchment
    df_stats = df.groupby(["catchment_id", "lon", "lat"]).agg(
        mean_error=("error", "mean"),
        std_error=("error", "std"),
        n_obs=("error", "count")
    ).reset_index()

    cmap_mean = 'RdBu_r'
    cmap_std = 'viridis'

    fig_path = os.path.join(output_dir, "catchment_error_map.png")

    # === Compute symmetric color range for mean error ===
    max_abs_error = np.nanmax(np.abs(df_stats["mean_error"]))
    vmin, vmax = -max_abs_error, max_abs_error

    # === Create figure with two subplots ===
    fig, axes = plt.subplots(2, 1, figsize=figsize,
                             subplot_kw={'projection': ccrs.PlateCarree()})
    plt.subplots_adjust(hspace=0.15)

    # ---- 1. Mean Error ----
    ax1 = axes[0]
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax1.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax1.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)
    ax1.add_feature(cfeature.OCEAN, facecolor="white")

    sc1 = ax1.scatter(df_stats["lon"], df_stats["lat"],
                      c=df_stats["mean_error"],
                      s=30,
                      cmap=cmap_mean,
                      vmin=vmin, vmax=vmax,
                      transform=ccrs.PlateCarree(),
                      edgecolor="k", linewidth=0.3)

    cbar1 = plt.colorbar(sc1, ax=ax1, orientation="vertical", pad=0.02, shrink=0.9)
    cbar1.set_label("Mean Prediction Error (Pred - True)")
    ax1.set_title("Catchment Mean Prediction Error", fontsize=12)

    # ---- 2. Standard Deviation of Error ----
    ax2 = axes[1]
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax2.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax2.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)
    ax2.add_feature(cfeature.OCEAN, facecolor="white")

    sc2 = ax2.scatter(df_stats["lon"], df_stats["lat"],
                      c=df_stats["std_error"],
                      s=30,
                      cmap=cmap_std,
                      transform=ccrs.PlateCarree(),
                      edgecolor="k", linewidth=0.3)

    cbar2 = plt.colorbar(sc2, ax=ax2, orientation="vertical", pad=0.02, shrink=0.9)
    cbar2.set_label("Standard Deviation of Prediction Error")
    ax2.set_title("Catchment Error Standard Deviation", fontsize=12)

    # Global labels
    for ax in axes:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved catchment error map (mean + std) to: {fig_path}")
    return fig_path