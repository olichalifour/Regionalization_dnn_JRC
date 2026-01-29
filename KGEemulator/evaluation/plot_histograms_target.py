import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use("Agg")
def plot_target_histograms(y_df, output_dir="plots", bins=50, log=False):
    """
    Plot histograms for each target variable in a 7x2 grid.

    Args:
        y_df (pd.DataFrame): DataFrame of targets (columns = target names).
        output_dir (str): directory where the plot will be saved.
        bins (int): number of bins in the histogram.
        log (bool): whether to use log scale on the y-axis.
    """
    hist_dir = os.path.join(output_dir, "target_histograms")
    os.makedirs(hist_dir, exist_ok=True)

    n_targets = y_df.shape[1]
    fig, axes = plt.subplots(2, 7, figsize=(25, 8))  # 2 rows × 7 cols
    axes = axes.flatten()

    for i, col in enumerate(y_df.columns):
        ax = axes[i]
        ax.hist(y_df[col].dropna().values, bins=bins, color="steelblue", edgecolor="black", alpha=0.7)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        if log:
            ax.set_yscale("log")

    # Hide unused subplots if any
    for j in range(n_targets, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    out_path = os.path.join(hist_dir, "target_histograms.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved target histograms to {out_path}")

