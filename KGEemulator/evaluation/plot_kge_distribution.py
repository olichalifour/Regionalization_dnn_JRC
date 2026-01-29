import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")

def plot_kge_distribution(kge_values, title="Distribution of KGE", savepath=None):
    """
    Plot histogram + KDE of KGE values.

    Args:
        kge_values (array-like or torch.Tensor): KGE values.
        title (str): Plot title.
        savepath (str or None): If given, save plot to this path instead of showing.
    """
    # Convert to numpy if torch tensor
    if isinstance(kge_values, torch.Tensor):
        kge_values = kge_values.detach().cpu().numpy()

    plt.figure(figsize=(8,5))
    sns.histplot(kge_values, bins=30, kde=True, color="skyblue")
    plt.xlabel("KGE values")
    plt.ylabel("Density")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    print(f"[INFO] Saved KGE distribution to {savepath}")