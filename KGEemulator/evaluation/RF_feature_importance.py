import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.cm as cm
import seaborn as sns

def save_multioutput_importances_scatter(model, feature_names,var_output_name, save_dir, average=False):
    """
    Save scatter plot of feature importances from a fitted MultiOutputRegressor with RandomForest base.

    Parameters
    ----------
    model : MultiOutputRegressor
        A fitted MultiOutputRegressor (with RandomForestRegressor base).
    feature_names : list or array
        Names of input features.
    save_dir : str
        Path to save the figure (e.g., 'figures/feature_importances.png').
    average : bool, default=False
        If True, also overlays a black line for the average importance across outputs.
    prefix : str, default="output"
        Prefix for title of saved plot.
    """
    n_outputs = len(model.estimators_)
    n_features = len(feature_names)

    # Collect importances per output
    importances_per_output = np.zeros((n_outputs, n_features))
    for i, est in enumerate(model.estimators_):
        importances_per_output[i, :] = est.feature_importances_

    # Colors for outputs
    colors = cm.get_cmap("jet", n_outputs)
    # markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*"]
    plt.figure(figsize=(max(10, n_features * 0.4), 6))

    # Scatter each output’s importance
    for i in range(n_outputs):
        plt.scatter(
            range(n_features),
            importances_per_output[i, :],
            label=var_output_name[i],
            alpha=0.7,
            s=50,
            color=colors(i),
            edgecolors="k",
            # marker= markers[i]
        )

    # Optionally add average importance line
    if average:
        mean_importances = importances_per_output.mean(axis=0)
        plt.plot(range(n_features), mean_importances, "k--", lw=2, label="Average")

    plt.xticks(range(n_features), feature_names, rotation=90, fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel("Feature Importance", fontsize=12)
    plt.title(f"Feature Importances per Output", fontsize=14)
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()

    # Save and close
    save_path = os.path.join(save_dir,"RF_feature_imp.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved scatter scatter feature importance {save_path}")

    plt.figure(figsize=(max(10, n_features * 0.4), 0.5 * n_outputs + 4))
    sns.heatmap(
        importances_per_output,
        cmap="inferno_r",
        xticklabels=feature_names,
        yticklabels=var_output_name,
        cbar_kws={"label": "Feature Importance"},
        linecolor="k",
        linewidths=0.5
    )
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(f"Feature Importances Heatmap", fontsize=14)
    plt.tight_layout()
    save_path_heatmap = os.path.join(save_dir, "RF_feature_imp_heatmap.png")
    plt.savefig(
        save_path_heatmap,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved heatmap scatter feature importance {save_path}")


def save_feature_importances_scatter(model, feature_names, save_dir, target_name="Output"):
    """
    Save scatter and heatmap plots of feature importances from a fitted RandomForestRegressor.

    Parameters
    ----------
    model : RandomForestRegressor
        A fitted RandomForestRegressor.
    feature_names : list or array
        Names of input features.
    save_dir : str
        Path to save the figures.
    target_name : str, default="Output"
        Name of the target variable (for labeling plots).
    """

    os.makedirs(save_dir, exist_ok=True)
    n_features = len(feature_names)

    # --- Extract feature importances ---
    try:
        importances = model.feature_importances_
    except AttributeError:
        raise ValueError("Model does not have feature_importances_. Make sure it's a fitted RandomForestRegressor.")

    if len(importances) != n_features:
        raise ValueError(f"Length mismatch: expected {n_features} feature importances, got {len(importances)}.")

    # --- Plot scatter of importances ---
    plt.figure(figsize=(max(10, n_features * 0.4), 6))
    colors = cm.get_cmap("viridis")(np.linspace(0, 1, n_features))
    plt.scatter(
        range(n_features),
        importances,
        s=60,
        color=colors,
        edgecolors="k",
        alpha=0.8
    )

    plt.xticks(range(n_features), feature_names, rotation=90, fontsize=10)
    plt.ylabel("Feature Importance", fontsize=12)
    plt.title(f"Random Forest Feature Importances ({target_name})", fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "RF_feature_importance_scatter.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved scatter feature importance plot to {save_path}")

    # --- Plot heatmap ---
    plt.figure(figsize=(max(10, n_features * 0.4), 2.5))
    sns.heatmap(
        importances[np.newaxis, :],
        cmap="inferno_r",
        cbar_kws={"label": "Feature Importance"},
        xticklabels=feature_names,
        yticklabels=[target_name],
        linewidths=0.5,
        linecolor="k"
    )
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(f"Feature Importances Heatmap ({target_name})", fontsize=14)
    plt.tight_layout()

    save_path_heatmap = os.path.join(save_dir, "RF_feature_importance_heatmap.png")
    plt.savefig(save_path_heatmap, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved heatmap feature importance plot to {save_path_heatmap}")

    # --- Print ranked list of features ---
    sorted_idx = np.argsort(importances)[::-1]
    print("\n[INFO] Top 10 Feature Importances:")
    for idx in sorted_idx[:10]:
        print(f"  {feature_names[idx]:25s}: {importances[idx]:.4f}")

    # Optional: Save to CSV
    csv_path = os.path.join(save_dir, "RF_feature_importances.csv")
    np.savetxt(
        csv_path,
        np.column_stack([feature_names, importances]),
        fmt="%s",
        delimiter=",",
        header="Feature,Importance",
        comments=""
    )
    print(f"�� Saved feature importances to {csv_path}")