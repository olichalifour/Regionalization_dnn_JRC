import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd

def plot_input_output_correlations(X, y, input_names, output_names, save_dir):
    """
    Create and save scatter/correlation plots:
      - inputs vs inputs
      - outputs vs outputs
      - inputs vs outputs

    Parameters
    ----------
    X : np.ndarray, shape (N, n_inputs)
        Input feature matrix.
    y : np.ndarray, shape (N, n_outputs)
        Output/parameter matrix.
    input_names : list of str, length n_inputs
        Names of the input variables.
    output_names : list of str, length n_outputs
        Names of the output parameters.
    save_dir : str
        Directory where plots are saved.
    """

    print(f"Plot correlation matrix in: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    X_df = pd.DataFrame(X, columns=input_names)
    y_df = pd.DataFrame(y, columns=output_names)


    # --- Outputs vs Outputs ---
    print("Outputs vs Outputs plot")
    g = sns.pairplot(y_df, diag_kind="hist",corner=True, plot_kws={"s": 5, "alpha": 0.5})
    g.fig.suptitle("Outputs vs Outputs Correlations", y=1.02)
    g.savefig(os.path.join(save_dir, "outputs_vs_outputs.png"), dpi=300, bbox_inches="tight")
    plt.close(g.fig)

    # --- Inputs vs Inputs ---
    # print("Inputs vs Inputs plot")
    # g = sns.pairplot(X_df, diag_kind="hist",corner=True, plot_kws={"s": 5, "alpha": 0.5})
    # g.fig.suptitle("Inputs vs Inputs Correlations", y=1.02)
    # g.savefig(os.path.join(save_dir, "inputs_vs_inputs.png"), dpi=300, bbox_inches="tight")
    # plt.close(g.fig)

    # --- Inputs vs Outputs ---
    print("Inputs vs Outputs plot")
    plot_inputs_vs_outputs(X,y,input_names,output_names,save_dir)
    combined = pd.concat([X_df, y_df], axis=1)
    # g = sns.pairplot(combined, vars=input_names + output_names,
    #                  corner=True,diag_kind="hist", plot_kws={"s": 5, "alpha": 0.5})
    # g.fig.suptitle("Inputs vs Outputs Correlations", y=1.02)
    # g.savefig(os.path.join(save_dir, "inputs_outputs.png"), dpi=100, bbox_inches="tight")
    # plt.close(g.fig)

    # --- Correlation heatmap (optional, clearer overview) ---
    corr = combined.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, cbar=True)
    plt.title("Correlation Heatmap: Inputs and Outputs")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "correlation_heatmap.png"), dpi=300)
    plt.close()

    print("Done Plotting Correlation")


def plot_inputs_vs_outputs(X, y, input_names, output_names, save_dir):
    """
    Scatterplots of inputs (x-axis) vs outputs (y-axis).
    Arranged in a grid: rows = outputs, cols = inputs.
    """
    os.makedirs(save_dir, exist_ok=True)

    n_inputs = len(input_names)
    n_outputs = len(output_names)

    fig, axes = plt.subplots(
        n_outputs, n_inputs,
        figsize=(2*n_inputs, 2*n_outputs),
        sharex=False, sharey=False
    )

    if n_outputs == 1 and n_inputs == 1:
        axes = np.array([[axes]])
    elif n_outputs == 1:
        axes = axes[np.newaxis, :]
    elif n_inputs == 1:
        axes = axes[:, np.newaxis]

    for i, out_name in enumerate(output_names):
        for j, in_name in enumerate(input_names):
            ax = axes[i, j]
            ax.scatter(X[:, j], y[:, i], s=5, alpha=0.5)
            if i == n_outputs - 1:  # bottom row
                ax.set_xlabel(in_name, fontsize=8)
            if j == 0:  # first column
                ax.set_ylabel(out_name, fontsize=8)
            ax.tick_params(labelsize=6)

    fig.suptitle("Inputs vs Outputs Correlations", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_path = os.path.join(save_dir, "inputs_vs_outputs.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
