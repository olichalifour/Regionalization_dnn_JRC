import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

def plot_training(history, save_path=None):
    """
    Plot training and validation metrics dynamically.

    history: dict with keys like 'train_loss', 'val_loss', 'val_mae', etc.
    save_path: optional path to save figure
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    # Separate train/val loss and other metrics
    val_metrics_keys = [k for k in history.keys() if k.startswith("val_") and k != "val_loss"]

    n_subplots = 1 + len(val_metrics_keys)  # 1 for loss + others
    plt.figure(figsize=(8, 4 * n_subplots))

    # ---- Plot train_loss vs val_loss ----
    plt.subplot(n_subplots, 1, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    if "val_loss" in history:
        plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.grid(True)
    plt.legend()

    # ---- Plot other validation metrics dynamically ----
    for i, key in enumerate(val_metrics_keys):
        plt.subplot(n_subplots, 1, i + 2)
        plt.plot(epochs, history[key], label=key.replace("_", " ").title())
        plt.xlabel("Epoch")
        plt.ylabel(key.split("_")[1].upper())
        plt.title(f"Validation {key.split('_')[1].upper()}")
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Training plot saved to {save_path}")
    # plt.show()