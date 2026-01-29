import os
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from data.dataloader import inverse_transform_y  # your helper
from experiments.list_variable import var_output, list_var  # fallback var names


def save_val_predictions(model, val_loader, config, exp_dir,
                         var_output_names=None, input_files=None):
    """
    Save validation predictions along with input variables and optional lon/lat info.
    Inputs are in order of list_var + var_output.

    Parameters:
        - model: trained model wrapper (has .predict())
        - val_loader: torch DataLoader with batches (X, y, id)
        - config: experiment config dict
        - exp_dir: output directory
        - var_output_names: list of output variable names (optional)
        - input_files: list of input CSVs (used to retrieve lon/lat info)
    """

    # === Load best model checkpoint ===
    ckpt_dir = config["logging"]["checkpoint_dir"]
    best_model_path = os.path.join(ckpt_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        try:
            model.load(best_model_path)
        except Exception as e:
            warnings.warn(f"Could not load best model: {e}. Using current model.")
    else:
        warnings.warn(f"No best_model.pt found at {best_model_path}. Using current model state.")

    if var_output_names is None:
        var_output_names = var_output

    # === Collect predictions, true values, inputs, and IDs ===
    y_true_batches, y_pred_batches, x_batches, id_batches = [], [], [], []

    for batch in val_loader:
        # expected: (X, y, id)
        if len(batch) == 3:
            X_b, y_b, ids_b = batch
        else:
            raise ValueError("Expected val_loader to yield (X, y, id). Got different format.")

        X_np = X_b.detach().cpu().numpy()
        y_np = y_b.detach().cpu().numpy()

        preds_b = model.predict(X_b)
        if isinstance(preds_b, torch.Tensor):
            preds_b = preds_b.detach().cpu().numpy()

        x_batches.append(X_np)
        y_true_batches.append(y_np)
        y_pred_batches.append(preds_b)
        id_batches.append(ids_b.detach().cpu().numpy())

    X_all = np.concatenate(x_batches, axis=0)
    y_true_norm = np.concatenate(y_true_batches, axis=0)
    y_pred_norm = np.concatenate(y_pred_batches, axis=0)
    catchment_ids = np.concatenate(id_batches, axis=0).flatten()

    # === Inverse transform targets ===
    scaler_y = config["data"]["Y_scaler"]
    y_true = inverse_transform_y(y_true_norm, scaler_y)
    y_pred = inverse_transform_y(y_pred_norm, scaler_y)

    # === (Optional) Inverse transform inputs if available ===
    if "X_scaler" in config["data"]:
        try:
            scaler_x = config["data"]["X_scaler"]
            X_denorm = scaler_x.inverse_transform(X_all)
        except Exception as e:
            warnings.warn(f"Could not inverse-transform inputs: {e}")
            X_denorm = X_all
    else:
        X_denorm = X_all

    # === Build DataFrame ===
    input_names = list_var + var_output
    input_cols = [f"{name}" for name in input_names[:X_denorm.shape[1]]]

    if len(var_output_names) != y_true.shape[1]:
        true_cols = [f"true_{i}" for i in range(y_true.shape[1])]
        pred_cols = [f"pred_{i}" for i in range(y_true.shape[1])]
    else:
        true_cols = [f"true_{name}" for name in var_output_names]
        pred_cols = [f"pred_{name}" for name in var_output_names]

    df_inputs = pd.DataFrame(X_denorm, columns=input_cols)
    df_true = pd.DataFrame(y_true, columns=true_cols)
    df_pred = pd.DataFrame(y_pred, columns=pred_cols)
    df_ids = pd.DataFrame({"catchment_id": catchment_ids})

    df_out = pd.concat([df_ids, df_inputs, df_true, df_pred], axis=1)

    # === Attach lon/lat if available ===
    if input_files is not None and len(input_files) > 0:
        try:
            #  i have to change that so taht it works to save with the coordinate file with both head and not catchement
            df_in = pd.read_csv(input_files[-1])
            df_in.rename(columns={"LisfloodX": "lon", "LisfloodY": "lat"}, inplace=True)

            possible_lon = [c for c in df_in.columns if c.lower() in ["lon", "longitude", "x"]]
            possible_lat = [c for c in df_in.columns if c.lower() in ["lat", "latitude", "y"]]
            lon_col = possible_lon[0] if possible_lon else None
            lat_col = possible_lat[0] if possible_lat else None

            if lon_col and lat_col:
                df_in_sub = df_in[[config["data"].get("id_column", "ID"), lon_col, lat_col]]
                df_out = df_out.merge(
                    df_in_sub,
                    left_on="catchment_id",
                    right_on=config["data"].get("id_column", "ID"),
                    how="left"
                ).drop(columns=[config["data"].get("id_column", "ID")])
                print(f"Added lon/lat from {input_files[0]}: ({lon_col}, {lat_col})")
            else:
                warnings.warn("Longitude/latitude columns not found in input CSV.")
        except Exception as e:
            warnings.warn(f"Could not attach lon/lat info: {e}")

    # === Save CSV ===
    os.makedirs(exp_dir, exist_ok=True)
    csv_path = os.path.join(exp_dir, "Test_predictions.csv")
    df_out.sort_values("catchment_id", inplace=True)
    df_out.to_csv(csv_path, index=False)
    print(f"Test predictions saved to: {csv_path} (N = {len(df_out)})")

    return df_out, csv_path