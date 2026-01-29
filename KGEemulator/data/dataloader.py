#use path /eos/jeodpp/data/projects/FLOODS-RIVER/LisfloodPL/nh_dataset/attributes_casadje/
import glob
import warnings
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA,PCA
from evaluation.plot_correlation import plot_input_output_correlations
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from evaluation.plot_histograms_target import plot_target_histograms
from experiments.list_variable import list_var,var_output,list_var_inflow
import matplotlib.pyplot as plt



# --- helper to find a KGE column name in param history file ---
def _find_kge_column(df: pd.DataFrame):
    candidates = ["Kling Gupta Efficiency", "KGE", "kge", "kling_gupta_efficiency",
                  "KlingGuptaEfficiency", "Kling_Gupta_Efficiency"]
    for c in candidates:
        if c in df.columns:
            return c
    lower_cols = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower_cols:
            return lower_cols[c.lower()]
    return None

def _report_dataset_stats(X, y, label="Dataset"):
    """Print summary statistics about NaN values and dataset size."""
    total_rows = X.shape[0]
    nan_X = np.isnan(X).sum()
    nan_y = np.isnan(y).sum()

    rows_with_nan_X = np.isnan(X).any(axis=1).sum()
    rows_with_nan_y = np.isnan(y).any(axis=1).sum()
    rows_with_nan = np.logical_or(np.isnan(X).any(axis=1), np.isnan(y).any(axis=1)).sum()

    print(f"\n--- {label} Summary ---")
    print(f"Total samples: {total_rows:,}")
    print(f"Rows with NaN in X: {rows_with_nan_X:,}")
    print(f"Rows with NaN in y: {rows_with_nan_y:,}")
    print(f"Total rows with any NaN: {rows_with_nan:,}")
    print(f"Total NaN values in X: {nan_X:,}")
    print(f"Total NaN values in y: {nan_y:,}")
    print("-------------------------\n")

def load_global_defaults(reference_csv):
    raw = pd.read_csv(reference_csv, header=0, skip_blank_lines=True)
    default_row = raw[raw.iloc[:, 0] == "DefaultValue"]

    if default_row.empty:
        raise RuntimeError("No DefaultValue row found in reference CSV")

    defaults = default_row.iloc[0].to_dict()
    defaults.pop("randId", None)

    return {k: float(v) for k, v in defaults.items() if pd.notna(v)}

def _expand_inputs_with_paramhist(
    inputs_df,
    param_hist_dir,
    id_column="ID",
    randomness=None,
    remove_high_eq=False,
    eq_threshold=False
):
    """
    For each catchment, combine static features with all parameter realizations.
    Output:
        X = [list_var + 14 parameters]
        y = KGE
    """
    X_rows, y_rows = [], []
    failed_ids, eq_flags = [], {}
    id_rows = []

    GLOBAL_PARAM_DEFAULTS = load_global_defaults("/home/chaliol/Documents/catch_param_history/4/paramsHistory.csv")

    for _, input_row in inputs_df.iterrows():
        id_val = input_row[id_column]
        param_file = os.path.join(param_hist_dir, f"{int(id_val)}/paramsHistory.csv")

        if not os.path.exists(param_file):
            warnings.warn(f"Missing param file for {id_val}")
            failed_ids.append(id_val)
            continue

        try:
            ph = pd.read_csv(
                param_file,
                header=0,
                skip_blank_lines=True
            )

            # Remove metadata rows
            ph = ph[~ph.iloc[:, 0].isin(["MinValue", "DefaultValue", "MaxValue"])]

            # Drop fully empty columns (causes Unnamed columns)
            ph = ph.dropna(axis=1, how="all")

            # Drop fully empty rows
            ph = ph.dropna(axis=0, how="all")

            # Reset index after cleaning
            ph = ph.reset_index(drop=True)
        except Exception as e:
            warnings.warn(f"Could not read {param_file}: {e}")
            failed_ids.append(id_val)
            continue

        kge_col = _find_kge_column(ph)
        if kge_col is None:
            warnings.warn(f"No KGE column in {param_file}")
            failed_ids.append(id_val)
            continue

        missing = [c for c in var_output if c not in ph.columns]

        for param in missing:
            if param in GLOBAL_PARAM_DEFAULTS:
                warnings.warn(
                    f"{param_file}: missing '{param}', filled with GLOBAL default "
                    f"{GLOBAL_PARAM_DEFAULTS[param]}"
                )
                ph[param] = GLOBAL_PARAM_DEFAULTS[param]
            else:
                warnings.warn(
                    f"{param_file}: missing '{param}' and no global default available"
                )
                failed_ids.append(id_val)
                continue

        params = ph[var_output].astype(float).values
        kge_vals = ph[kge_col].astype(float).values
        param_var = np.mean(np.std(params, axis=0) / (np.mean(params, axis=0) + 1e-6))
        eq_flags[id_val] = param_var
        if remove_high_eq and param_var > eq_threshold:
            warnings.warn(f"Skipping {id_val} high equifinality ({param_var:.3f})")
            continue

        input_features = input_row[list_var].values.astype(float)
        repeated_inputs = np.repeat(input_features.reshape(1, -1), params.shape[0], axis=0)
        combined_inputs = np.hstack([repeated_inputs, params])

        id_rows.append(np.full(params.shape[0],id_val).reshape(-1, 1))
        X_rows.append(combined_inputs)
        y_rows.append(kge_vals.reshape(-1, 1))

    if len(X_rows) == 0:
        raise RuntimeError("No samples created; check param_hist files and columns.")

    X_expanded = np.vstack(X_rows)
    y_expanded = np.vstack(y_rows)

    id_expanded = np.vstack(id_rows)

    # --- Report before cleaning ---
    _report_dataset_stats(X_expanded, y_expanded, label="Before NaN removal")

    # --- Remove rows with NaN in inputs or outputs ---
    valid_mask = ~(
        np.isnan(X_expanded).any(axis=1) | np.isnan(y_expanded).any(axis=1)
    )
    X_clean = X_expanded[valid_mask]
    y_clean = y_expanded[valid_mask]
    id_clean = id_expanded[valid_mask]

    # --- Report after cleaning ---
    _report_dataset_stats(X_clean, y_clean, label="After NaN removal")

    # --- Remove rows with KGE below threshold ---
    kge_threshold = -0.4
    kge_mask = y_clean.flatten() >= kge_threshold
    X_clean = X_clean[kge_mask]
    y_clean = y_clean[kge_mask]
    id_clean = id_clean[kge_mask]

    print(f"Removed {np.sum(~kge_mask):,} samples with KGE < {kge_threshold}")
    print(f"Remaining samples: {X_clean.shape[0]:,}")
    print("Skewness:", np.mean(((y_clean - np.mean(y_clean)) / np.std(y_clean)) ** 3))

    # --- Final report ---
    _report_dataset_stats(X_clean, y_clean, label=f"After KGE filtering (≥ {kge_threshold})")

    return X_clean, y_clean, id_clean, failed_ids, eq_flags


def load_and_merge_data(input_csv_files_head,input_csv_files_inflow,config, id_column="ID",
                        param_hist_dir=None,
                        remove_high_eq=False, eq_threshold=False,
                        debug=True):
    """
    Merge inputs and expand with parameters + KGE.
    Validation split is done by catchment IDs (entire catchments removed).
    """

    # --- Merge all input CSVs ---

    inputs_head = [pd.read_csv(f) for f in input_csv_files_head]
    Xhead_df = inputs_head[0]
    for df in inputs_head[1:]:
        Xhead_df = Xhead_df.merge(df, on=id_column, how="inner")

    Xhead_df.rename(columns={"LisfloodX": "lon", "LisfloodY": "lat"}, inplace=True)
    Xhead_df = Xhead_df[[id_column] + list_var]
    X_df = Xhead_df
    # inputs_inflow = [pd.read_csv(f) for f in input_csv_files_inflow]
    # Xinflow_df = inputs_inflow[0]
    # for df in inputs_inflow[1:]:
    #
    #     Xinflow_df = Xinflow_df.merge(df, on=id_column, how="inner")
    #
    # Xinflow_df.rename(columns={"LisfloodX": "lon", "LisfloodY": "lat"}, inplace=True)
    # Xinflow_df = Xinflow_df[[id_column] + list_var + list_var_inflow]


    # X_df = pd.concat([Xhead_df, Xinflow_df],ignore_index=True)
    # fill the nan for the inflow data of the head catchement by 0 instead of nan
    # X_df.fillna(0, inplace=True)


    # --- Expand with parameter history ---
    assert param_hist_dir is not None, "param_hist_dir required."

    X_exp, y_exp, id_exp, failed, eq_flags = _expand_inputs_with_paramhist(
        X_df,
        param_hist_dir,
        id_column=id_column,
        remove_high_eq=remove_high_eq,
        eq_threshold=eq_threshold
    )

    # --- Normalize/flatten id_exp to 1-D numeric array ---
    id_exp = np.asarray(id_exp)
    if id_exp.ndim > 1:
        # collapse to 1D (works for shape (N,1) etc.)
        id_exp = id_exp.reshape(-1)
    else:
        id_exp = id_exp.flatten()

    # --- Ensure X_exp and y_exp are numpy arrays ---
    X_exp = np.asarray(X_exp)
    y_exp = np.asarray(y_exp)

    # --- Debugging prints to help you confirm shapes ---
    if debug:
        print("X_exp.shape:", X_exp.shape)
        print("y_exp.shape:", y_exp.shape)
        print("id_exp.shape:", id_exp.shape)

    # Basic sanity checks
    N = X_exp.shape[0]
    if id_exp.shape[0] != N:
        raise RuntimeError(f"Length mismatch: X_exp has {N} rows but id_exp has {id_exp.shape[0]} elements.")

    # split using Carlo splitting

    def read_id_list(path):
        with open(path, "r") as f:
            return np.array([int(line.strip()) for line in f if line.strip()])

    list_train = []
    list_val = []
    list_test = []
    for idx_list in range(len(config["id_list_train"])):
        list_train.append(read_id_list(config["id_list_train"][idx_list]))
        list_val.append(read_id_list(config["id_list_val"][idx_list]))
        list_test.append(read_id_list(config["id_list_test"][idx_list]))

    list_train = np.concatenate(list_train)
    list_val = np.concatenate(list_val)
    list_test = np.concatenate(list_test)


    set_train = set(list_train.tolist())
    set_val = set(list_val.tolist())
    set_test = set(list_test.tolist())

    # Safety check
    overlap = (set_train & set_val) | (set_train & set_test) | (set_val & set_test)
    if len(overlap) > 0:
        raise RuntimeError(f"Your ID lists overlap! Conflicting IDs: {overlap}")

    # Masks
    train_mask = np.isin(id_exp, list_train)
    val_mask = np.isin(id_exp, list_val)
    test_mask = np.isin(id_exp, list_test)

    # Extract splits
    X_train = X_exp[train_mask]
    y_train = y_exp[train_mask]
    id_train = id_exp[train_mask]

    X_val = X_exp[val_mask]
    y_val = y_exp[val_mask]
    id_val = id_exp[val_mask]

    X_test = X_exp[test_mask]
    y_test = y_exp[test_mask]
    id_test = id_exp[test_mask]

    print("Manual split by ID lists:")
    print(f"  Train catchments: {len(list_train)}   rows: {len(X_train)}")
    print(f"  Val catchments:   {len(list_val)}     rows: {len(X_val)}")
    print(f"  Test catchments:  {len(list_test)}    rows: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, id_train, id_val, id_test


def normalize_data(X_train, y_train, X_val, y_val,X_test,y_test):
    scaler_X = StandardScaler()
    # scaler_y = StandardScaler()
    scaler_y = QuantileTransformer(output_distribution="normal",n_quantiles=1000)

    X_train_norm = scaler_X.fit_transform(X_train)
    X_val_norm = scaler_X.transform(X_val)
    X_test_norm = scaler_X.transform(X_test)

    y_train_norm = scaler_y.fit_transform(y_train)
    y_val_norm = scaler_y.transform(y_val)
    y_test_norm = scaler_y.fit_transform(y_test)

    return X_train_norm, X_val_norm,X_test_norm, y_train_norm, y_val_norm, y_test_norm, scaler_X, scaler_y


def get_dataloaders(data_config, training_config, plot_dir):
    input_files_head = data_config["input_files_heads"]
    input_files_inflow = data_config["input_files_inflows"]

    param_hist_dir = data_config["param_hist_dir"]
    shuffle = data_config.get("shuffle", True)
    id_column = data_config.get("id_column", "ID")
    batch_size = training_config.get("batch_size", 64)

    X_train, X_val,X_test, y_train, y_val,y_test, id_train, id_val,id_test = load_and_merge_data(
        input_files_head,input_files_inflow, config = data_config,id_column=id_column,
        param_hist_dir=param_hist_dir,
    )

    X_train_norm, X_val_norm,X_test_norm, y_train_norm, y_val_norm,y_test_norm, scaler_X, scaler_y = normalize_data(
        X_train, y_train, X_val, y_val,X_test,y_test
    )


    X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_norm, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train_norm, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_norm, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_norm, dtype=torch.float32)

    id_train_tensor = torch.tensor(id_train, dtype=torch.float32)
    id_val_tensor = torch.tensor(id_val, dtype=torch.float32)
    id_test_tensor = torch.tensor(id_test, dtype=torch.float32)

    train_ds = TensorDataset(X_train_tensor, y_train_tensor,id_train_tensor)
    val_ds = TensorDataset(X_val_tensor, y_val_tensor,id_val_tensor)
    test_ds = TensorDataset(X_test_tensor, y_test_tensor, id_test_tensor)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return {
        "torch": (train_loader, val_loader,test_loader),
        "numpy": (X_train_norm, X_val_norm,X_test_norm, y_train_norm, y_val_norm,y_test_norm,id_train,id_val,id_test),
        "scalers": (scaler_X, scaler_y)
    }


def inverse_transform_y(y_pred_scaled, scaler_y):
    """
    Inverse transform predictions (y) back to the original scale.
    If PCA was used, first invert PCA, then inverse-scale.
    """

    return scaler_y.inverse_transform(y_pred_scaled)

if __name__ == "__main__":
    # Example usage
    data_config = {
        "input_files": ["data/raw/input1.csv", "data/raw/input2.csv"],
        "output_file": "data/raw/output.csv",
        "val_split": 0.2,
        "shuffle": True
    }

    training_config = {"batch_size": 64}

    train_loader, val_loader, X_scaler, y_scaler = get_dataloaders(data_config, training_config)

    # Check shapes
    for X_batch, y_batch in train_loader:
        print("X:", X_batch.shape, "y:", y_batch.shape)
        break