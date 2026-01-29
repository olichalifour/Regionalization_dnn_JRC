import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd

def per_target_diagnostics(y_true, y_pred, target_names=None):
    n_targets = y_true.shape[1]
    if target_names is None:
        target_names = [f"t{i}" for i in range(n_targets)]

    rows = []
    for i, name in enumerate(target_names):
        true = y_true[:, i]
        pred = y_pred[:, i]
        mean_true = true.mean()
        mean_pred = pred.mean()
        std_true = true.std()
        std_pred = pred.std()
        r = np.corrcoef(true, pred)[0,1]
        r2 = r2_score(true, pred)
        mae = mean_absolute_error(true, pred)
        rows.append({
            "target": name,
            "mean_true": mean_true,
            "mean_pred": mean_pred,
            "std_true": std_true,
            "std_pred": std_pred,
            "corr": r,
            "r2": r2,
            "mae": mae
        })
    df = pd.DataFrame(rows)
    print(df)
    return df

