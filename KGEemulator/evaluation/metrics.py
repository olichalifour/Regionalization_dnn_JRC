# evaluation/metrics.py
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy.stats import pearsonr
from numpy.fft import fft2

def to_numpy(x):
    """Convert torch tensor or list to numpy array."""
    if "torch" in str(type(x)):
        return x.detach().cpu().numpy()
    return np.array(x)

# --- Individual metrics ---

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def pearson(y_true, y_pred):
    r, _ = pearsonr(y_true.flatten(), y_pred.flatten())
    return r



# --- Central dispatcher ---

def compute_metrics(y_pred, y_true, metric_list):
    """Compute selected metrics and return dict."""
    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)

    results = {}
    for m in metric_list:
        if m == "mae":
            results["mae"] = mae(y_true, y_pred)
        elif m == "mse":
            results["mse"] = mse(y_true, y_pred)
        elif m == "r2":
            results["r2"] = r2(y_true, y_pred)
        elif m == "pearson":
            results["pearson"] = pearson(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {m}")
    return results