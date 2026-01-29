import pandas as pd

def load_param_bounds(path="parameter_range.csv"):
    df = pd.read_csv(path)
    bounds = {}
    for _, row in df.iterrows():
        bounds[row["Parameter"]] = (float(row["Min"]), float(row["Max"]))
    return bounds