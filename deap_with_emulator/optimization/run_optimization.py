
import os
import yaml
import torch
import joblib
import pandas as pd
from model.DNN_model import DNN
from optimization.fitness_function import make_fitness_function
from optimization.utils_deap import setup_deap
from optimization.optimize_catchment import optimize_for_catchment
from optimization.list_variable import list_var,var_output
from optimization.load_param_bounds import load_param_bounds
import pickle

def load_data_dict(plot_dir):

    save_path = os.path.join(plot_dir, "preprocessed_data.pkl")

    with open(save_path, "rb") as f:
        data = pickle.load(f)
    return data

def main():
    # === Config ===
    # with open("config/dnn_config.yaml", "r") as f:
    #     dnn_config = yaml.safe_load(f)
    with open("config/deap_config.yaml", "r") as f:
        deap_config = yaml.safe_load(f)["deap"]



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # === Load trained model ===
    # model = DNN(dnn_config)
    # model.load("/home/chaliol/Documents/AI_exp/kge_dnn_baseline/checkpoints/best_model.pt")
    # model.model.to(device)
    # model.model.eval()



    # === Load trained Random Forest ===
    rf_model_path = "/home/chaliol/Documents/AI_exp/kge_random_forest_glofas5/checkpoints/random_forest.joblib"
    model = joblib.load(rf_model_path)

    print("✅ Random Forest model loaded")

    # === Load data ===
    data_path = "/home/chaliol/Documents/AI_exp/kge_dnn_baseline_glofas5/Test_predictions.csv"

    plot_dir = "/home/chaliol/Documents/AI_exp/kge_random_forest_glofas5/plots/"
    df = pd.read_csv(data_path)

    # they are not normalize. therefore, we have to normalize them to do tommorow.
    data = load_data_dict(plot_dir)
    X_scaler, Y_scaler = data["scalers"]

    # input_cols = [c for c in df.columns if c.startswith("input_")]
    catchments = df["catchment_id"].unique()

    # Load CSV bounds
    bounds_dict = load_param_bounds("/home/chaliol/PycharmProjects/deap_with_emulator/config/param_ranges.csv")

    results = []

    for cid in catchments:
        print(f"\n===== Optimizing catchment {cid} =====")
        catch_df = df[df["catchment_id"] == cid].reset_index(drop=True)

        # Extract fixed + optimized input sets
        # opt_cols = [f"{v}" for v in var_output if f"{v}" in input_cols]
        # fixed_cols = [c for c in input_cols if c not in opt_cols]

        first_row = catch_df.iloc[0]
        print(first_row)
        fixed_values = {col: float(first_row[col]) for col in list_var}



        # Ensure var_output exists in that CSV
        missing = [v for v in var_output if v not in bounds_dict]
        if missing:
            raise ValueError(f"Missing bounds for: {missing}")

        # Build fitness function
        # fitness_fn = make_fitness_function(model, device, fixed_values, X_scaler, Y_scaler)
        fitness_fn = make_fitness_function(model, fixed_values, X_scaler, Y_scaler)


        # Setup DEAP with real bounds
        toolbox = setup_deap(fitness_fn, var_output, bounds_dict, deap_config)

        best_params, best_kge = optimize_for_catchment(
            toolbox, deap_config, cid, var_output
        )

        # Build final full input set (fixed + optimized)
        best_full = fixed_values.copy()
        for c, v in zip(var_output, best_params):
            best_full[c] = v

        results.append({
            "catchment_id": cid,
            "best_kge": best_kge,
            **{col: best_full[col] for col in var_output}
        })

    # === Save all results ===
    results_df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/optimized_params.csv", index=False)
    print("\n✅ Saved final best parameters to results/optimized_params.csv")

if __name__ == "__main__":
    main()