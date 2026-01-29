import argparse
import yaml
import os
import numpy as np
from sklearn.model_selection import KFold,cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.base import clone
from sklearn.decomposition import PCA
from models import create_model
from training.trainer import Trainer
from data.load_or_prep_data import load_or_prepare_data
from data.save_prediction import save_val_predictions
from data.disgnostic_target import per_target_diagnostics
from evaluation.plot_training import plot_training
from evaluation.plot_diagnostics import plot_diagnostics, plot_diagnostics_2
from evaluation.plot_kge_distribution import plot_kge_distribution
from evaluation.RF_feature_importance import save_multioutput_importances_scatter, save_feature_importances_scatter
from evaluation.map_error_plot import plot_catchment_error_map
from utils.serialization import make_json_serializable
from utils.verbose import print_config_verbose
import json
from data.dataloader import inverse_transform_y
import torch
import joblib

from experiments.list_variable import list_var,var_output

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(config_path):
    # 1. Load config
    config = load_config(config_path)
    exp_name = config["experiment_name"]

    if config.get("verbose", False):
        print_config_verbose(config)

    # 2. Set up experiment folder structure
    base_dir = config["logging"].get("base_dir", "AI_exp")
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Subfolders
    plot_dir = os.path.join(exp_dir, config["logging"]["plot_training"].get("folder", "plots"))
    ckpt_dir = os.path.join(exp_dir, config["logging"]["checkpoints"].get("folder", "checkpoints"))
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    config["logging"]["checkpoint_dir"] = ckpt_dir

    # 3. Get data

    data = load_or_prepare_data(config,plot_dir)
    train_loader, val_loader, test_loader = data["torch"]
    X_scaler, Y_scaler = data["scalers"]
    X_train, X_val, X_test, y_train, y_val,y_test,id_train,id_val,id_test = data["numpy"]



    # set nb out output equal to number of paramter or feature in the pca

    config["model"]["input_dim"] = len(list_var)+len(var_output)
    config["model"]["output_dim"] = 1

    os.makedirs(os.path.join(plot_dir, "target_histograms"),exist_ok=True)
    kge_plot_plath_norm = os.path.join(plot_dir, "target_histograms", "kge_distribuition_norm.png")
    plot_kge_distribution(y_val.reshape(-1, 1), savepath=kge_plot_plath_norm)

    kge_plot_plath = os.path.join(plot_dir, "target_histograms", "kge_distribuition.png")
    y_true = inverse_transform_y(y_val.reshape(-1, 1), Y_scaler)
    plot_kge_distribution(y_true,savepath=kge_plot_plath)



    config["data"]["Y_scaler"] = Y_scaler
    config["data"]["X_scaler"] = X_scaler



    # 4. Create model
    model = create_model(config)
    model_type = config["model"]["type"].lower()

    # 5. Train depending on model type
    trainer = Trainer(model, config)

    # put this inside your main() replacing your current "if 'rf' in model_type:" block
    if "rf" in model_type:

        # --- Configurable mode: shared vs. independent forests ---
        multioutput = config["model"].get("multioutput", False)

        base_rf = model.model  # assume create_model() returns a wrapper with .model = RandomForestRegressor

        if multioutput:
            print("Using MultiOutputRegressor (independent RF per output).")
            estimator = MultiOutputRegressor(base_rf)
        else:
            print("Using native RandomForestRegressor (shared splits across outputs).")
            estimator = base_rf



        # --- Cross-validation ---
        cv = config.get("training", {}).get("cv", 5)
        scores = cross_val_score(clone(estimator), X_train, np.ravel(y_train), scoring="neg_root_mean_squared_error", cv=cv, n_jobs=-1)
        cv_mean, cv_std = float(scores.mean()), float(scores.std())
        print(f"RandomForest CV mean RMSE: {-cv_mean:.3f} ± {cv_std:.3f}")


        estimator.fit(X_train, np.ravel(y_train))

        # --- Evaluate on hold-out validation set using sklearn metrics ---
        y_val_pred = estimator.predict(X_val)
        val_r2 = float(r2_score(y_val, y_val_pred, multioutput="uniform_average"))
        val_mae = float(mean_absolute_error(y_val, y_val_pred))

        # --- Store history (clear, explicit keys) ---
        trainer.history.setdefault("cv_r2", []).append(cv_mean)
        trainer.history.setdefault("cv_r2_std", []).append(cv_std)
        trainer.history.setdefault("val_r2", []).append(val_r2)
        trainer.history.setdefault("val_mae", []).append(val_mae)

        # --- Evaluate on TEST set instead of validation ---
        y_test_pred = estimator.predict(X_test)
        test_r2 = float(r2_score(y_test, y_test_pred, multioutput="uniform_average"))
        test_mae = float(mean_absolute_error(y_test, y_test_pred))

        trainer.history.setdefault("test_r2", []).append(test_r2)
        trainer.history.setdefault("test_mae", []).append(test_mae)

        # --- Save the final sklearn model ---
        ckpt_path = os.path.join(ckpt_dir, "random_forest.joblib")
        joblib.dump(estimator, ckpt_path)
        print(f"RandomForest saved to {ckpt_path}")

        # --- Diagnostics: use numpy arrays (don't iterate the torch val_loader which is for DNNs) ---
        if config["logging"]["diagnostics"].get("enabled", False):


            # inverse-transform back to original units (use your existing helper)

            y_true = inverse_transform_y(y_test.reshape(-1,1), Y_scaler)
            y_pred = inverse_transform_y(y_test_pred.reshape(-1,1), Y_scaler)

            diag_dir = os.path.join(plot_dir, "diagnostics")
            os.makedirs(diag_dir, exist_ok=True)

            plot_diagnostics(y_true, y_pred, output_dir=diag_dir, prefix="exp_diagnostics")

            save_path = os.path.join(diag_dir, "diag_2.png")
            plot_diagnostics_2(y_true, y_pred, savepath=save_path)

            # save_multioutput_importances_scatter(
            #     model=estimator,
            #     feature_names=list_var+var_output,
            #     var_output_name=["KGE"],
            #     save_dir = diag_dir,
            #     average=True,
            # )

            save_feature_importances_scatter(estimator, list_var+var_output, diag_dir, target_name="KGE")

    else:
        # --- Train model ---
        trainer.train(train_loader, val_loader)

        # --- Save history ---
        history_file = os.path.join(exp_dir, "history.json")
        serializable_history = make_json_serializable(trainer.history)
        with open(history_file, "w") as f:
            json.dump(serializable_history, f, indent=4)
        print(f"History saved to {history_file}")

        # --- Reload best model before plotting ---
        best_ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
        if os.path.exists(best_ckpt_path):
            print(f"Reloading best model from {best_ckpt_path} for diagnostics/plots...")
            model.load(best_ckpt_path)
        else:
            print("⚠️ No best_model.pt found, using final epoch model for diagnostics.")

        # --- Training curve plot ---
        if config["logging"]["plot_training"].get("enabled", False):
            plot_file = os.path.join(plot_dir, config["logging"]["plot_training"].get("file_name", "training_plot.png"))
            plot_training(trainer.history, save_path=plot_file)

        # --- Diagnostics using BEST model ---
        if config["logging"]["diagnostics"].get("enabled", False):
            y_true, y_pred = [], []

            for batch in test_loader:
                if len(batch) == 3:
                    X_test_batch, y_test_batch, _kge = batch
                else:
                    X_test_batch, y_test_batch = batch

                # --- Targets ---
                if isinstance(y_test_batch, torch.Tensor):
                    y_true.append(y_test_batch.detach().cpu().numpy())
                else:
                    y_true.append(y_test_batch)

                # --- Predictions with best model ---
                y_pred_batch = model.predict(X_test_batch)
                if isinstance(y_pred_batch, torch.Tensor):
                    y_pred.append(y_pred_batch.detach().cpu().numpy())
                else:
                    y_pred.append(y_pred_batch)

            # Concatenate
            y_true_norm = np.concatenate(y_true, axis=0)
            y_pred_norm = np.concatenate(y_pred, axis=0)

            # Inverse-transform back to original units
            Y_scaler = config["data"]["Y_scaler"]
            y_true = inverse_transform_y(y_true_norm, Y_scaler)
            y_pred = inverse_transform_y(y_pred_norm, Y_scaler)

            # --- Run diagnostics ---
            diag_dir = os.path.join(plot_dir, "diagnostics_best")
            os.makedirs(diag_dir, exist_ok=True)

            plot_diagnostics(y_true, y_pred, output_dir=diag_dir, prefix="exp_diagnostics_best")

            save_path = os.path.join(diag_dir, "diag_2_best.png")
            plot_diagnostics_2(y_true, y_pred, savepath=save_path)

            if config["logging"].get("save_validation", False):
                df_out,csv_path = save_val_predictions(
                    model=model,
                    val_loader=val_loader,
                    config=config,
                    exp_dir=exp_dir,
                    var_output_names=['KGE'],
                    input_files=config["data"]["input_files_heads"],  # required for ID reconstruction
                )
                plot_catchment_error_map(df_out, output_dir=diag_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML experiment config")
    args = parser.parse_args()

    main(args.config)