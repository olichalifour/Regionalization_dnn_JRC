import torch
import numpy as np
from optimization.list_variable import list_var,var_output
import os
import pickle









def make_fitness_function(model, fixed_values, X_scaler, Y_scaler):
    """
    Fitness function using a sklearn RandomForestRegressor.
    DEAP will MINIMIZE, so we return (-KGE,)
    """

    all_input_columns = list_var + var_output

    def fitness(individual):

        # --- Reconstruct full input vector ---
        input_dict = fixed_values.copy()
        for col, val in zip(var_output, individual):
            input_dict[col] = val

        x_full = np.array(
            [input_dict[c] for c in all_input_columns],
            dtype=np.float32
        ).reshape(1, -1)

        # --- Normalize inputs ---
        x_norm = X_scaler.transform(x_full)

        # --- Predict with RF ---
        pred_norm = model.predict(x_norm)

        # --- Inverse-transform target ---
        pred_true = Y_scaler.inverse_transform(
            np.array(pred_norm).reshape(-1, 1)
        )

        kge_pred = float(pred_true[0, 0])

        # DEAP minimizes → NEGATE KGE
        kge_pred = np.clip(kge_pred, -1.0, 1.0)

        return (kge_pred,)

    return fitness
