def print_config_verbose(config: dict):
    """
    Print verbose information about the experiment configuration.
    Works for DNN, Multihead DNN, Random Forest, etc.
    """

    print("=" * 60)
    print(f" Experiment: {config.get('experiment_name', 'Unnamed')} ")
    print("=" * 60)

    # --- Model section ---
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "Unknown")
    print(f"[Model] Type: {model_type}")

    if model_type.startswith("DNN"):
        print(f"  Input dim: {model_cfg.get('input_dim', '?')}")
        print(f"  Hidden layers: {model_cfg.get('hidden_layers', [])}")
        if "head_hidden" in model_cfg:
            print(f"  Head hidden layers: {model_cfg['head_hidden']}")
        print(f"  Activation: {model_cfg.get('activation', 'relu')}")
        print(f"  Dropout: {model_cfg.get('dropout', 0.0)}")
        loss = model_cfg.get("loss", {})
        print("  Loss config:")
        for k, v in loss.items():
            print(f"    {k}: {v}")
    elif model_type == "RF":
        print(f"  Multioutput: {model_cfg.get('multioutput', False)}")
        print(f"  n_estimators: {model_cfg.get('n_estimators', 100)}")
        print(f"  min_samples_leaf: {model_cfg.get('min_samples_leaf', 1)}")
        print(f"  max_features: {model_cfg.get('max_features', 'auto')}")
        print(f"  max_depth: {model_cfg.get('max_depth', None)}")
        print(f"  random_state: {model_cfg.get('random_state', None)}")

    # --- Training section ---
    print("\n[Training]")
    training_cfg = config.get("training", {})
    if any(v is not None for v in training_cfg.values()):
        for k, v in training_cfg.items():
            print(f"  {k}: {v}")
    else:
        print("  (No training hyperparameters specified)")

    # --- Data section ---
    print("\n[Data]")
    data_cfg = config.get("data", {})
    print(f"  Input files: {data_cfg.get('input_files', [])}")
    print(f"  Output file: {data_cfg.get('output_file', '')}")
    print(f"  Augment with param hist: {data_cfg.get('augment_with_paramhist', False)}")
    if data_cfg.get("use_pca_targets", False):
        print("  PCA targets: Enabled")
    else:
        print("  PCA targets: Disabled")
    print(f"  Realizations: {data_cfg.get('n_realizations', 1)} (val={data_cfg.get('val_n_realizations', 1)})")
    print(f"  Validation split: {data_cfg.get('val_split', 0.2)}")
    print(f"  Shuffle: {data_cfg.get('shuffle', False)}")

    # --- Randomness section ---
    print("\n[Randomness]")
    rand_cfg = config.get("randomness", {"enabled": False})
    if rand_cfg.get("enabled", False):
        print(f"  Enabled (seed={rand_cfg.get('seed', 0)})")
        print(f"  Strategy: {rand_cfg.get('strategy', 'gaussian')}")
        print(f"  Perturbation: {rand_cfg.get('perturbation', 'relative')}")
        print(f"  Scale: {rand_cfg.get('scale', 0.01)}")
        if rand_cfg.get("per_feature_scale"):
            print("  Per-feature scale overrides: Yes")
        else:
            print("  Per-feature scale overrides: None")
        features = rand_cfg.get("features", None)
        print(f"  Features to perturb: {features if features else 'All numeric features'}")
        print(f"  Clip: {rand_cfg.get('clip', False)}")
        print(f"  Categorical strategy: {rand_cfg.get('categorical_strategy', 'none')}")
        print(f"  Dropout prob: {rand_cfg.get('dropout_prob', 0.0)}")
    else:
        print("  Disabled")

    # --- Evaluation section ---
    print("\n[Evaluation]")
    eval_cfg = config.get("evaluation", {})
    print(f"  Metrics: {eval_cfg.get('metrics', [])}")
    print(f"  Save predictions: {eval_cfg.get('save_predictions', False)}")

    # --- Logging section ---
    print("\n[Logging]")
    log_cfg = config.get("logging", {})
    print(f"  Base dir: {log_cfg.get('base_dir', './logs')}")
    plot_cfg = log_cfg.get("plot_training", {})
    print(f"  Plot training: {'Enabled' if plot_cfg.get('enabled', False) else 'Disabled'}")
    diag_cfg = log_cfg.get("diagnostics", {})
    print(f"  Diagnostics: {'Enabled' if diag_cfg.get('enabled', False) else 'Disabled'}")
    ckpt_cfg = log_cfg.get("checkpoints", {})
    print(f"  Checkpoints every {ckpt_cfg.get('save_every', 'N/A')} → folder: {ckpt_cfg.get('folder', './checkpoints')}")

    print("=" * 60)
    print()