
from .DNN_model import DNN
from .base_model import BaseModel
from .RF_model import RandomForestModel

def create_model(config: dict):
    """Factory to create a model from config dict."""

    model_config = config["model"]
    model_type = model_config["type"]
    if model_type == "DNN":
        if model_config["per_output"]:
            return DNN(config,per_output_model=True)
        else:
            return DNN(config)
    elif model_type == "RF":
        return RandomForestModel(model_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")