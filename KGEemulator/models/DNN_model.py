import torch
import torch.nn as nn
import torch.optim as optim
from .base_model import BaseModel
from loss_fct.weghtedmseloss import WeightedMSELoss
import numpy as np

class DNN(BaseModel):
    def __init__(self, config: dict, per_output_model: bool = False):
        """
        per_output_model: if True, create one sub-model per output.
        """
        super().__init__(config)
        model_config = config["model"]
        self.input_dim = model_config.get("input_dim", 50)
        self.output_dim = model_config.get("output_dim", 14)
        self.hidden_layers = model_config.get("hidden_layers", [128, 64])
        self.activation = model_config.get("activation", "relu")
        self.dropout = model_config.get("dropout", 0.0)
        self.learning_rate = model_config.get("learning_rate", 1e-3)
        self.weight_decay = model_config.get("weight_decay", 0.0)

        loss_config = config["model"]["loss"]
        self.alpha = loss_config.get("alpha", 1)
        self.beta = loss_config.get("beta", 1)
        self.gamma = loss_config.get("gamma", 1)
        self.eps = loss_config.get("eps", 1e-6)

        self.per_output_model = per_output_model
        # Map string → PyTorch activation
        activations = {
            "linear":None,
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }
        self.activation_layer = activations[self.activation]

        if self.per_output_model:
            # Create a ModuleList of independent model
            self.model = nn.ModuleList([
                self._build_single_output_model() for _ in range(self.output_dim)
            ])
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        else:
            self.model = self._build_single_output_model(output_dim=self.output_dim)
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )

        self.criterion = WeightedMSELoss(alpha=self.alpha,
                                         beta=self.beta,
                                         gamma=self.gamma,
                                         eps=self.eps)

    def _build_single_output_model(self, output_dim: int = 1):
        layers = []
        prev_dim = self.input_dim
        for h in self.hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            if self.activation_layer is not None:
                layers.append(self.activation_layer)
            else:
                pass

            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers)

    def forward(self, X):
        if self.per_output_model:
            # Concatenate predictions from all independent model
            outputs = [m(X) for m in self.model]  # list of [batch_size,1]
            return torch.cat(outputs, dim=1)       # [batch_size, output_dim]
        else:
            return self.model(X)

    def train_step(self, batch):
        """
        Accepts batch as either (X, y) or (X, y, kge)
        Moves tensors to model device automatically.
        """
        # Unpack batch robustly
        if len(batch) == 3:
            X, y, id = batch
        elif len(batch) == 2:
            X, y = batch
            id = None
        else:
            raise ValueError(f"Unexpected batch length {len(batch)}")

        self.model.train()
        self.optimizer.zero_grad()
        preds = self.forward(X)
        loss = self.criterion(preds, y)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu().item())


    def eval_val(self,val_loader):
        # compute validation loss explicitly (to ensure KGE used if present)
        val_losses = []
        for batch in val_loader:
            # batch can be (X,y) or (X,y,kge)
            if len(batch) == 3:
                X, y, id = batch
            elif len(batch) == 2:
                X, y = batch
                id = None
            else:
                raise ValueError("Unexpected batch format in validation loader")


            self.model.eval()

            with torch.no_grad():
                preds = self.predict(X)
                loss = self.criterion(preds, y)
                val_losses.append(float(loss.detach().cpu().item()))

        avg_val_loss = float(np.mean(val_losses))
        return avg_val_loss

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            out = self.forward(X)
        return out

    def save(self, path: str):
        # Save model state dict(s) and - optionally - optimizer state
        state = {
            "model_state": [m.state_dict() for m in self.model] if self.per_output_model else self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict()
        }
        torch.save(state, path)

    def load(self, path: str):
        state = torch.load(path, map_location="cpu")
        if self.per_output_model:
            state_dicts = state["model_state"]
            for m, sd in zip(self.model, state_dicts):
                m.load_state_dict(sd)
        else:
            self.model.load_state_dict(state["model_state"])
        if "optimizer_state" in state:
            try:
                self.optimizer.load_state_dict(state["optimizer_state"])
            except Exception:
                pass
