import os
import numpy as np
from evaluation.metrics import compute_metrics
from data.dataloader import inverse_transform_y
import torch

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.checkpoint_dir = config["logging"]["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.history = {"train_loss": [], "val_loss": []}
        self.metrics = config.get("evaluation", {}).get("metrics", [])
        for m in self.metrics:
            self.history[f"val_{m}"] = []
        # track best model
        self.best_val_loss = float("inf")
        self.best_model_path = os.path.join(self.checkpoint_dir, "best_model.pt")

    def train(self, train_loader, val_loader=None):
        epochs = self.config["training"].get("epochs", 1)

        for epoch in range(1, epochs + 1):
            train_losses = []
            for batch in train_loader:
                loss = self.model.train_step(batch)
                train_losses.append(loss)

            avg_train_loss = float(np.mean(train_losses))
            self.history["train_loss"].append(avg_train_loss)
            print(f"[Epoch {epoch}/{epochs}] Train Loss: {avg_train_loss:.6f}")

            # Validation
            if val_loader is not None:
                # compute val metrics (this runs predictions & metric computation)
                val_metrics = self.evaluate(val_loader, log=False)
                print(f"           Validation metrics: {val_metrics}")

                avg_val_loss = self.model.eval_val(val_loader)
                self.history["val_loss"].append(avg_val_loss)

                # Log metrics to history
                for m in self.metrics:
                    self.history[f"val_{m}"].append(val_metrics.get(m, 0.0))

                print(f"           Validation Loss: {avg_val_loss:.6f}")
                # check if best so far
                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    print(f"           New best model! Saving to {self.best_model_path}")
                    self.model.save(self.best_model_path)
            # checkpointing
            if epoch % self.config["logging"].get("save_every", 5) == 0:
                ckpt_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch}.pt")
                self.model.save(ckpt_path)

        return avg_train_loss

    def evaluate(self, data_loader, log=True):
        preds_list, targets_list = [], []
        for batch in data_loader:
            if len(batch) == 3:
                X, y, _kge = batch
            elif len(batch) == 2:
                X, y = batch
            else:
                raise ValueError("Unexpected batch format in evaluate()")

            y_hat = self.model.predict(X)

            # convert to numpy
            if isinstance(y_hat, torch.Tensor):
                y_hat = y_hat.detach().cpu().numpy()
            if isinstance(y, torch.Tensor):
                y = y.detach().cpu().numpy()

            preds_list.append(y_hat)
            targets_list.append(y)

        preds = np.concatenate(preds_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)

        # inverse transform
        Y_scaler = self.config["data"]["Y_scaler"]
        preds = inverse_transform_y(preds, Y_scaler)
        targets = inverse_transform_y(targets, Y_scaler)

        metrics = compute_metrics(preds, targets, self.config["evaluation"]["metrics"])
        if log:
            print("Evaluation:", metrics)
        return metrics