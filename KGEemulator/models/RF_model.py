import torch
import torch.nn as nn
import torch.optim as optim
from .base_model import BaseModel

from sklearn.ensemble import RandomForestRegressor
import joblib
import numpy as np
from .base_model import BaseModel


class RandomForestModel(BaseModel):
    def __init__(self, config: dict):
        super().__init__(config)

        # Get parameters from YAML config
        self.n_estimators = config.get("n_estimators", 100)
        self.max_depth = config.get("max_depth", None)
        self.random_state = config.get("random_state", 42)
        self.min_samples_leaf = config.get("min_samples_leaf", 1)
        self.max_features= config.get("max_features","auto")

        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            max_depth=self.max_depth,
            min_samples_leaf = self.min_samples_leaf,
            n_jobs=-1,
            max_features=self.max_features
        )


        # In sklearn, we don’t define optimizer/criterion like in DNN
        self.criterion = None
        self.optimizer = None

    def train_step(self, batch):
        """
        Since RandomForest is not batch-based, we train once on full data.
        We'll call this only once in Trainer.
        """
        X, y = batch
        self.model.fit(X, y)
        return 0.0  # No batch loss to return

    def validation_step(self, batch):
        """
        Compute validation loss on a batch.
        """
        X, y = batch
        preds = self.model.predict(X)
        # Use MSE for compatibility
        loss = np.mean((preds - y) ** 2)
        return loss

    def predict(self, X):
        return self.model.predict(X)

    def eval_val(self,val_loader):
        pass


    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)