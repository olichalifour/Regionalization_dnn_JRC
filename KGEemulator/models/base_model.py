# models/base_model.py
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Abstract base class for all models."""

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def train_step(self, batch):
        """Perform a single training step on a batch."""
        pass

    @abstractmethod
    def predict(self, X):
        """Generate predictions for input data X."""
        pass

    @abstractmethod
    def save(self, path: str):
        """Save model parameters to file."""
        pass


    @abstractmethod
    def load(self, path: str):
        """Load model parameters from file."""
        pass

    @abstractmethod
    def eval_val(self,val_loader):

        pass