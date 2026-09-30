from abc import ABC, abstractmethod
from typing import Any

class BaseModel(ABC):
    """
    Base class for all machine learning models.
    """

    @abstractmethod
    def fit(self, X: Any, y: Any) -> "BaseModel":
        """
        Fit the model to the training data.

        Parameters:
        X (Any): The input features for training.
        y (Any): The target labels for training.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """
        Predict the target labels for the given input features.

        Parameters:
        X (Any): The input features for prediction.

        Returns:
        Any: The predicted target labels.
        """
        raise NotImplementedError

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Return model hyperparameters."""
        raise NotImplementedError

    @abstractmethod
    def set_params(self, **params: Any) -> "BaseModel":
        """Update model hyperparameters."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Return whether the model has been trained."""
        raise NotImplementedError