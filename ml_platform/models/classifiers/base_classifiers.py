from abc import ABC, abstractmethod
from typing import Any
from ..base_model import BaseModel

class BaseClassifier(BaseModel):
    """
    Base class for all machine learning classifiers.
    """
    @abstractmethod
    def predict_proba(self, X: Any) -> Any:
        """
        Predict the class probabilities for the given input features.

        Parameters:
        X (Any): The input features for prediction.

        Returns:
        Any: The predicted class probabilities.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def classes_(self) -> Any:
        """
        Return the unique class labels.

        Returns:
        Any: The unique class labels.
        """
        raise NotImplementedError