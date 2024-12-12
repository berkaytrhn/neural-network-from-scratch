from abc import ABC, abstractmethod
import numpy as np


class Component(ABC):
    """
    Every used component must be inherited from Component class
    """
    
    def __init__(self):
        self.name = self.__class__.__name__
    
    @abstractmethod
    def forward(self, X: np.ndarray):
        pass
    
    @abstractmethod
    def backward(self, x: np.ndarray):
        pass
    
    def __repr__(self):
        return f"Component '{self.name}'"