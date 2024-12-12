import numpy as np
from abc import ABC, abstractmethod
from .component import Component

class Activation(Component):
    """
    Base Class For Activations
    """
    @abstractmethod
    def forward(self, x):
        pass
    @abstractmethod
    def backward(self):
        pass

class ReLU(Activation):
    
    def __init__(self):
        # will be used for storing input values for backprop.
        self.cache=None
    
    def forward(
        self, 
        x: np.ndarray
    ) -> np.ndarray:
        """
        ReLU Activation Function
        """
        # store input value for backward pass
        self.cache=x
        return np.maximum(0, x)
    
    def backward(self, grad_out):
        """
        Derivative of ReLU, 
        1 if >=0, else 0
        """
        # Same shape since relu does not change the shape 
        return grad_out * np.where(self.cache > 0, 1, 0)
    
    
    
class Sigmoid(Activation):
    
    def __init__(self) -> None:
        self.output_cache=None

    def forward(self, z:np.ndarray):
        self.output_cache = 1 / (1 + np.exp(-z))
        return self.output_cache

    def backward(self, grad_out):
        # Same shape since sigmoid does not change the shape 
        return grad_out * self.output_cache * (1 - self.output_cache)
    
class Softmax(Activation):
    
    def forward(
        self, 
        input_array: np.ndarray
    ) -> np.ndarray:
        """
        Softmax activation
        TBD
        """
        # subtract from max to avoid exponential explosion to infinity, now exp result will be max 1
        exps = np.exp(input_array - np.max(input_array, axis=1, keepdims=True)) 
        
        # max along axis=1(check among columns, horizontal direction) and shape[0] same with inputs
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def backward(self):
        pass