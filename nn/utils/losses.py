from .component import Component
import numpy as np

class Loss(Component):
    
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass
    

class BCELoss(Loss):
    """TODO: Implement"""
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass
    
    
class MSELoss(Loss):
    """TODO: Implement"""
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray):
        return np.mean(np.square(y_true-y_pred))
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray):
        n_samples = len(y_true)
        factor = (2 / n_samples)
        residual = (y_true - y_pred)
        return factor * residual