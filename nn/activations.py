import numpy as np

class ReLUActivation:
    def __init__(self) -> None:
        pass
    
    def __call__(
        self, 
        input_array: np.ndarray
    ) -> np.ndarray:
        """
        ReLU Activation Function
        
        TBD
        """
        return np.max(0, input_array)
    

class SoftmaxActivation:
    def __init__(self) -> None:
        pass
    
    def __call__(
        self, 
        input_array: np.ndarray
    ) -> np.ndarray:
        """
        ReLU Activation Function
        
        TBD
        """
        # max along axis=1(check among columns, horizontal direction) and shape[0] same with inputs
        exps = np.exp(input_array - np.max(input_array, axis=1, keepdims=True)) 
        return exps / np.sum(exps, axis=1, keepdims=True)