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