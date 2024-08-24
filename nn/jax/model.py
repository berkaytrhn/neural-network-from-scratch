import jax.numpy as jnp
import jaxlib
import time
from typing import Callable
from Initialization import Initialization

class DenseLayer:
    def __init__(
            self, 
            input_size: int, 
            neurons: int,
            seed: int= time.time_ns(), # optional, provide seed for controlled experiments
            initialization_method: Callable[[tuple, int], jnp.ndarray] = Initialization.xavier_init
        ) -> None:
        """
        Initialize the one layer of network with 
        given initialization function(default xavier init)
        
        """
        
        self.weights = initialization_method((input_size, neurons), seed)
        pass

    def __call__(X: jnp.ndarray) -> None:
        pass
    
    def xavier_init():


class NeuralNetwork:
    def __init__(self) -> None:
        pass

    def forward(X):
        pass

    def backward():
        pass

    def train():
        pass

    def predict():
        pass