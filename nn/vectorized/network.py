from nn.vectorized import DenseLayer
import numpy as np
from typing import List, Union
from nn.utils import Component


class Network(Component):
    
    def __init__(self):
        self.layers: List[DenseLayer]=list()
    
    
    def add(self, sequence: Union[DenseLayer, List[DenseLayer]]):
        if isinstance(sequence, list):
            # List of DenseLayer objects
            self.layers.extend(sequence)
        else: 
            # single DenseLayer
            self.layers.append(sequence)

    def forward(self, x:np.ndarray):
        value = x
        for layer in self.layers:
            value = layer.forward(value)
        return value
    
    def __call__(self, x:np.ndarray):
        return self.forward(x)
    
    def backward(self, x:np.ndarray):
        value = x
        for i,layer in enumerate(reversed(self.layers)):
            print(f"Processing Layer: '{i}'")
            grad = layer.backward(value)
            value=grad
    
    