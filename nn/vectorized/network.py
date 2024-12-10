from dense import DenseLayer
import numpy as np
from nn.utils import Sigmoid
from typing import List


class Network:
    
    def __init__(self):
        self.layers: List[DenseLayer]=list()
    
    
    def add(self, layer: DenseLayer):
        self.layers.append(layer)
        

    def __call__(self, x:np.ndarray):
        value = x
        for layer in self.layers:
            value = layer.forward(value)

        print(value)
    def backward(self):
        pass
    
    
if __name__ == "__main__":
    
    
    nn = Network()
    X = np.random.randn(20, 5)
    layer1 = DenseLayer(5, 6)
    layer2 = DenseLayer(6,12)
    layer3 = DenseLayer(12,6)
    layer4 = DenseLayer(6,1)
    sig = Sigmoid()
    
    nn.add(
        layer1,
        layer2,
        layer3,
        layer4,
        sig
    )
    
    outputs = nn(X)
    print(outputs)