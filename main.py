import numpy as np

from nn.vectorized import Network, DenseLayer
from nn.utils import Sigmoid, ReLU
from nn.utils import MSELoss


def main():
    
    
    nn = Network()
    X = np.random.randn(20, 5)
    layer1 = DenseLayer(5, 6) # 20,6
    layer2 = DenseLayer(6,12) # 20,12
    layer3 = DenseLayer(12,6) # 20,6
    layer4 = DenseLayer(6,1) # 20,1 -> w.shape=6,1 ; inputs.shape=20,6
    relu = ReLU()
    loss = MSELoss()
    
    
    layers = [
        layer1,
        layer2,
        layer3,
        layer4,
        relu
    ]
    nn.add(layers)
      
    
    interm = nn(X)
    
    _loss = loss.forward(interm, interm+1)
    print(_loss)
    
    loss_grad = loss.backward(interm,interm+1)
    
    # no loss 
    nn.backward(loss_grad)


if __name__ == "__main__":
    
    main()
    