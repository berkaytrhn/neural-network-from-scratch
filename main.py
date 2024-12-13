import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from nn.vectorized import Network, DenseLayer
from nn.utils import Sigmoid, ReLU
from nn.utils import MSELoss
from ml_lib.preprocessing import StandardScaler

def main():
    
    
    california_housing = fetch_california_housing()

    X = california_housing["data"]
    y = california_housing["target"]
    y = y.reshape(y.shape[0], 1)
    print(X.shape, y.shape)
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # standarization for overcome overflow
    # scaler = StandardScaler()
    
    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)
    
    
    nn = Network()
    #X = np.random.randn(20, 5)
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
    
    _loss = loss.forward(y, interm)
    print(_loss)
    
    loss_grad = loss.backward(y,interm)
    
    # no loss 
    nn.backward(loss_grad)
    
    # TODO: Use optimizers.SGD to optimize using layer.grad in a loop, then implement training loop to check if it is learning!
    
    


if __name__ == "__main__":
    
    main()
    