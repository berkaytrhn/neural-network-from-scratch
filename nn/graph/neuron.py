import numpy as np
from nn.utils import Component

class Neuron(Component):

    def __init__(
        self,
        output_shape: int,# (1, n) single neuron, this number should be number of neurons on previous layer(or number of samples if this is the input layer)
        learning_rate:float=0.1
    ) -> None:
        
        self.weights = np.randn(output_shape) # (n,1), value for each feature(or sample if input)
        self.bias = np.random.randn() # (1), single number
        self.learning_rate = learning_rate
        
        # empty initializations
        self.inputs = np.array([])
        self.output = np.array([])
            
    def forward(
            self,
            inputs: np.ndarray # (m,n)
    ) -> None:
        """
        Forward propagation -> matmul.
        :param inputs: Input array
        :return: Output of the neuron, shape (m,1)
        """
        self.inputs = inputs
        self.output = np.dot(self.inputs, self.weights) + self.bias  # matrix multip of (m,n) and (n,1) -> (m,!) + bias ; with broadcast, behaves like (m, 1) --> Resulted with shape (m,1)
        
    def backward(self, d_output) -> None:
        """
        Backward pass over a neuron
        
        Gradient Formula -> dL/dW = dL/dY * dY/dW 
            - d_output is dL/dY which is provided for us from next neurons, with shape(m,1)
            - dY/dW = d(X*W+b)/dW = X (self.input with shape (m,n))
            - (dL/dY * dY/dW) = element wise multiplication of T(inputs) and derivatives to reach shape of weights -> (n,m) * (m,1) = (n,1) -> shape of weights
            - dY/db = 1 since f(x) = y = X*W+b
        
        :param d_output: Gradient of the loss with respect to the output of the neuron, shape (m,1)
        """
        # Gradient calculation
        # Taking average of gradients since we need to make all neurons contribute similarly
        d_weights = np.dot(self.inputs.T, d_output) / self.inputs.shape[0] 
        d_bias = np.mean(d_output)
        
        # Update weights and bias
        self.weights -= self.learning_rate * d_weights
        self.bias -= self.learning_rate * d_bias