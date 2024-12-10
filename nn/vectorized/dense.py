import numpy as np

class DenseLayer:
    """
    A class used to represent a single layer in a neural network.

    This class implements a fully connected layer where each neuron
    is connected to every neuron in the previous layer.

    Attributes
    ----------
    weights : np.ndarray
        The weights of the layer, of shape (n_features, n_neurons).
    biases : np.ndarray
        The biases of the layer, of shape (1, n_neurons).
    inputs : np.ndarray
        The input data to the layer, set during the forward pass of shape (m, n_features); m -> number of samples in the batch

    Methods
    -------
    forward(inputs)
        Performs the forward pass of the neural network layer.
    
    backward(d_output)
        Performs the backward pass of the neural network layer and updates weights and bias.
    """

    def __init__(
        self,
        n_features: int,
        n_neurons: int
    ) -> None:
        """
        Constructor of one network layer

        Parameters:
                n_features (int): number of features
                n_neurons  (int): number of neurons

        Returns:
                None
        """
        
        self.weights = np.random.randn(n_features, n_neurons) # (n_features,n_neurons), each neuron has a set of weights same number as n_features
        self.biases = np.zeros((1,n_neurons)) # (1,n_neurons) one for all neurons
        
        # empty initializations
        self.inputs = np.array([])
        self.outputs = np.array([])


    def forward(
        self,
        inputs: np.ndarray
    ) -> np.ndarray:
        """
        Forward pass of one layer of network
        Parameters:
                inputs (np.ndarray): (m,n_features) input matrix; m -> number of samples

        Returns:
                np.ndarray: dot product of inputs and weights with shape (m,n_neurons)
        """
        self.inputs = inputs # for later backward propagation
        return np.dot(self.inputs, self.weights) + self.biases
        
    def backward(
        self,
        d_output: np.ndarray
    ) -> None:
        """
        Backward pass of one layer of network
        Parameters:
                d_output (np.ndarray): (n_features, n_neurons), gradient values for this layer
        Returns:
                None
        """
        # TBD
        print(d_output)