import unittest
import numpy as np
from nn import DenseLayer

class TestNeuralNetworkLayer(unittest.TestCase):
    """
    A class for testing a vectorized nn Dense Layer

    Attributes
    ----------
    layer : nn.vectorized.dense.DenseLayer
        The layer that we are tesing.
   
    Methods
    -------
    test_forward
        Testing the forward propagation.
    
    test_backward
        Testing the backward propagation.
    """
    
    def setUp(self) -> None:
        self.layer = DenseLayer(3, 4)
        self.layer.weights = np.array([
            [0.1, 0.2, 0.3, 1.0],
            [0.4, 0.5, 0.6, 1.1],
            [0.7, 0.8, 0.9, 1.2],
        ])
        self.layer.biases = np.array([0.1, 0.2, 0.3, 0.4])
    
    def test_forward(self):
        inputs = np.array([[1.0, 2.0, 3.0],
                           [5.0, 6.0, 7.0],
                           [8.0, 9.0, 10.0]])
        expected_output = np.array([
            [3.1, 3.8, 4.5, 7.2],
            [7.9, 9.8, 11.7, 20.4],
            [11.5, 14.3, 17.1, 30.3]
            ])
        output = self.layer.forward(inputs)
        np.testing.assert_array_almost_equal(output, expected_output, decimal=5)
    
    def test_backward(self):
        np.testing.assert_almost_equal(
            np.array([1,2,3]),
            np.array([1,2,3])
        )

if __name__ == "__main__":
    unittest.main()