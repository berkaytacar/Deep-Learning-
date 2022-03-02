import numpy as np
from layer_with_params import LayerWithParams


class LinearLayer(LayerWithParams):
    """ Models the linear part of a fully-connected layer. i.e. z = Wx+b """
    
    @property
    def output_shape(self):
        return (self._n_units,)

    @property
    def parameters(self):
        return self._w, self._b

    @property
    def forward_prop_args(self):
        return self.parameters

    def __init__(self, n_units):
        super().__init__()
        self._n_units = n_units  # Number of units in this layer
        self._w = None           # Layer weights
        self._b = None           # Layer biases
      
    def initialize_parameters(self):
        """ Initialize all parameters """
        np.random.seed(0)
        self._w = np.random.randn(self._n_units, self.input_shape[0])*0.01
        self._b = np.zeros((self._n_units, 1)) + 0.01

    def update_parameters(self, learning_rate):
        """ Update parameters used during Gradient Descent """
        dJ_dw, dJ_db = self.load_parameter_gradients()

        self._w += -learning_rate * dJ_dw
        self._b += -learning_rate * dJ_db

    @staticmethod
    def forward_propagate(x, w, b):
        raise NotImplementedError()

    @staticmethod
    def backward_propagate(dJ_dz, cache):
        raise NotImplementedError()
