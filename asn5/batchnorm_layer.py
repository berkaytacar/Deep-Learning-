import numpy as np
from layer_with_params import LayerWithParams

class BatchNormLayer(LayerWithParams):
    
    @property
    def output_shape(self):
        return self.input_shape

    @property
    def parameters(self):
        return self._gamma, self._beta

    @property
    def forward_prop_args(self):
        return (self._gamma, self._beta)

    def __init__(self):
        super().__init__()
        self._gamma = None
        self._beta = None
      
    def initialize_parameters(self):
        """ Initialize all parameters """
        if len(self._input_shape) == 1:
            n = self._input_shape[0]
            self._gamma = np.ones((n, 1))
            self._beta = np.zeros((n, 1))
        else:
            raise NotImplementedError

    def update_parameters(self, learning_rate):
        """ Update parameters used during Gradient Descent """
        dJ_dgamma, dJ_dbeta = self.load_parameter_gradients()

        self._gamma += -learning_rate * dJ_dgamma
        self._beta  += -learning_rate * dJ_dbeta

    @staticmethod
    def forward_propagate(x, gamma, beta):
        raise NotImplementedError

    @staticmethod
    def backward_propagate(dJ_dz, cache):
        raise NotImplementedError