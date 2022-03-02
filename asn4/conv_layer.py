import numpy as np
from layer_with_params import LayerWithParams


class Conv2DLayer(LayerWithParams):
    
    @property
    def output_shape(self):
        h_in, w_in, c_in = self.input_shape
        return (h_in-self._kernel_size[0]+1, 
                w_in-self._kernel_size[1]+1,
                self._n_filters)

    @property
    def parameters(self):
        return self._w, self._b

    @property
    def forward_prop_args(self):
        return (self._w, self._b)

    def __init__(self, n_filters, kernel_size):
        super().__init__()

        assert kernel_size[0] == kernel_size[1], "Only square filters supported"

        self._n_filters = n_filters  # Number of units in this layer
        self._kernel_size = kernel_size

        self._w = None           # Layer weights
        self._b = None           # Layer biases
      
    def initialize_parameters(self):
        """ Initialize all parameters """
        np.random.seed(0)
        _, _, n_channels_in = self.input_shape

        # He Normal
        # self._w = np.random.randn(self._n_filters, self._kernel_size[0], 
        #                           self._kernel_size[1], n_channels_in)*0.01

        # Guassian
        # self._w = np.random.randn(self._n_filters, self._kernel_size[0], 
        #                           self._kernel_size[1], n_channels_in)*np.sqrt(1/np.prod(self.input_shape))

        #glorot_uniform
        v = np.sqrt(6/10.)/np.sqrt(np.prod(self.input_shape)+np.prod(self.output_shape))
        self._w = np.random.uniform(-v, v, 
            (self._n_filters, self._kernel_size[0], self._kernel_size[1], n_channels_in))

        self._b = np.zeros((self._n_filters,)) + 0.01

    def update_parameters(self, learning_rate):
        """ Update parameters used during Gradient Descent """
        dJ_dw, dJ_db = self.load_parameter_gradients()

        self._w += -learning_rate * dJ_dw
        self._b += -learning_rate * dJ_db

    @staticmethod
    def forward_propagate(v_in, w, b):
        raise NotImplementedError()

    @staticmethod
    def backward_propagate(dJ_dz, cache):
        raise NotImplementedError()
