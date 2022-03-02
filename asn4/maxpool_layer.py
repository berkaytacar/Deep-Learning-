import numpy as np
from layer import Layer

class MaxpoolLayer(Layer):

    @property
    def output_shape(self):
        h_in, w_in, c_in = self._input_shape
        px, py = self._pool_size
        return (int(h_in/px), int(w_in/py), c_in)

    @property
    def forward_prop_args(self):
        return (self._pool_size, )

    def __init__(self, pool_size):
        super().__init__()
        self._pool_size = pool_size

    @staticmethod
    def forward_propagate(v_in, pool_size):
        raise NotImplementedError

    @staticmethod
    def backward_propagate(dJ_da, cache):
        raise NotImplementedError
