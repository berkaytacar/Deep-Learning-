import numpy as np
from layer import Layer

class DropoutLayer(Layer):

    @property
    def output_shape(self):
        return self._input_shape

    @property
    def forward_prop_args(self):
        return (self._keep_prob, )

    def __init__(self, keep_prob):
        super().__init__()
        self._keep_prob = keep_prob

    @staticmethod
    def forward_propagate(x, keep_prob):
        raise NotImplementedError

    @staticmethod
    def backward_propagate(dJ_dd, cache):
        raise NotImplementedError
