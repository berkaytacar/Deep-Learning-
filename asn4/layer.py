import numpy as np


class Layer(object):
    """ Abstract Base class for a layer """

    @property
    def output_shape(self):
        raise NotImplementedError

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def has_params(self):
        return False

    @property
    def forward_prop_args(self):
        return None

    def __init__(self):
        self._saved_values = None
        self._input_shape = None

    def compile(self, input_shape):
        self._input_shape = input_shape

    # Functions related to caching
    def save_values(self, values):
        self._saved_values = values
        
    def load_values(self):
        return self._saved_values

    # Functions related to forward and backward propagation
    def forward_propagate(self, X):
        raise NotImplementedError()

    def backward_propagate(self, dJ):
        raise NotImplementedError()
