import numpy as np


class Layer(object):
    """ Abstract Base class for a layer """

    @property
    def parameters(self):
        return None

    def __init__(self, n_inputs=0):
        self._saved_values = None
        self._parameter_gradients = None
        self._n_inputs = n_inputs

    def compile(self, n_inputs):
        self._n_inputs = n_inputs

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

    # Functions related to parameters
    def initialize_parameters(self):
        pass

    def save_parameter_gradients(self, *parameter_gradients):
        self._parameter_gradients = parameter_gradients

    def load_parameter_gradients(self):
        if len(self._parameter_gradients)==1:
            return self._parameter_gradients[0]
        return self._parameter_gradients

    def update_parameters(self, learning_rate):
        pass
