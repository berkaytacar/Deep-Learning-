import numpy as np
from layer import Layer

class LayerWithParams(Layer):

    @property
    def parameters(self):
        return None

    @property
    def has_params(self):
        return True

    def __init__(self):
        super().__init__()
        self._parameter_gradients = None

    # Functions related to parameters
    def initialize_parameters(self):
        raise NotImplemented

    def save_parameter_gradients(self, *parameter_gradients):
        self._parameter_gradients = parameter_gradients

    def load_parameter_gradients(self):
        if len(self._parameter_gradients)==1:
            return self._parameter_gradients[0]
        return self._parameter_gradients

    def update_parameters(self, learning_rate):
        raise NotImplemented
