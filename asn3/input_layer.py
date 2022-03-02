import numpy as np
from layer import Layer

class InputLayer(Layer):
    
    @property
    def n_outputs(self):
        return self._input_shape
        
    def __init__(self, input_shape):
        super().__init__()
        self._input_shape = input_shape
        
    def compile(self, n_inputs):
        self._n_inputs = 0

    def forward_propagate(self, X):
        return X, None

    def backward_propagate(self, dl_dx, cache):
        return dl_dx