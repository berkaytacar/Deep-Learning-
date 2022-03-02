import numpy as np
from layer import Layer

class InputLayer(Layer):
    
    @property
    def output_shape(self):
        return self._input_shape

    @property
    def input_shape(self):
        return ()

    def __init__(self, input_shape):
        super().__init__()
        if isinstance(input_shape, int):
            input_shape = (input_shape,)
        self._input_shape = input_shape
        
    def compile(self):
        super.compile(0)

    def forward_propagate(self, X):
        return X, None

    def backward_propagate(self, dl_dx, cache):
        return dl_dx