import numpy as np
from layer import Layer

class FlattenLayer(Layer):
    
    @property
    def output_shape(self):
        return (np.prod(self._input_shape),)

    @staticmethod
    def forward_propagate(x):
        raise NotImplementedError

    @staticmethod
    def backward_propagate(dJ_df, cache):
        raise NotImplementedError