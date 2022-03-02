import numpy as np
from activation_layer import ActivationLayer


class ReLULayer(ActivationLayer):
    """ Models the ReLU activation part of a fully-connected layer. 
        i.e. a = ReLU(z) """

    @staticmethod
    def forward_propagate(z):
        raise NotImplementedError()

    @staticmethod
    def backward_propagate(dJ_da, cache):
        raise NotImplementedError()
