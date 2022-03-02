import numpy as np
from activation_layer import ActivationLayer


class TanhLayer(ActivationLayer):
    """ Models the Tanh activation part of a fully-connected layer. 
        i.e. a = tanh(z) """

    @staticmethod
    def forward_propagate(z):
        raise NotImplementedError()

    @staticmethod
    def backward_propagate(dJ_da, cache):
        raise NotImplementedError()
    