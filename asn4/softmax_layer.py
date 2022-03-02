import numpy as np
from activation_layer import ActivationLayer


class SoftmaxLayer(ActivationLayer):
    """ Models the Softmax activation part of a fully-connected layer. 
        i.e. yhat = softmax(z) """

    @staticmethod
    def forward_propagate(z):
        raise NotImplementedError()

    @staticmethod
    def backward_propagate(dJ_dyhat):
        raise NotImplementedError()
