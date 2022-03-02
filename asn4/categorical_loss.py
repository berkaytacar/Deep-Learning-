import numpy as np

from layer import Layer

class CategoricalCrossEntropyLoss(Layer):
    """ Models the Categorical Cross Entropy Loss Cost function """

    @staticmethod
    def forward_propagate(yhat, y):
        raise NotImplementedError()

    @staticmethod
    def backward_propagate():
        raise NotImplementedError()
    