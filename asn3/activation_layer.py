from layer import Layer

class ActivationLayer(Layer):
    """ Base class for an Activation layer """

    @property
    def n_outputs(self):
        return self._n_inputs
