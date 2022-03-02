from layer import Layer

class ActivationLayer(Layer):
    """ Base class for an Activation layer """

    @property
    def output_shape(self):
        return self._input_shape
