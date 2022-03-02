import numpy as np


class Model(object):
    """ Models a Neural Network """

    # Functions related to model creation
    def __init__(self, layers):
        self._layers = layers
        self._loss_layer = None

    def compile(self, loss):
        """ Compile the model and specify a Loss function """
        self._loss_layer = loss
        for prev_layer, next_layer in zip(self._layers, self._layers[1:]):
            next_layer.compile(prev_layer.output_shape)
   
    # Functions related to forward and backward propagation
    def _forward_propagate(self, x):
        """ Perform forward propagation through the network
        
        Inputs: 
            x: Inputs to the network. Shape (nx, m) where nx is the number of
               input features and m is number of samples
        Returns:
            a: The network output (e.g. prediction).  Shape is dependent on
               the model architecture. For example, for softmax output, the 
               output shape would be (nc, m) where nc is the number of 
               classification classes and m is number of samples.
        """
        a = x
        for layer in self._layers:
            if layer.forward_prop_args:
                a, cache = layer.forward_propagate(a, *(layer.forward_prop_args))
            else:
                a, cache = layer.forward_propagate(a)

            layer.save_values(cache)
        return a

    def _backward_propagate(self, dJ_dyhat):
        """ Perform backward propagation through the network
        
        Inputs: 
            dJ_dyhat: Gradients of cost with respect to the model outputs.
                      Shape is (nc, m) where nc is the number of model outputs,
                      and m is number of samples
        Returns:
            None
        """
        dJ = dJ_dyhat
        for layer in reversed(self._layers):
            cache = layer.load_values()
            gradients = layer.backward_propagate(dJ, cache)
            if layer.has_params:
                dJ = gradients[0]
                layer.save_parameter_gradients(*gradients[1:])
            else:
                dJ = gradients

    # Functions related to training
    def _update_parameters(self, learning_rate):
        """ Update parameters in each layer """
        for layer in self._layers:
            if layer.has_params:
                layer.update_parameters(learning_rate)

    def train(self, x, y, validation_data, epochs, learning_rate, print_freq=50):
        """ Train the model using classic gradient descent """
        return self.train_sgd(x, y, validation_data, epochs, learning_rate, 
                              x.shape[1], print_freq)

    def train_sgd(self, x, y, validation_data, epochs, learning_rate, 
                  batch_size=200, print_freq=50):
        """ Train the model using mini-batch gradient descent
        
        Inputs:
            x: Inputs to the network. Shape (nx, m) where nx is the number of
               input features and m is number of samples
            y: True labels for x.
            epochs: Number of training iterations
            learning_rate: learning rate for parameter update
        """
        if validation_data:
            x_valid, y_valid = validation_data
        
        costs_train = [] # For tracking cost per iteration
        costs_valid = []
        
        # Intialize all model parameters
        for layer in self._layers:
            if layer.has_params:
                layer.initialize_parameters()

        for iter in range(epochs):
            for j in range(0, x.shape[1], batch_size):

                # Forward Propagate all the way through Cost function
                yhat = self._forward_propagate(x[...,j:j+batch_size])
                cost, cache = self._loss_layer.forward_propagate(yhat, y[...,j:j+batch_size])
                self._loss_layer.save_values(cache)
                
                # Backward propagate starting from cost function
                cache = self._loss_layer.load_values()
                dJ_dyhat = self._loss_layer.backward_propagate(cache)
                self._backward_propagate(dJ_dyhat)

                self._update_parameters(learning_rate)

            # Compute Metrics
            yhat = self._forward_propagate(x)
            cost, _ = self._loss_layer.forward_propagate(yhat, y)
            costs_train.append(cost)

            if validation_data:
                yhat_valid = self._forward_propagate(x_valid)
                cost_valid, _ = self._loss_layer.forward_propagate(yhat_valid, y_valid)
                costs_valid.append(cost_valid)
                
            if iter % print_freq==0:
                if validation_data:
                    print('{0} loss: {1:.8f} - accuracy: {2:.3f} - val_loss: {3:.8f} - val_accuracy: {4:.3f}'.format(
                        iter,  cost, self.compute_accuracy(yhat, y),
                        cost_valid, self.compute_accuracy(yhat_valid, y_valid)))
                else:
                    print('{0} loss: {1:.8f} - accuracy: {2:.3f}'.format(
                        iter,  cost, self.compute_accuracy(yhat, y)))

        if validation_data:
            return costs_train, costs_valid
        return costs_train
    
    def compute_accuracy(self, yhat, y):
        assert(yhat.shape == y.shape)
        return np.mean(np.argmax(yhat, axis=0) == np.argmax(y, axis=0))

    def predict(self, X):
        yhat = self._forward_propagate(X)
        return np.argmax(yhat, axis=0), yhat

