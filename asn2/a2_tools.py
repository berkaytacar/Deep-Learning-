import numpy as np


def load_swirls():
    with np.load('a2_data.npz') as data:
        x = data['swirls_x'] 
        y = data['swirls_y']
    return x, y

def load_noisy_circles():
    with np.load('a2_data.npz') as data:
        x = data['circles_x'] 
        y = data['circles_y']
    return x, y

def load_noisy_moons():
    with np.load('a2_data.npz') as data:
        x = data['moons_x']
        y = data['moons_y']
    return x, y

def load_partitioned_circles():
    with np.load('a2_data.npz') as data:
        x = data['partitioned_circles_x']
        y = data['partitioned_circles_y']
    return x, y

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def compute_accuracy(X, Y, W1, B1, W2, B2):
    """ Compute the accuracy of the model 
    
    Inputs:
        X:  NumPy array of feature data of shape (n, m)
        Y:  NumPy array of labels of shape (m,)
        W1: NumPy array of first layer of parameters with shape (n_h, n_y)
        B1: NumPy array of second layer bias parameters with shape (n_h, 1)
        W2: NumPy array of second layer of parameters with shape (1, n_h)
        B2: NumPy array of second layer bias parameters with shape (1, 1)

    Returns:
        NumPy array (m, ) of predictions.  Values are 0 or 1.
    """
    Y_predicted = predict(X, W1, B1, W2, B2)
    accuracy = np.mean(Y_predicted == Y)

    return accuracy


def load_data_and_hyperparams(data_set_name):
    if data_set_name == 'noisy_circles':
        X, Y = load_noisy_circles()
        n_iters = 1400
        learning_rate = 1.5
    elif data_set_name == 'noisy_moons':
        X, Y = load_noisy_moons()
        n_iters = 1000
        learning_rate = 1.8
    elif data_set_name == 'swirls':
        X, Y = load_swirls()
        n_iters = 700
        learning_rate = 1.2
    elif data_set_name == 'flower':
        X, Y = load_flower()
        n_iters = 500
        learning_rate = 1.2
    else:
        raise ValueError("Unexpected value '{0}' for data_set_name".format(data_set_name))

    return X, Y, n_iters, learning_rate
