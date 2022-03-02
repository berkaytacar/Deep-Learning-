import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def one_hot(Y, n_c):
    """ Convert labels to one-hot encoding 
    
    Inputs:
        Y: NumPy array of shape (m,) for the labels
        n_c: Int for number of classes
        
    Returns
        NumPy array of shape (n_c, m) with the one-hot encoded labels
    """
    Y_onehot = np.zeros((Y.size, n_c))
    Y_onehot[np.arange(Y.size), Y] = 1
    return Y_onehot.T


def plot_sample(X, Y, img_dims, n_c, n):
    
    fig, axes = plt.subplots(nrows=n_c, ncols=n, figsize=(8,8))

    for i in range(n_c):
        indices_for_this_class = np.squeeze(np.argwhere(np.squeeze(Y)==i))
        selection = np.random.choice(indices_for_this_class, n)
        
        for j in range(n):
            axes[i][j].set_axis_off()
            axes[i][j].imshow(X[:,selection[j]].reshape(*img_dims))

            
def load_blobs():
    x = np.load('blobs_x.npy')
    y = np.load('blobs_y.npy')
    return x, y


def load_mnist():
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.transpose((1,2,0))
    X_train = X_train.reshape((784, -1))
    X_test = X_test.transpose((1,2,0))
    X_test = X_test.reshape((784, -1))

    return X_train, Y_train, X_test, Y_test


def plot_decision_boundary(X, Y, model):
    """ Plot decision boundary 
    
    Inputs:
        X:  NumPy array of training feature data of shape (n, m)
        Y:  NumPy array of training labels of shape (m,)
        model: function that makes predictions
    """
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    
    # Generate a grid of points with distance h between them
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()].T)
    Z = Z.reshape(xx.shape)
    
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=Y, cmap=plt.cm.Spectral)

def plot_costs(costs):
    plt.plot(costs)
    plt.title('Training Cost')
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.show()

def plot_training_and_validation_costs(costs_train, costs_validation):
    epochs = range(len(costs_train))
    plt.plot(epochs, costs_train, 'r', label='Training loss')
    plt.plot(epochs, costs_validation, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    
def plot_training_history(history, title_str=''):
    """ Plot Keras training history """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy ' + title_str)

    ymin = min([min(acc), min(val_acc)])

    axes = plt.gca()
    axes.set_xlim([0, len(acc)])
    axes.set_ylim([ymin, 1])

    plt.legend(loc=0)
    plt.figure()
    plt.show
    
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, loss, 'b', label='Validation loss')
    plt.title('Training and validation loss ' + title_str)

    ymin = min([min(loss), min(val_loss)])

    axes = plt.gca()
    axes.set_xlim([0, len(loss)])
    axes.set_ylim([ymin, 1])

    plt.legend(loc=0)
    plt.figure()
    plt.show()

    