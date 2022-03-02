import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


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

def load_mnist():
    (X_train, Y_train), (X_validation, Y_validation) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.transpose((1,2,0))
    X_train = X_train.reshape((28, 28, 1, -1))
    X_validation = X_validation.transpose((1,2,0))
    X_validation = X_validation.reshape((28, 28, 1, -1))

    X_train = X_train.astype('float')
    X_train = X_train/255

    X_validation = X_validation.astype('float')
    X_validation = X_validation/255

    print("X (training)   shape:", X_train.shape)
    print("Y (training)   shape:", Y_train.shape)
    print("X (validation) shape:", X_validation.shape)
    print("Y (validation) shape:", Y_validation.shape)
    print()

    return X_train, Y_train, X_validation, Y_validation

def load_cifar10():
    (X_train, Y_train), (X_validation, Y_validation) = tf.keras.datasets.cifar10.load_data()
    X_train = X_train.transpose((1,2,3,0))
    X_validation = X_validation.transpose((1,2,3,0))

    X_train = X_train.astype('float')
    X_train = X_train/255

    X_validation = X_validation.astype('float')
    X_validation = X_validation/255

    Y_train = np.squeeze(Y_train)
    Y_validation = np.squeeze(Y_validation)

    print("X (training)   shape:", X_train.shape)
    print("Y (training)   shape:", Y_train.shape)
    print("X (validation) shape:", X_validation.shape)
    print("Y (validation) shape:", Y_validation.shape)
    print()

    return X_train, Y_train, X_validation, Y_validation


def plot_sample(X, Y, img_dims, n_c, n):
    
    fig, axes = plt.subplots(nrows=n_c, ncols=n, figsize=(8,8))

    for i in range(n_c):
        indices_for_this_class = np.squeeze(np.argwhere(np.squeeze(Y)==i))
        selection = np.random.choice(indices_for_this_class, n)
        
        for j in range(n):
            axes[i][j].set_axis_off()
            axes[i][j].imshow(X[:,selection[j]].reshape(*img_dims))

def plot_costs(training_costs, validation_costs=None):
    if validation_costs:
        plt.plot(training_costs, label='training')
        plt.plot(validation_costs, label='validation')
        plt.legend()
    else:
        plt.plot(training_costs)
    plt.title('Training Cost')
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.show()


def plot_training_history(history, title_str=''):
    """ Plot Keras training history """
#     acc = history.history['acc']
#     val_acc = history.history['val_acc']

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
    ymax = max([max(loss), max(val_loss)])

    axes = plt.gca()
    axes.set_xlim([0, len(loss)])
    axes.set_ylim([ymin, ymax])

    plt.legend(loc=0)
    plt.figure()
    plt.show()


def numeric_gradient(f, x, df, h=1e-5):
    x_flat = x.reshape(-1)
    dx_flat = np.zeros(x_flat.shape)
    for i in range(dx_flat.shape[0]):
        x_orig = x_flat[i]
        x_flat[i] = x_orig - h
        f_minus = f(x)

        x_flat[i] = x_orig + h
        f_plus = f(x)

        x_flat[i] = x_orig

        df_dx = (f_plus-f_minus)/(2*h)
        dx_flat[i] = np.sum(np.multiply(df_dx, df))

    return dx_flat.reshape(x.shape)

def rel_error(actual, expected, epsilon=1e-8):
    return np.amax(np.absolute(expected-actual) /  np.clip(np.absolute(expected), epsilon, None))
