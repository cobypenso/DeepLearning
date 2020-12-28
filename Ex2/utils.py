# Coby Penso, 208254128
import numpy as np
import matplotlib.pyplot as plt
import pickle

from numpy.testing._private.utils import break_cycles

def softmax(z):
    """implement the softmax functions
    @params: z - array to apply softmax on
    @returns: result - softmax
    """
    max_logist = np.max(z, axis = 1, keepdims=True)
    normalized_z = z - max_logist
    exp_list = np.exp(normalized_z)
    result = exp_list * (1 / np.sum(exp_list, axis=1, keepdims=True))
    result = result.reshape((len(z),-1))

    return result

def neg_log_loss(pred, label):
    """implement the negative log loss
    @params: 
        pred
        label
    @returns: loss
    """
    log_likelihood = -np.log(pred[range(pred.shape[0]),label])
    loss = np.sum(log_likelihood) / pred.shape[0]
    return loss

def split_train_and_val(X, Y, precentage = 0.8):
    '''
    @params:
        X  - images
        Y  - labels
        precentage - precentage to be training and the rest would be validation
    @returns: train_training, train_labels, validation_labels, validation_labels
    '''
    partition = int(X.shape[0] * precentage)

    s = np.arange(X.shape[0])
    s = np.random.shuffle(s)
    X_shuffle = (X[s])[0]
    Y_shuffle = (Y[s])[0]

    return np.array(X_shuffle[:partition]), \
           np.array(Y_shuffle[:partition]), \
           np.array(X_shuffle[partition:]), \
           np.array(Y_shuffle[partition:])


def visualize(data, labels, pics_from_each_labels = 4):
    '''
    visualize the dataset
    @params:
        data - images
        labels - labels corresponding to the images
        pics_from_each_labels - number of pics of each labels to visualize
    '''
    pics = {}
    for j in range(10):
        pics[j] = [data[i] for i in range(len(data)) if (labels[i] == j)][0:4]

    fig, axs = plt.subplots(nrows=pics_from_each_labels, ncols = 10, figsize=(20,20)) # specifying the overall grid size
    for j in range(pics_from_each_labels):
        for i in range(10):
            axs[j, i].imshow(pics[i][j].reshape(28,28), cmap='gray')
            axs[j, i].set_title("Class {}-Image {}".format(i, j))
    plt.tight_layout()
    plt.show()

def normalize(data):
    '''
    normalize the data 
    @params:
        data - data to be normalized
    '''
    # get min and max per feature (except the first one which corresponds to the bias term)
    X_min = np.min(data[:, 1:], axis=0)
    X_max = np.max(data[:, 1:], axis=0)
    return (data[:, 1:] - X_min)/(X_max - X_min + 1e-8)

def ReLU(x):
    '''
    non linearity
    '''
    return np.maximum(0, x)

def save_model(model, file):
    '''
    save the model params to a file
    @params:
        model - model to be saved
        file - file to save to the model parameteres
    '''
    params_dict = model.get_params()
    file = open(file, "wb")
    pickle.dump(params_dict, file)
    file.close()

def upload_model(file):
    '''
    upload the model params to a dict from a file
    @params:
        file - file that contains the model inforamtion
    @returns:
        dict - model params
    '''
    file = open(file, "rb")
    dict = pickle.load(file)
    return dict