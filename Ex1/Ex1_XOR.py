# Coby Penso, 208254128

import numpy as np
import matplotlib.pyplot as plt

###########################################
########### Helper Functions ##############
###########################################

def train(U, W, b1, b2, X, Y, lr, epochs):
    '''
    Train the model given data X and Y the corresponding labels
        U, W, b1, b2 - the model parameters
        lr - learning rate
        epochs - number of training iterations
    '''
    loss_track = []

    for i in range(epochs):
        ###### FORWARD PASS ######
        Z = np.dot(U.T, X) + b1
        H = np.maximum(Z, [[0], [0]])
        Y_pred = np.dot(W.T, H) + b2

        ###### BACKWARD PASS ######
        W -= lr * dL_dW(Y, Y_pred, H)
        b2 -= lr * dL_db2(Y, Y_pred)
        U -= lr * dL_dU(Y, Y_pred, X, W, Z)
        b1 -= lr * dL_db1(Y, Y_pred, W, Z)

        ###### LOSS CALCULATION ######
        loss = np.power(Y_pred - Y, 2)
        loss = np.sum(loss)
        loss_track.append(loss)

        if i % 50:
            print(f"square loss after iteration - {i}: {loss}")
            
    return loss_track

def eval(U, W, b1, b2, testset, labels):
    '''
    Train the model given teset data the corresponding labels
        U, W, b1, b2 - the model parameters
    '''
    Z = np.dot(U.T, testset) + b1
    H = np.maximum(Z, [[0], [0]])
    Y_pred = np.dot(W.T, H) + b2
    
    for i in range(testset.shape[1]):
        print (f'Input: {testset[:, i]}')
        print (f'True Output: {labels[:,i]}')
        print (f'Estimated Output: {Y_pred[:,i]}')

###########################################
################### Main ##################
###########################################

def main():

    #Learning parameters
    lr = 0.01
    epochs = 1000
    
    # Data
    dataset = np.array([(0, 0), (0, 1), (1, 0), (1, 1)]).T
    labels = np.array([[1], [-1], [-1], [1]]).T
    
    # Model parameters
    U = np.random.rand(2, 2)
    W = np.random.rand(2, 1)
    b1 = np.zeros((2, dataset.shape[1]))
    b2 = np.zeros((1, dataset.shape[1]))

    loss_track = train(U, W, b1, b2, dataset, labels, lr = lr, epochs = epochs)

    plt.figure()
    plt.plot([i for i in range(epochs)], loss_track)
    plt.show()

    eval(U, W, b1, b2, dataset, labels)

###########################################
######### Derivative Calculations #########
###########################################

def dL_dW(Y, Y_pred, H):
    return np.dot(H, dL_dY(Y, Y_pred).T)

def dL_dY(Y, Y_pred):
    return -2 * (Y - Y_pred)

def dL_db2(Y, Y_pred):
    return dL_dY(Y, Y_pred)

def dL_db1(Y, Y_pred, W, Z):
    temp = dL_dY(Y, Y_pred) * W
    return np.where(Z > 0, temp, np.zeros(temp.shape))
    
def dL_dU(Y, Y_pred, X, W, Z):
    temp = dL_dY(Y, Y_pred) * W
    dL_dZ = np.where(Z > 0, temp, np.zeros(temp.shape))
    return np.dot(dL_dZ, X.T)

if __name__ == "__main__":
    main()