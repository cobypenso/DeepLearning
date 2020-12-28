# Coby Penso, 208254128
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
import argparse

class NeuralNet(object):

    def __init__(self, input_size, hidden_size, output_size, std=1e-4): 
        """
        W1: First layer weights
        b1: First layer biases
        W2: Second layer weights
        b2: Second layer biases

        Inputs:
        - input_size: The size of the inputs to the net
        - hidden_size: Size of the hidden layer
        - output_size: Size of the ouput layer, the number of the classes
        """
        self.params = {}    
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)   
        self.params['b1'] = np.zeros((1, hidden_size))    
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)   
        self.params['b2'] = np.zeros((1, output_size))
        self.params['hidden_size'] = hidden_size

    def load_params(self, dict_params):
        self.params['W1'] = dict_params['W1']
        self.params['W2'] = dict_params['W2']
        self.params['b1'] = dict_params['b1']
        self.params['b2'] = dict_params['b2']

    def get_params(self):
        return self.params

    def loss(self, X, Y, lmbd=0.0):
        """
        Calculate the loss, given the data and the expected labels

        @params:
        - X: Input data. Each X[i] is a training sample.
        - Y: Vector of training labels
        - lmbd: Regularization coeff.

        @returns
        - loss: Loss calculation
        - grads: Dict of the grads of the parameters of the net, the keys are the same as
                 the keys of self.params
        """
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2'] 
        b2 = self.params['b2']
        N= X.shape[0]
        
        # forward pass
        layer1= ReLU(np.dot(X, W1) + b1)      
        layer2 = np.dot(layer1, W2) + b2          
        out = softmax(layer2)
        
        # Compute the loss
        probs = out
        correct_logprobs = neg_log_loss(probs, Y)

        data_loss = np.sum(correct_logprobs) / N
        reg_loss = (0.5 * lmbd * np.sum(W1*W1) + 0.5 * lmbd * np.sum(W2*W2)) / N
        loss = data_loss + reg_loss
        
        # Backward pass
        grads = {}
        tmp = probs
        tmp[range(N), Y] -= 1
        tmp /= N
        
        dW2 = np.dot(layer1.T, tmp) + (lmbd / N) * W2
        db2 = np.sum(tmp, axis=0, keepdims=True)    
        dl1 = np.dot(tmp, W2.T)                     
        dl1[layer1 <= 0] = 0
        dW1 = np.dot(X.T, dl1) + (lmbd / N) * W1                   
        db1 = np.sum(dl1, axis=0, keepdims=True)
        
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2

        return loss, grads

    def train(self, X, y, X_val, y_val, learning_rate=1e-3, 
               learning_rate_decay=0.95, lmbd=1e-5, num_epochs=10, 
               batch_size=200, verbose=False):   
        """
        Train the network via SGD

        @params:
        - X: training data.
        - y: training labels
        - X_val: validation data.
        - y_val: validation labels.
        - learning_rate: learning rate for the SGD
        - learning_rate_decay: factor in which the lr will be multiplied by every given epochs
        - lmbd: regularization coeff
        - num_epochs: Number of epochs 
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.

        @returns:
            dict of - loss_history, train_acc_history, val_acc_history
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(int(num_train / batch_size), 1)
        
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for i in range(1, num_epochs * iterations_per_epoch + 1):
            # Create a random minibatch of training data and labels
            sample_index = np.random.choice(num_train, batch_size, replace=True)   
            X_batch = X[sample_index, :]          
            y_batch = y[sample_index]             
            
            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, Y=y_batch, lmbd=lmbd) 
            
            
            # Use the gradients to update the parameters of the network
            self.params['W2'] -= learning_rate * grads['W2']
            self.params['b2'] -= learning_rate * grads['b2'] 
            self.params['W1'] -= learning_rate * grads['W1']
            self.params['b1'] -= learning_rate * grads['b1']  

            if verbose and i % iterations_per_epoch == 0:    
            # Every epoch, check train and val accuracy and decay learning rate.
                loss_history.append(loss)
                epoch = i / iterations_per_epoch    
                train_acc = (self.predict(X_batch) == y_batch).mean()    
                val_acc = (self.predict(X_val) == y_val).mean()    
                train_acc_history.append(train_acc)    
                val_acc_history.append(val_acc)    
                print("epoch %d / %d: loss %f, train_acc: %f, val_acc: %f" % 
                                    (epoch, num_epochs, loss, train_acc, val_acc))
                
                # Decay learning rate
                learning_rate *= learning_rate_decay    

        return {'loss_history': loss_history,   
                'train_acc_history': train_acc_history,   
                'val_acc_history': val_acc_history}

    def predict(self, X):    
        '''
            Method to predict the labels of the data X

            @params - X the data to predict

            @returns - vector of labels corresponding to the vector of images, X[i] to y_pred[i]
        '''
        y_pred = None    
        layer1 = ReLU(np.dot(X, self.params['W1']) + self.params['b1'])    
        layer2 = np.dot(layer1, self.params['W2']) + self.params['b2']
        out = softmax(layer2)
        y_pred = np.argmax(out, axis=1)    

        return y_pred

def train_and_eval(data, lmbd, hidden_size, batch_size, lr, lr_decay):
    '''
        train the NN with the given data and hyper parameters.

        @returns: info - dict with info about the results of the trained model
                  net  - the trained neural network
    '''
    #load data (train and validation) from dict - unpack
    train_images = data['train_images']
    train_labels = data['train_labels']
    validation_images = data['validation_images']
    validation_labels = data['validation_labels']

    num_classes = 10
    input_size = train_images.shape[1]

    net = NeuralNet(input_size, hidden_size, num_classes)

    # Train the network
    info = net.train(train_images, train_labels, validation_images, validation_labels,
                num_epochs=100, batch_size=batch_size,
                learning_rate=lr, learning_rate_decay=lr_decay,
                lmbd=lmbd, verbose=True)

    # Predict on the validation set
    val_acc = (net.predict(validation_images) == validation_labels).mean()
    info['val_acc'] = val_acc

    return info, net

def hyperparameter_grid_search(data):
    '''
        grid search on combinations of hyper parameters to find the best one

        @return: results - list of list(batch_size, hidden_size, lmbd, lr, lr_decay, accuracy)
    '''
    results = []

    for lmbd in [0.0, 0.2, 0.8]:
        for hidden_size in [10, 32, 64, 128]:
            for batch_size in [16, 32, 64]:
                for lr in [1e-3, 1e-2]:
                    for lr_decay in [0.98, 0.95]:
                        info, _ = train_and_eval(data, lmbd=lmbd, hidden_size=hidden_size, batch_size=batch_size, lr = lr, lr_decay = lr_decay)
                        results.append([batch_size, hidden_size, lmbd, lr, lr_decay, info['val_acc']])
    return results

def training_mode(data):
    '''
    @params - data contains the test and validation images and labels

    @returns - the best model found after grid search

    @note - training phase, containing few steps: 
                - grid search of combinations of hyperparameters
                - save to a CSV file the table of results from step 1
                - training the best model found in step 1
                - ploting the accuracy and loss progress
    '''
    train_images = data[data.columns[0:].values].values
    train_images = normalize(train_images)
    train_labels = data['label'].values
    train_images, train_labels, validation_images, validation_labels = split_train_and_val(train_images, train_labels)
    # setting the random seed
    np.random.seed(1024)
    #pack the data
    data = {'train_images': train_images,
            'train_labels': train_labels,
            'validation_images': validation_images,
            'validation_labels': validation_labels}
    #seach for the best hyperparameters of the model
    # results = hyperparameter_grid_search(data)
    # #save the results to a CSV file
    # df = pd.DataFrame(results)
    # df.to_csv('NN_summary.csv', index=False, header=["Batch size", "hidden_size", "lambda", "lr", "lr_decay", "Accuracy"])
    # #find the best model params that achive the highest accuracy
    # max_accuracy = 0
    # arg_max = 0
    # for i in range(len(results)):
    #     if (results[i])[5] > max_accuracy:
    #         max_accuracy = (results[i])[5]
    #         arg_max = i

    #train again with the best model to plot the loss&accuracy progress
    # batch_size = (results[arg_max])[0]
    # hidden_size = results[arg_max][1]
    # lmbd = (results[arg_max])[2]
    # lr = (results[arg_max])[3]
    # lr_decay = (results[arg_max])[4]
    batch_size = 16
    hidden_size = 128
    lmbd = 0
    lr = 0.01
    lr_decay = 0.98
    
    model  = NeuralNet(train_images.shape[1], hidden_size, 10)
    info = model.train(train_images, train_labels, validation_images, validation_labels,
                num_epochs=100, batch_size=batch_size,
                learning_rate=lr, learning_rate_decay=lr_decay,
                lmbd=lmbd, verbose=True)

    plt.figure()
    plt.plot([i for i in range(len(info['loss_history']))], info['loss_history'])
    plt.show()

    plt.figure()
    plt.plot([i for i in range(len(info['val_acc_history']))], info['val_acc_history'])
    plt.show()

    return model

def predict_mode(dict, data):
    '''
    @params: dict of params of the model and data.

    @note: predict the output of the data and save the results to CSV file.
    '''
    test_images = data[data.columns[0:].values].values

    model = NeuralNet(test_images.shape[1], dict['hidden_size'], 10)
    model.load_params(dict)
    predictions = model.predict(data)
    #save predictions to file
    (pd.DataFrame(data = predictions)).to_csv("NN_test.csv", index=False, header=False)

###########################################
################### Main ##################
###########################################

def main(args):
    data_train=pd.read_csv("train.csv")
    data_test=pd.read_csv("test.csv")

    if (args.mode == 'Train'):
        best_model = training_mode(data_train)
        save_model(best_model, "NN_model.pkl")

    elif (args.mode == 'Predict'):
        dict = upload_model("NN_model.pkl")
        predict_mode(dict, data_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Network")
    parser.add_argument("--mode", default="Predict", type=str)
    args = parser.parse_args()
    main(args)