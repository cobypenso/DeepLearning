# Coby Penso, 208254128
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import *

class LogisticRegressionClassifier(object):
    
    def __init__(self, num_inputs, num_classes):
        """initialize the paramseters"""
        self.params = {}   
        self.params['w'] = np.random.randn(num_inputs, num_classes) / np.sqrt(num_classes*num_inputs) # (10*784)
        self.params['b'] = np.random.randn(1, num_classes) / np.sqrt(num_classes) # (10*1) 

    def load_params(self, dict_params):
        self.params['w'] = dict_params['w']
        self.params['b'] = dict_params['b']

    def get_params(self):
        return self.params

    def loss(self, x_batch, y_batch, lmbd):
        """ gradiens of the model params
        @params:
            x_batch - batch of images
            y_batch - corresponding labels
            lmbd - L2 regularization coeff
        @returns: grads - dict of gradients
                  loss - loss for the given model and the data
        """
        batch_size = x_batch.shape[0]
        w = self.params['w']
        pred = softmax((np.dot(x_batch, w)+self.params['b'])) #(10*1)
        
        cross_entropy_loss = np.sum(neg_log_loss(pred, y_batch)) / batch_size
        L2_regularization_cost =  (np.sum(w*w)) * (lmbd/2) / batch_size
        loss = cross_entropy_loss + L2_regularization_cost

        grads = {}
        dscores = pred
        dscores[range(batch_size), y_batch] -= 1
        dscores /= batch_size

        w_grad = np.dot(x_batch.T, dscores) + (lmbd / batch_size) * self.params['w']

        b_grad = np.sum(dscores, axis=0, keepdims=True)

        grads['W'] = w_grad
        grads['b'] = b_grad
        return grads, loss

    def predict(self, x_data):
        """ predict the y given x's using the model
        @params:
            x_data - images
        @returns: result - predictions, the labels of the images
        """
        loss_list = []
        w = self.params['w']
        dist = softmax(np.dot(x_data, w))
        result = np.argmax(dist,axis=1)
        return result

    def eval(self, x_data, y_data):
        """ evalate the model given x,y test data
        @param:
            x_data - images
            y_data - labels
        @returns: loss and accuracy
        """
        loss_list = []
        w = self.params['w']
        dist = softmax(np.dot(x_data, w))

        result = np.argmax(dist,axis=1)

        accuracy = np.sum(result == y_data)/float(len(y_data))

        loss_list = neg_log_loss(dist,y_data)
        loss = np.sum(loss_list)
        print ("Accuracy: " + str(accuracy))
        return loss, accuracy

    def train(self, x_train, y_train, x_val, y_val, lmbd = 0, batch_size = 128, learning_rate = 0.0001):
        """ train the model
        @params:
            x_train - training data
            y_train - training labels
            x_val - validation data
            y_val - validation labels
            lmbd - L2 regularization coeff

        @returns: test_loss_list, test_accu_list
        """
        num_epoches = 250
        test_loss_list, test_accu_list = [],[]

        for epoch in range(num_epoches):

            # select the random sequence of training set
            rand_indices = np.random.choice(x_train.shape[0],x_train.shape[0],replace=False)
            num_batch = int(x_train.shape[0] / batch_size)
            total_loss = 0

            # for each batch of train data
            for batch in range(num_batch):
                index = rand_indices[batch_size * batch: batch_size * (batch + 1)]
                x_batch = x_train[index]
                y_batch = y_train[index]

                # calculate the gradient w.r.t w and b
                grads, batch_loss = self.loss(x_batch, y_batch, lmbd)

                dw = grads['W']
                db = grads['b']

                total_loss += batch_loss / num_batch
                # update the paramseters with the learning rate
                self.params['w'] -= learning_rate * dw
                self.params['b'] -= learning_rate * db

            print (total_loss)
            loss, accuracy = self.eval(x_val, y_val)
            test_loss_list.append(loss)
            test_accu_list.append(accuracy)

        return test_loss_list, test_accu_list



def hyperparameter_grid_search(data, num_inputs, num_classes):
    '''
        grid search on combinations of hyper parameters to find the best one

        @return: results - list of list(batch_size, hidden_size, lmbd, lr, lr_decay, accuracy)
    '''
    results = []
    for lmbd in [0, 0.1, 0.3, 0.5, 0.7, 1]:
        for batch_size in [16, 64, 128]:
            for learning_rate in [0.0001, 0.001]:
                model  = LogisticRegressionClassifier(num_inputs, num_classes)
                loss_list, accu_list = model.train(data['train_images'], data['train_labels'], data['validation_images'], data['validation_labels'], lmbd, batch_size, learning_rate)
                results.append([loss_list[-1], accu_list[-1], lmbd, batch_size, learning_rate])
    
    return results


def training_mode(data_train):
    '''
    @params - data contains the test and validation images and labels

    @returns - the best model found after grid search

    @note - training phase, containing few steps: 
                - grid search of combinations of hyperparameters
                - save to a CSV file the table of results from step 1
                - training the best model found in step 1
                - ploting the accuracy and loss progress
    '''
    train_images = data_train[data_train.columns[1:].values].values
    train_images = normalize(train_images)
    train_labels = data_train['label'].values
    train_images, train_labels, validation_images, validation_labels = split_train_and_val(train_images, train_labels)

    data = {'train_images': train_images,
            'train_labels': train_labels,
            'validation_images': validation_images,
            'validation_labels': validation_labels}
    # setting the random seed
    np.random.seed(1024)

    # initialize the paramseters
    num_inputs = train_images.shape[1]
    num_classes = 10
    # train the model
    results = hyperparameter_grid_search(data, num_inputs, num_classes)
    # Save to file the results
    df = pd.DataFrame(results)
    df.to_csv('LRC_summary.csv', index=False, header=["Loss", "Accuracy", "lambda", "batch size", "learning rate"])

    max_accuracy = 0
    arg_max = 0
    for i in range(len(results)):
        if (results[i])[1] > max_accuracy:
            max_accuracy = (results[i])[1]
            arg_max = i
    
    print ("After grid search, the best model:")
    print ("loss - " + str(results[arg_max][0]))
    print ("accuracy - " + str(results[arg_max][1]))
    print ("lmbd - " + str(results[arg_max][2]))

    #train again with the best lmbd to plot the loss&accuracy progress
    lmbd = (results[arg_max])[2]
    batch_size = (results[arg_max])[3]
    learning_rate = (results[arg_max])[4]

    model  = LogisticRegressionClassifier(num_inputs, num_classes)
    loss_list, accu_list = model.train(data['train_images'], data['train_labels'], data['validation_images'], data['validation_labels'], lmbd, batch_size, learning_rate)
    #plot the loss progress during training
    plt.figure()
    plt.plot([i for i in range(len(loss_list))], loss_list)
    plt.show()
    #plot the accuracy osn the val data at every epoch in the training phase
    plt.figure()
    plt.plot([i for i in range(len(accu_list))], accu_list)
    plt.show()

    return model

def predict_mode(dict, data):
    '''
    @params: dict of params of the model and data.

    @note: predict the output of the data and save the results to CSV file.
    '''
    test_images = data[data.columns[0:].values].values
    model = LogisticRegressionClassifier(test_images.shape[1], 10)
    model.load_params(dict)
    # train_images = normalize(train_images)
    predictions = model.predict(data)
    #save predictions to file
    (pd.DataFrame(data = predictions)).to_csv("LRC_test.csv", index=False, header=False)
    

###########################################
################### Main ##################
###########################################

def main(args):
    data_train=pd.read_csv("train.csv")
    data_test=pd.read_csv("test.csv")

    train_images = data_train[data_train.columns[1:].values].values
    train_labels = data_train['label'].values
    visualize(train_images, train_labels, 4)
    
    if (args.mode == 'Train'):
        best_model = training_mode(data_train)
        save_model(best_model, "LRC_model.pkl")

    elif (args.mode == 'Predict'):
        dict = upload_model("LRC_model.pkl")
        predict_mode(dict, data_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian neural network example")
    parser.add_argument("--mode", default="Predict", type=str)
    args = parser.parse_args()
    main(args)