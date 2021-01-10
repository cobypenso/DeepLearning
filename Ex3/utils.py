# Coby Penso, 208254128
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch

def split_train_and_val(dataset, train_percentage):
    '''
    @params:
        dataset - torch dataset to split to train and validation
        train_percentage - the percentile of the dataset to be for train and the rest for validation

    @returns:
        (train_dataset, validation_dataset)
    '''
    train_size = int(train_percentage * len(dataset))
    test_size = len(dataset) - train_size
    return torch.utils.data.random_split(dataset, [train_size, test_size])


def visualize(dataset, pics_from_each_labels = 4):
    '''
    visualize the dataset
    @params:
        data - images
        labels - labels corresponding to the images
        pics_from_each_labels - number of pics of each labels to visualize
    '''
    pics = {}
    for j in range(10):
        idx = dataset.labels==j
        pics[j] = (dataset.data[idx])[0:pics_from_each_labels]

    fig, axs = plt.subplots(nrows=pics_from_each_labels, ncols = 10, figsize=(20,20)) # specifying the overall grid size
    for j in range(pics_from_each_labels):
        for i in range(10):
            axs[j, i].axis("off")
            axs[j, i].imshow(np.transpose(pics[i][j], (1,2,0)))
            axs[j, i].set_title("Class {}-Image {}".format(i, j))
    plt.tight_layout()
    plt.show()

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

def calc_number_params(model):
    '''
        Calculate the number of params to be learned in the given model

        @param: model

        @returns: number of params in the model
    '''
    count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            count_specific_param = np.prod(param.size())
            if param.dim() > 1:
                print('Layer - ',name, ':', 'x'.join(str(x) for x in list(param.size())), '=', count_specific_param)
            else:
                print ('Later - ', name, ':', count_specific_param)
            count += count_specific_param
    return count