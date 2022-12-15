from os.path import join

import numpy as np
import torch
from matplotlib import pyplot as plt

MODEL_NAME = 'model'


def save_model(model, dir_config):
    """
    > Save the model to the model directory

    :param model: the model to save
    :param dir_config: This is the directory configuration object that we created earlier
    """
    path = join(dir_config.get_model_dir(), MODEL_NAME + ".pth")
    torch.save(model, path)


def load_model(dir_config):
    """
    > Load the model from the model directory

    :param dir_config: This is a class that contains the directory paths for the data, model, and logs
    :return: The model is being returned.
    """
    path = join(dir_config.get_model_dir(), MODEL_NAME + ".pth")
    model = torch.load(path)

    return model


def plot_train_valid_loss(train_loss_values, val_loss_values, dir_config):
    """
    It plots the evolution of the training and validation losses over the epochs

    :param train_loss_values: list of training losses
    :param val_loss_values: list of validation losses
    :param dir_config: the directory configuration object
    """
    plt.clf()
    plt.tight_layout()
    plt.style.use('seaborn-darkgrid')

    plt.plot(np.arange(1, len(train_loss_values) + 1, 1).astype(int), train_loss_values, label="Training loss")
    plt.plot(np.arange(1, len(val_loss_values) + 1, 1).astype(int), val_loss_values, label="Validation loss")

    plt.title("Evolution of training vs validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.legend()

    plt.savefig(join(dir_config.get_plot_dir(), MODEL_NAME + "_train_vs_valid_loss.svg"), dpi=1200)
    plt.close()


def plot_class_distribution(class_sample_count, class_names, dir_config):
    """
    It takes in a list of class sample counts and class names, and plots a bar chart of the class distribution

    :param class_sample_count: The number of samples in each class
    :param class_names: The names of the classes
    :param dir_config: This is a class that contains the paths to the directories where the data is stored
    """
    plt.clf()
    plt.tight_layout()
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(12, 7))

    classes = [i for i in range(10)]
    plt.bar(classes, class_sample_count)
    plt.xticks(np.arange(len(classes)), class_names)
    # plt.show()
    plt.savefig(dir_config.get_plot_dir() + "/class_distribution.svg", dpi=1200)
    plt.close()
