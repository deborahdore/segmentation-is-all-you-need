from os.path import join

import numpy as np
import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


def create_configuration():
    """
    It creates a dictionary with keys `out1`, `out2`, `l`, `p`, `lr`, and `wd`, and assigns each key a value that is sampled
    from a distribution
    :return: A dictionary with the keys being the parameters and the values being the values that the parameters can take.
    """
    config = {
        "out1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "out2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "p": tune.grid_search([0.0, 0.2, 0.5]),
        "lr": tune.grid_search([0.001, 0.01]),
        "wd": tune.grid_search([0.001, 0.01]),
    }
    return config


def create_reporter():
    """
    It creates a reporter that will print out the loss, accuracy, and training iteration every 20 iterations
    :return: A CLIReporter object.
    """
    reporter = CLIReporter(metric_columns=["loss", "accuracy", "training_iteration"],
                           max_report_frequency=20)
    return reporter


def create_scheduler(max_num_epochs):
    """
    It creates a scheduler that will reduce the learning rate by a factor of 2 every time the loss stops improving for 1
    epoch

    :param max_num_epochs: The maximum number of epochs to train for
    :return: A scheduler object
    """
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    return scheduler


def save_model(model, dir_config):
    """
    > Save the model to the model directory

    :param model: The model to save
    :param dir_config: This is a class that contains the directory paths for the data, model, and logs
    """
    path = join(dir_config.get_model_dir(), "hypertuned.pth")
    torch.save(model, path)
