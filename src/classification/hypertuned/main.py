from functools import partial

from src.classification.hypertuned.config.config_dir import ConfigDirectory
from src.classification.hypertuned.loader.loader import KMeansDataLoader
from src.classification.hypertuned.model.model import ConvNet
from src.classification.hypertuned.test import test
from src.classification.hypertuned.train import train
from src.classification.hypertuned.utils.util import *


def main(dir_config, num_epochs, batch_size, max_num_epochs, num_samples):
    """
    It loads the dataset, creates a configuration, creates a scheduler, creates a reporter, and then runs the
    hypertuner to find the best combination of hyperparameters

    :param dir_config: the directory configuration object
    :param num_epochs: the number of epochs to train for
    :param batch_size: The number of samples in each batch
    :param max_num_epochs: The maximum number of epochs to train for
    :param num_samples: The number of trials to run
    """
    # load dataset
    loader = KMeansDataLoader(dir_config.get_dataset_dir(), batch_size)

    # configuration
    tune_config = create_configuration()

    scheduler = create_scheduler(max_num_epochs)

    reporter = create_reporter()

    result = tune.run(
        partial(train,
                num_epochs=num_epochs,
                train_loader=loader.get_train_loader(),
                val_loader=loader.get_val_loader()),
        config=tune_config,
        local_dir=dir_config.get_logs_dir(),
        num_samples=num_samples,
        scheduler=scheduler,
        resources_per_trial={'cpu': 4},
        progress_reporter=reporter,
        checkpoint_at_end=True)

    best_trial = result.get_best_trial("loss", "min", "last")

    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    best_trained_model = ConvNet(best_trial.config['out1'],
                                 best_trial.config['out2'],
                                 best_trial.config['l'],
                                 best_trial.config['p'])

    save_model(best_trained_model, dir_config)

    test_acc = test(best_trained_model, loader.get_test_loader())
    print(f"Best trial test set accuracy: {test_acc}")


if __name__ == '__main__':
    NUM_EPOCHS = 15
    MAX_NUM_EPOCHS = 20
    BATCH_SIZE = 64
    NUM_SAMPLES = 2

    config = ConfigDirectory()
    main(config, NUM_EPOCHS, BATCH_SIZE, MAX_NUM_EPOCHS, NUM_SAMPLES)
