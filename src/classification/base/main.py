import time

from src.classification.base.config.config_dir import ConfigDirectory
from src.classification.base.loader.loader import KMeansDataLoader
from src.classification.base.test import test
from src.classification.base.train import train_with_validation
from src.classification.base.utils.util import plot_train_valid_loss


def main(dir_config, num_epochs, batch_size, learning_rate, weight_decay, already_trained=False):
    """
    It trains the model using a validation set, and then tests it on the test set

    :param dir_config: the directory configuration object
    :param num_epochs: number of epochs to train for
    :param batch_size: the number of samples in each batch
    :param learning_rate: The learning rate for the optimizer
    :param weight_decay: the weight decay parameter for the Adam optimizer
    :param already_trained: if you want to train the model from scratch, set this to False. If you want to load a
    pre-trained model, set this to True, defaults to False (optional)
    """
    # loader = ValidDataLoader(dir_config.get_dataset_dir(), batch_size)

    loader = KMeansDataLoader(dir_config.get_dataset_dir(), batch_size)

    if not already_trained:
        # training using a validation set for better results
        training_start_time = time.time()
        train_loss_values, val_loss_values = train_with_validation(dir_config, num_epochs, learning_rate,
                                                                   weight_decay, loader.get_train_loader(),
                                                                   loader.get_val_loader())
        print('Training finished, took {:.2f}s'.format(time.time() - training_start_time))

        plot_train_valid_loss(train_loss_values, val_loss_values, dir_config)

    test(dir_config, loader.get_test_loader())


if __name__ == '__main__':
    NUM_EPOCHS = 15
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.001

    main(ConfigDirectory(), NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, already_trained=False)
