from os.path import join

import numpy as np
import torch
from ray import tune
from sklearn.utils import compute_class_weight

from src.classification.hypertuned.model.model import ConvNet


def train(config, num_epochs, train_loader, val_loader):
    """
    It trains a model for one epoch, saves the model, and reports the loss and accuracy

    :param config: a dictionary containing the hyperparameters for the model
    :param num_epochs: Number of epochs to train for
    :param train_loader: the training data
    :param val_loader: the validation set
    """

    # model
    model = ConvNet(config['out1'], config['out2'], config['l'], config['p'])

    # calculate weight for classes
    target = train_loader.dataset.dataset.dataset
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(target.targets),
                                   y=np.array(target.targets))
    weights = torch.tensor(weights, dtype=torch.float)

    # loss
    criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction="mean")

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['wd'])

    for epoch in range(1, num_epochs + 1):

        running_train_loss = 0.0
        running_accuracy = 0.0
        running_val_loss = 0.0
        total = 0

        model.train()
        for data in train_loader:
            inputs, outputs = data

            optimizer.zero_grad()

            output = model(inputs)

            train_loss = criterion(output, outputs)

            train_loss.backward()
            optimizer.step()

            running_train_loss += train_loss.item()

        train_loss_value = running_train_loss / len(train_loader.dataset)

        with torch.no_grad():
            model.eval()
            for data in val_loader:
                inputs, outputs = data

                output = model(inputs)
                val_loss = criterion(output, outputs)

                _, predicted = torch.max(output, 1)
                running_val_loss += val_loss.item()
                total += outputs.size(0)
                running_accuracy += (predicted == outputs).sum().item()

        val_loss_value = running_val_loss / len(val_loader.dataset)

        accuracy = (100 * running_accuracy / total)

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=val_loss_value, accuracy=accuracy)
